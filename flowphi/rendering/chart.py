import bisect
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import wave

import numpy as np
import skia

from flowphi.charts.media import read_info
from flowphi.charts.model import NoteType
from flowphi.rendering.preview import ChartRenderer, make_paint

# Drawing order/UI proportions informed by sim-phi/src/index.ts.
# Atlas layout and Hold effects informed by Phira prpr/src/core/{resource,note}.rs.
# Assets remain in the user's resource directory; this module does not bundle them.


def background_image(path, width, height, dim, blur):
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    canvas.clear(skia.Color(95, 108, 124))
    if path:
        image = skia.Image.open(str(path))
        if image is None:
            raise ValueError(f'Cannot read illustration: {path}')
        scale = max(width / image.width(), height / image.height())
        sw, sh = width / scale, height / scale
        source = skia.Rect.MakeXYWH((image.width() - sw) / 2, (image.height() - sh) / 2, sw, sh)
        canvas.drawImageRect(image, source, skia.Rect.MakeWH(width, height), skia.SamplingOptions(skia.FilterMode.kLinear))
    image = surface.makeImageSnapshot()
    canvas.clear(skia.ColorBLACK)
    paint = skia.Paint(ImageFilter=skia.ImageFilters.Blur(blur, blur, skia.TileMode.kClamp))
    canvas.drawImage(image, 0, 0, paint=paint)
    canvas.drawColor(skia.ColorSetARGB(round(255 * dim), 0, 0, 0))
    return surface.makeImageSnapshot()


class PlayfieldRenderer(ChartRenderer):
    def __init__(self, chart, width, height, resources, title='', level='', duration=1.):
        super().__init__(chart, width, height, line_width=.005)
        self.ui_lines = {line.attach_ui: line for line in chart.lines if line.attach_ui is not None}
        self.block_renderer = None
        if chart.block_areas:
            from flowphi.rendering.blocks import BlockRenderer
            self.block_renderer = BlockRenderer(chart.block_areas, width, height)
        chart.warnings[:] = [w for w in chart.warnings if w != 'attachUI lines are not displayed in the simplified preview.']
        resources = Path(resources)
        pack = resources / 'respack' if (resources / 'respack').is_dir() else resources
        self.info = read_info(pack / 'info.yml')
        self.sprites = {}
        for name in ('click', 'drag', 'hold', 'flick', 'hit_fx'):
            for suffix in (('', '_mh') if name != 'hit_fx' else ('',)):
                path = pack / f'{name}{suffix}.png'
                if not path.is_file():
                    raise ValueError(f'Missing Phira resource: {path}; use --resources')
                self.sprites[name + suffix] = skia.Image.open(str(path))
                if self.sprites[name + suffix] is None:
                    raise ValueError(f'Invalid skin image: {path}')
        self.atlases = [self.info.get('holdatlas', [50, 50]), self.info.get('holdatlasmh', [130, 130])]
        for suffix, atlas in zip(('', '_mh'), self.atlases):
            if not isinstance(atlas, list) or len(atlas) != 2 or not all(isinstance(v, int) and v > 0 for v in atlas):
                raise ValueError('holdAtlas must contain two positive integers')
            if sum(atlas) >= self.sprites['hold' + suffix].height():
                raise ValueError('holdAtlas leaves no Hold body')
        self.hold_bodies = {}
        if self.info.get('holdrepeat', False):
            for suffix, (tail, head) in zip(('', '_mh'), self.atlases):
                texture = self.sprites['hold' + suffix]
                self.hold_bodies['hold' + suffix] = texture.makeSubset(skia.IRect.MakeXYWH(0, tail, texture.width(), texture.height() - tail - head))
        self.fx_grid = self.info.get('hitfx', [8, 7])
        if not isinstance(self.fx_grid, list) or len(self.fx_grid) != 2 or not all(isinstance(v, int) and v > 0 for v in self.fx_grid):
            raise ValueError('hitFx must contain two positive integers')
        fx = self.sprites['hit_fx']
        if fx.width() % self.fx_grid[0] or fx.height() % self.fx_grid[1]:
            raise ValueError('hitFx grid does not divide the image')
        self.fx_duration = float(self.info.get('hitfxduration', .5))
        self.fx_scale = float(self.info.get('hitfxscale', 1.))
        color = self.info.get('colorperfect', 0xe1ffec9f)
        if not isinstance(color, int) or not 0 <= color <= 0xffffffff:
            raise ValueError('colorPerfect must be an RGB/ARGB integer')
        self.perfect = tuple((color >> shift) & 255 for shift in (16, 8, 0))
        if not all(math.isfinite(v) and v > 0 for v in (self.fx_duration, self.fx_scale)):
            raise ValueError('Invalid hitFx duration or scale')
        self.title, self.level, self.duration = title, level, duration
        font = next((p for p in (resources / 'phigros.ttf', resources / 'font.ttf') if p.is_file()), None)
        self.ui_face = (skia.Typeface.MakeFromFile(str(font)) if font else None) or self.typeface
        self.visuals = [(line, visual) for line in chart.lines for visual in line.visual_notes]
        real = [(line, v) for line, v in self.visuals if not v.is_fake]
        counts = Counter(round(v.note.seconds, 6) for _, v in real)
        self.multi = {id(v) for _, v in real if counts[round(v.note.seconds, 6)] > 1}
        # Autoplay counts a Hold at its tail; sound and repeated particles begin at its head.
        self.judged = sorted(v.note.seconds + (v.note.hold if v.note.type == NoteType.HOLD else 0.) for _, v in real)
        self.hits = sorted((v.note.seconds + chart.offset, int(v.note.type)) for _, v in real)
        self.effects = []
        for line, visual in real:
            note = visual.note
            times = [note.seconds]
            if note.type == NoteType.HOLD:
                times += [note.seconds + i * .15 for i in range(1, math.ceil(note.hold / .15))]
            for t in times:
                point = line.note_state(t, visual)
                if point is not None:
                    self.effects.append((t, point.head, math.degrees(line.angle @ t)))
        self.effects.sort(key=lambda event: event[0])
        order = {NoteType.HOLD: 0, NoteType.DRAG: 1, NoteType.TAP: 2, NoteType.FLICK: 3}
        self.visuals.sort(key=lambda item: (order.get(item[1].note.type, 4), -item[1].note.seconds))

    def stats(self, seconds):
        combo = bisect.bisect_right(self.judged, seconds)
        return combo, round(1_000_000 * combo / len(self.judged)) if self.judged else 0

    def sprite(self, canvas, name, rect, alpha=1., tint=(255, 255, 255), source=None):
        image = self.sprites[name]
        paint = make_paint((255, 255, 255), alpha)
        if tint != (255, 255, 255):
            paint.setColorFilter(skia.ColorFilters.Blend(make_paint(tint).getColor(), skia.BlendMode.kModulate))
        canvas.drawImageRect(image, source or skia.Rect.MakeWH(image.width(), image.height()), rect,
                             skia.SamplingOptions(skia.FilterMode.kLinear), paint)

    def draw_note(self, canvas, line, visual, seconds):
        note = visual.note
        if not visual.is_fake and seconds >= note.seconds + (note.hold if note.type == NoteType.HOLD else 0.):
            return
        state = line.note_state(seconds, visual)
        if state is None:
            return
        hx, hy = self.point(state.head)
        tx, ty = self.point(state.tail)
        scale = self.width / 8080
        name = {NoteType.TAP: 'click', NoteType.DRAG: 'drag', NoteType.HOLD: 'hold', NoteType.FLICK: 'flick'}.get(note.type)
        if name is None:
            return
        multi = id(visual) in self.multi
        name += '_mh' if multi else ''
        texture = self.sprites[name]
        width = texture.width() * scale * abs(state.width)
        height = texture.height() * scale * abs(state.height)
        if not all(math.isfinite(v) for v in (hx, hy, tx, ty, width, height)):
            return
        margin = max(width, self.width * .06)
        if max(hx, tx) < -margin or min(hx, tx) > self.width + margin or max(hy, ty) < -margin or min(hy, ty) > self.height + margin:
            return
        canvas.save()
        canvas.translate(hx, hy)
        angle = math.degrees(line.angle @ seconds) + (0 if visual.above else 180)
        canvas.rotate(angle)
        if note.type == NoteType.HOLD:
            length = math.hypot(tx - hx, ty - hy)
            # Local negative Y points away from the line, for both above and below notes.
            radians = math.radians(angle)
            if (tx - hx) * math.sin(radians) - (ty - hy) * math.cos(radians) < 0:
                canvas.scale(1, -1)
            tail, head = self.atlases[multi]
            body = texture.height() - tail - head
            if name in self.hold_bodies and width > 0:
                canvas.save()
                canvas.translate(-width / 2, -length)
                factor = width / texture.width()
                canvas.scale(factor, factor)
                paint = make_paint((255, 255, 255), state.alpha)
                paint.setShader(self.hold_bodies[name].makeShader(skia.TileMode.kClamp, skia.TileMode.kRepeat,
                                                               skia.SamplingOptions(skia.FilterMode.kLinear)))
                paint.setColorFilter(skia.ColorFilters.Blend(make_paint(visual.tint).getColor(), skia.BlendMode.kModulate))
                canvas.drawRect(skia.Rect.MakeWH(texture.width(), length / factor), paint)
                canvas.restore()
            else:
                self.sprite(canvas, name, skia.Rect.MakeXYWH(-width / 2, -length, width, length), state.alpha, visual.tint,
                            skia.Rect.MakeXYWH(0, tail, texture.width(), body))
            compact = .5 if self.info.get('holdcompact', False) else 1.
            self.sprite(canvas, name, skia.Rect.MakeXYWH(-width / 2, -length - tail * scale * compact, width, tail * scale),
                        state.alpha, visual.tint, skia.Rect.MakeWH(texture.width(), tail))
            if state.draw_head or self.info.get('holdkeephead', False):
                self.sprite(canvas, name, skia.Rect.MakeXYWH(-width / 2, -head * scale * (1 - compact), width, head * scale), state.alpha, visual.tint,
                            skia.Rect.MakeXYWH(0, texture.height() - head, texture.width(), head))
        else:
            self.sprite(canvas, name, skia.Rect.MakeXYWH(-width / 2, -height / 2, width, height), state.alpha, visual.tint)
        canvas.restore()

    def draw_effects(self, canvas, seconds):
        begin = bisect.bisect_right(self.effects, seconds - self.fx_duration, key=lambda e: e[0])
        end = bisect.bisect_right(self.effects, seconds, key=lambda e: e[0])
        columns, rows = self.fx_grid
        texture = self.sprites['hit_fx']
        fw, fh = texture.width() / columns, texture.height() / rows
        for index in range(begin, end):
            t, pos, angle = self.effects[index]
            u = (seconds - t) / self.fx_duration
            frame = min(columns * rows - 1, int(u * columns * rows))
            size = self.width * .14 * self.fx_scale
            canvas.save()
            canvas.translate(*self.point(pos))
            if self.info.get('hitfxrotate', False):
                canvas.rotate(angle)
            tint = self.perfect if self.info.get('hitfxtinted', True) else (255, 255, 255)
            self.sprite(canvas, 'hit_fx', skia.Rect.MakeXYWH(-size / 2, -size / 2, size, size), 1 - u * u, tint,
                        skia.Rect.MakeXYWH(frame % columns * fw, frame // columns * fh, fw, fh))
            if not self.info.get('hideparticles', False):
                paint = make_paint(self.perfect, (1 - u) ** 2)
                distance = self.width * .075 * (1 - (1 - u) ** 3)
                side = self.width * .006 * (1 - u)
                for particle in range(4):
                    direction = index * 2.39996 + particle * math.pi / 2
                    x, y = math.cos(direction) * distance, math.sin(direction) * distance
                    canvas.drawRect(skia.Rect.MakeXYWH(x - side / 2, y - side / 2, side, side), paint)
            canvas.restore()

    def ui_begin(self, canvas, name, seconds, x, y):
        canvas.save()
        line = self.ui_lines.get(name)
        if line is not None:
            px, py = self.point(line.position @ seconds)
            canvas.translate(x + px - self.width / 2, y + py - self.height / 2)
            canvas.rotate(math.degrees(line.angle @ seconds))
            canvas.scale(line.scale_x @ seconds, line.scale_y @ seconds)
            canvas.translate(-x, -y)
            paint = make_paint((255, 255, 255), line.opacity @ seconds)
            paint.setColorFilter(skia.ColorFilters.Blend(make_paint(line.color @ seconds).getColor(), skia.BlendMode.kModulate))
            canvas.saveLayer(paint=paint)
        return line is not None

    @staticmethod
    def ui_end(canvas, attached):
        if attached:
            canvas.restore()
        canvas.restore()

    def ui_text(self, canvas, text, x, y, size, align=0., max_width=None, element='', seconds=0.):
        attached = self.ui_begin(canvas, element, seconds, x, y)
        font = skia.Font(self.ui_face, size)
        # Retain the preview's CJK fallback, including mixed-language song titles.
        runs = []
        for char in str(text):
            face = self.ui_face if font.unicharToGlyph(ord(char)) else self.font_for(char).refTypeface()
            if runs and runs[-1][1] == face:
                runs[-1] = runs[-1][0] + char, face
            else:
                runs.append((char, face))
        widths = [skia.Font(face, size).measureText(part) for part, face in runs]
        scale = min(1., max_width / sum(widths)) if max_width and sum(widths) else 1.
        x -= sum(widths) * scale * align
        for (part, face), width in zip(runs, widths):
            canvas.drawString(part, x, y, skia.Font(face, size * scale), make_paint((255, 255, 255)))
            x += width * scale
        self.ui_end(canvas, attached)

    def draw(self, canvas, seconds):
        for line in self.lines:
            self.draw_line(canvas, line, seconds)
        for line, visual in self.visuals:
            self.draw_note(canvas, line, visual, seconds)
        self.draw_effects(canvas, seconds)
        w, h = self.width, self.height
        unit = min(w / 18.75, h / 14)
        combo, score = self.stats(seconds)
        progress = max(0., min(1., (seconds + self.chart.offset) / self.duration))
        attached = self.ui_begin(canvas, 'bar', seconds, w / 2, 0)
        canvas.drawRect(skia.Rect.MakeWH(w * progress, max(1., h * .004)), make_paint((255, 255, 255), .85))
        self.ui_end(canvas, attached)
        attached = self.ui_begin(canvas, 'pause', seconds, .85 * unit, 1.06 * unit)
        for x in (.7 * unit, .9 * unit):
            canvas.drawRect(skia.Rect.MakeXYWH(x, .8 * unit, .09 * unit, .52 * unit), make_paint((255, 255, 255)))
        self.ui_end(canvas, attached)
        self.ui_text(canvas, f'{score:07d}', w - .65 * unit, 1.375 * unit, .95 * unit, 1., element='score', seconds=seconds)
        if combo > 2:
            self.ui_text(canvas, combo, w / 2, 1.375 * unit, 1.32 * unit, .5, element='combonumber', seconds=seconds)
            self.ui_text(canvas, 'AUTOPLAY', w / 2, 2.05 * unit, .53 * unit, .5, element='combo', seconds=seconds)
        self.ui_text(canvas, self.title, .65 * unit, h - .66 * unit, .63 * unit, max_width=w * .7, element='name', seconds=seconds)
        self.ui_text(canvas, self.level, w - .65 * unit, h - .66 * unit, .63 * unit, 1., w * .23, element='level', seconds=seconds)
        # Capture the complete game screen, including attached HUD elements,
        # so notes and UI receive the same block-area material treatment.
        if self.block_renderer:
            self.block_renderer.draw(canvas, seconds)


def mix_audio(music, resources, hits, start, duration, volume, output):
    """Mix at sample precision, including the tails of hits just before the clip."""
    rate = 48000
    def decode(path, begin=None, seconds=None):
        command = ['ffmpeg', '-v', 'error']
        if begin is not None:
            command += ['-ss', str(begin)]
        command += ['-i', str(path)]
        if seconds is not None:
            command += ['-t', str(seconds), '-af', f'aresample={rate}:async=1:first_pts=0,apad']
        raw = subprocess.check_output(command + ['-vn', '-ac', '2', '-ar', str(rate), '-f', 'f32le', '-'])
        return np.frombuffer(raw, dtype='<f4').reshape(-1, 2)
    samples = round(duration * rate)
    original = decode(music, start, duration)
    if len(original) < samples:
        raise ValueError('Music decoder returned fewer samples than the clip needs')
    mixed = original[:samples].copy()
    sounds = {kind: decode(Path(resources) / f'{name}.ogg') for kind, name in
              ((NoteType.TAP, 'click'), (NoteType.DRAG, 'drag'), (NoteType.FLICK, 'flick'))} if volume else {}
    count = 0
    for time, kind in hits:
        sound = sounds.get(NoteType.TAP if kind == NoteType.HOLD else kind)
        if sound is None:
            continue
        offset = round((time - start) * rate)
        a, b = max(0, offset), min(samples, offset + len(sound))
        if a < b:
            mixed[a:b] += sound[a - offset:b - offset] * volume
            count += 1
    peak = float(np.max(np.abs(mixed)))
    gain = min(1., .98 / peak) if peak else 1.
    pcm = np.rint(mixed * gain * 32767).astype('<i2')
    with wave.open(str(output), 'wb') as stream:
        stream.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
        stream.writeframes(pcm.tobytes())
    return dict(sample_rate=rate, samples=samples, hit_sounds=count, gain=gain, peak_before_gain=peak)
