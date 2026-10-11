from __future__ import annotations
import math
from pathlib import Path
from flowphi.charts.model import Chart, NoteType

import skia

WW, WH = 1280, 720
PR = 16
NC = {
    NoteType.TAP: (10, 195, 255),
    NoteType.DRAG: (240, 237, 105),
    NoteType.HOLD: (0, 255, 255),
    NoteType.FLICK: (254, 67, 101),
    NoteType.UNKNOWN: (100, 100, 100),
}


def draw_center_rotate_rect(canvas, cx, cy, w, h, deg, paint):
    canvas.save()
    canvas.translate(cx, cy)
    canvas.rotate(deg)
    canvas.drawRect(skia.Rect.MakeXYWH(-w / 2, -h / 2, w, h), paint)
    canvas.restore()


def draw_center_rotate_round_rect(canvas, cx, cy, w, h, rx, ry, deg, paint):
    canvas.save()
    canvas.translate(cx, cy)
    canvas.rotate(deg)
    canvas.drawRoundRect(skia.Rect.MakeXYWH(-w / 2, -h / 2, w, h), rx, ry, paint)
    canvas.restore()


def make_paint(color, alpha=1.0):
    paint = skia.Paint(AntiAlias=True)
    paint.setColor(skia.Color(*(max(0, min(255, round(v))) for v in color), round(max(0, min(1, alpha)) * 255)))
    return paint


class ChartRenderer:
    def __init__(s, chart: Chart, width=WW, height=WH, line_width=.0075):
        s.chart = chart
        s.width, s.height = width, height
        s.line_width = line_width
        s.sc = (width / chart.width, height / chart.height)
        s.lines = sorted((line for line in chart.lines if line.attach_ui is None), key=lambda line: line.z_order)
        s.font_manager = skia.FontMgr.RefDefault()
        s.typeface = s.font_manager.matchFamilyStyle(None, skia.FontStyle.Normal())
        s.font = skia.Font(s.typeface, height * 0.045)
        s.fallback_fonts = {}
        s.textures = {}
        for line in s.lines:
            if line.texture != 'line.png':
                s.texture(line.texture)

    def point(s, pos):
        return pos.real * s.sc[0], pos.imag * s.sc[1]

    def font_for(s, char):
        if s.font.unicharToGlyph(ord(char)):
            return s.font
        if char not in s.fallback_fonts:
            face = s.font_manager.matchFamilyStyleCharacter("", skia.FontStyle.Normal(), ["zh", "ja"], ord(char))
            s.fallback_fonts[char] = skia.Font(face or s.typeface, s.font.getSize())
        return s.fallback_fonts[char]

    def draw_text(s, canvas, text, anchor, paint):
        rows = text.split("\n")
        row_height = s.font.getSize() * 1.2
        for index, row in enumerate(rows):
            runs = []
            for char in row:
                font = s.font_for(char)
                if runs and runs[-1][1] is font:
                    runs[-1] = runs[-1][0] + char, font
                else:
                    runs.append((char, font))
            width = sum(font.measureText(part) for part, font in runs)
            x = -width * anchor[0]
            y = (index - (len(rows) - 1) * (1 - anchor[1])) * row_height + s.font.getSize() * (anchor[1] - 0.2)
            for part, font in runs:
                canvas.drawString(part, x, y, font, paint)
                x += font.measureText(part)

    def texture(s, name):
        if name not in s.textures:
            image = None
            if s.chart.source is not None:
                root = s.chart.source.resolve().parent
                path = (root / name).resolve()
                if path.is_relative_to(root) and path.is_file():
                    try:
                        image = skia.Image.open(str(path))
                    except (RuntimeError, ValueError, OSError):
                        pass
            s.textures[name] = image
            if image is None:
                warning = f"Cannot load line texture {name!r}; displaying a plain line."
                if warning not in s.chart.warnings:
                    s.chart.warnings.append(warning)
        return s.textures[name]

    def draw_line(s, canvas, line, seconds):
        alpha = max(0.0, min(1.0, line.opacity @ seconds))
        if alpha == 0 or line.attach_ui is not None:
            return
        position = line.position @ seconds
        angle = math.degrees(line.angle @ seconds)
        sx, sy = line.scale_x @ seconds, line.scale_y @ seconds
        if not all(math.isfinite(v) for v in (position.real, position.imag, angle, sx, sy)):
            return
        color = line.color @ seconds
        paint = make_paint(color, alpha)
        canvas.save()
        canvas.translate(*s.point(position))
        canvas.rotate(angle)
        canvas.scale(sx, sy)
        text = line.text @ seconds
        if text is not None:
            s.draw_text(canvas, text, line.anchor, paint)
        else:
            texture = s.texture(line.texture) if line.texture != "line.png" else None
            if texture is not None:
                # 贴图宽高使用相同的像素倍率，避免在 16:9 屏幕上被压扁
                scale = s.width / 1350
                w, h = texture.width() * scale, texture.height() * scale
                paint.setColorFilter(skia.ColorFilters.Blend(make_paint(color).getColor(), skia.BlendMode.kModulate))
                canvas.drawImageRect(
                    texture, skia.Rect.MakeXYWH(-w * line.anchor[0], -h * (1 - line.anchor[1]), w, h), paint=paint
                )
            else:
                # 简洁预览仍使用长判定线，缩放事件同时作用于长度和线宽
                length = s.height * 5.76
                canvas.drawRect(skia.Rect.MakeXYWH(-length, -s.height * s.line_width / 2, length * 2, s.height * s.line_width), paint)
        canvas.restore()

    def draw(s, canvas, seconds):
        for line in s.lines:
            s.draw_line(canvas, line, seconds)
            floor = line.floor @ seconds
            angle = math.degrees(line.angle @ seconds)
            for visual in line.visual_notes:
                state = line.note_state(seconds, visual, floor)
                if state is None:
                    continue
                hx, hy = s.point(state.head)
                tx, ty = s.point(state.tail)
                note_width = s.width * 0.1234375 * abs(state.width)
                head_height = s.width * 0.1234375 * 0.14 * abs(state.height)
                if not all(math.isfinite(v) for v in (hx, hy, tx, ty, note_width, head_height)):
                    continue
                margin = max(note_width, head_height)
                # 用完整的 Hold 包围盒裁剪，头在屏幕外时身体和尾部仍可能可见
                if (
                    max(hx, tx) < -margin
                    or min(hx, tx) > s.width + margin
                    or max(hy, ty) < -margin
                    or min(hy, ty) > s.height + margin
                ):
                    continue
                color = tuple(a * b / 255 for a, b in zip(NC[visual.note.type], visual.tint))
                paint = make_paint(color, state.alpha)
                rotation = angle + (0 if visual.above else 180)
                if visual.note.type == NoteType.HOLD:
                    length = math.hypot(tx - hx, ty - hy)
                    if length > 0:
                        body = make_paint(color, state.alpha * 0.4)
                        body_angle = math.degrees(math.atan2(ty - hy, tx - hx)) - 90
                        draw_center_rotate_rect(
                            canvas, (hx + tx) / 2, (hy + ty) / 2, note_width, length, body_angle, body
                        )
                        draw_center_rotate_round_rect(
                            canvas, tx, ty, note_width, head_height, head_height / 4, head_height / 4, rotation, paint
                        )
                if state.draw_head:
                    draw_center_rotate_round_rect(
                        canvas, hx, hy, note_width, head_height, head_height / 4, head_height / 4, rotation, paint
                    )

