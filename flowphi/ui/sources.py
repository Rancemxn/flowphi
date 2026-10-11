from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from flowphi.charts.loader import load_chart
from flowphi.charts.media import find_media

from flowphi.config import ROOT, DEFAULT_CACHE


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@dataclass
class ChartSource:
    chart: object
    chart_path: Path
    reference: dict

    @property
    def duration(self):
        return max((n.seconds + n.hold for line in self.chart.lines for n in line.notes), default=0.) + self.chart.offset + 2.


def load_source(path, cache=DEFAULT_CACHE, *, expected=None, audio=None, background=None):
    from flowphi.rendering.pipeline import unpack
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'找不到谱面：{path}')
    container_digest = file_digest(path)
    destination = Path(cache) / 'charts' / container_digest
    # Unpack verifies entry paths and unpacked size. Cache only our successful
    # extraction; a cancelled or interrupted extraction is retried.
    ready = destination.parent / (container_digest + '.ready.json')
    if ready.is_file():
        data = json.loads(ready.read_text(encoding='utf-8'))
        def cached_file(name):
            target = (destination / name).resolve()
            if not target.is_relative_to(destination.resolve()) or not target.is_file():
                raise ValueError('谱面缓存不完整，请清理该谱面的缓存后重新打开')
            return target
        chart_path = cached_file(data['chart'])
        music = cached_file(data['audio']) if data.get('audio') else None
        picture = cached_file(data['background']) if data.get('background') else None
    elif path.suffix.lower() in ('.zip', '.pez'):
        chart_path, music, picture = unpack(path, destination)
        data = dict(chart=str(chart_path.relative_to(destination)),
                    audio=str(music.relative_to(destination)) if music else None,
                    background=str(picture.relative_to(destination)) if picture else None)
        from flowphi.ui.model import write_json_atomic
        write_json_atomic(ready, data)
    else:
        chart_path = path
        music = picture = None
    digest = file_digest(chart_path)
    if expected and expected.get('sha256') and digest != expected['sha256']:
        raise ValueError('所选谱面与工程记录不一致请重新选择原谱面；不会将旧修改套用到其他音符上')
    content = chart_path.read_text(encoding='utf-8-sig')
    chart = load_chart(content, source=chart_path)
    media_root = destination if path.suffix.lower() in ('.zip', '.pez') else None
    metadata = json.loads(content).get('META', {}) if chart.format == 'rpe' else None
    found_music, found_picture = find_media(chart_path, media_root, metadata=metadata)
    music, picture = found_music or music, found_picture or picture
    if not any(line.notes for line in chart.lines):
        raise ValueError('谱面中没有可编辑的音符')
    if not math.isfinite(chart.offset) or any(not math.isfinite(n.seconds) or not math.isfinite(n.hold) or n.hold < 0
                                             for line in chart.lines for n in line.notes):
        raise ValueError('谱面包含无效的音符时间或音乐偏移')
    reference = dict(path=str(path), sha256=digest, container_sha256=container_digest,
                     note_count=sum(len(line.notes) for line in chart.lines),
                     width=chart.width, height=chart.height, offset=chart.offset)
    for key, value in (('audio', audio or music), ('background', background or picture)):
        if value:
            reference[key] = str(Path(value).resolve())
    return ChartSource(chart, chart_path, reference)


def create_demo(cache=DEFAULT_CACHE):
    """A small original chart and a metronome, requiring no downloaded assets."""
    import array
    import math
    import wave
    from flowphi.ui.model import write_json_atomic
    destination = Path(cache) / 'demo'
    destination.mkdir(parents=True, exist_ok=True)
    chart_path = destination / 'chart.json'
    notes = []
    for i in range(16):
        notes.append(dict(type=1, time=(1 + i * .5) * 64, holdTime=0,
                          positionX=(-1 if i % 2 == 0 else 1) * (2.3 + .4 * math.sin(i)),
                          speed=1., floorPosition=1 + i * .5))
    notes += [dict(type=3, time=9 * 64, holdTime=1.5 * 64, positionX=-2.5, speed=1., floorPosition=9.),
              dict(type=1, time=9.5 * 64, holdTime=0, positionX=2.5, speed=1., floorPosition=9.5),
              dict(type=1, time=10 * 64, holdTime=0, positionX=2., speed=1., floorPosition=10.),
              dict(type=2, time=11.5 * 64, holdTime=0, positionX=-2., speed=1., floorPosition=11.5),
              dict(type=4, time=12 * 64, holdTime=0, positionX=2., speed=1., floorPosition=12.),
              dict(type=1, time=13 * 64, holdTime=0, positionX=-2., speed=1., floorPosition=13.),
              dict(type=1, time=13 * 64, holdTime=0, positionX=2., speed=1., floorPosition=13.)]
    data = dict(formatVersion=3, offset=0., judgeLineList=[dict(bpm=120,
        notesAbove=notes, notesBelow=[], speedEvents=[dict(startTime=0, endTime=99999, value=1., floorPosition=0.)],
        judgeLineMoveEvents=[dict(startTime=0, endTime=99999, start=.5, end=.5, start2=.5, end2=.5)],
        judgeLineDisappearEvents=[dict(startTime=0, endTime=99999, start=1., end=1.)],
        judgeLineRotateEvents=[dict(startTime=0, endTime=99999, start=0., end=0.)])])
    write_json_atomic(chart_path, data)
    audio = destination / 'metronome.wav'
    if not audio.is_file():
        rate, duration = 24000, 15
        samples = array.array('h', [0]) * (rate * duration)
        for beat in range(1, duration * 2):
            start = round(beat * .5 * rate)
            for j in range(round(.035 * rate)):
                samples[start + j] = round(6000 * math.exp(-j / (rate * .009)) * math.sin(2 * math.pi * 880 * j / rate))
        with wave.open(str(audio), 'wb') as stream:
            stream.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
            stream.writeframes(samples.tobytes())
    return chart_path
