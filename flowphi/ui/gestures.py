from __future__ import annotations

import math

from flowphi.charts.model import NoteType
from flowphi.motion import point_at


def release_points(contact, end, reshape='trim'):
    """Shorten at the new UP, or extend the final dwell; optionally retime."""
    if reshape not in ('trim', 'scale'):
        raise ValueError('未知的轨迹时间调整方式')
    start, points = contact['start'], contact['points']
    last = round(end * 1000) / 1000 - .001
    last = round(last, 3)
    if last < start - 1e-9:
        raise ValueError('每段动作至少按住 1 毫秒，抬起时间必须晚于按下时间')
    if reshape == 'scale':
        old_last = points[-1][0]
        if old_last <= start:
            return [list(points[0])]
        # Scale the contact lifetime, retaining a proportional terminal dwell.
        if last <= start:
            raise ValueError('时间太短，无法保留滑动；请增加按住时长')
        result = []
        old_span = max(old_last - start, contact['end'] - .001 - start)
        for t, x, y in points:
            when = round((start + (t - start) * (last - start) / old_span) * 1000) / 1000
            if result and when <= result[-1][0]:
                raise ValueError('加快后有控制点落在同一毫秒，请先减少控制点或增加按住时长')
            result.append([when, x, y])
        return result
    if end >= contact['end'] or points[-1][0] <= last:
        return [list(p) for p in points]
    result = [list(p) for p in points if p[0] < last - 1e-9]
    result.append([max(start, last), *point_at(points, last)])
    return result


def flick_contacts(document, chart, indices):
    notes = [(line, n) for line in chart.lines for n in line.notes]
    result = []
    for i in document.sorted_ids(indices):
        c = document.contact(i)
        flicks = []
        for nid in c['note_ids']:
            if not 0 <= nid < len(notes):
                raise ValueError('音符编号与关联谱面不匹配')
            line, note = notes[nid]
            if note.type == NoteType.FLICK:
                flicks.append((nid, line, c.get('judgement_times', {}).get(str(nid), note.seconds)))
        if flicks:
            result.append((i, c, sorted(flicks, key=lambda row: row[2])))
    return result


def flick_defaults(contact, flicks, chart):
    _, line, beat = flicks[0]
    # Find the moving part around judgement, excluding stationary lead/tail.
    points = contact['points']
    moving = [(a[0], b[0]) for a, b in zip(points, points[1:]) if a[1:] != b[1:]]
    local = [pair for pair in moving if pair[0] <= beat + .045 and pair[1] >= beat - .025]
    segments = local or moving
    if segments:
        start, finish = segments[0][0], segments[-1][1]
    else:
        start, finish = beat - .025, beat + .045
    a, b = point_at(contact['points'], start), point_at(contact['points'], finish)
    dx, dy = b[0] - a[0], (b[1] - a[1]) * chart.height / chart.width
    angle = -math.degrees(math.atan2(dy, dx))
    relative = (angle + math.degrees(line.angle @ beat) + 180) % 360 - 180
    direction = next((name for name, degrees in (('up', 90), ('down', -90), ('right', 0), ('left', 180))
                      if abs((relative - degrees + 180) % 360 - 180) < 1.), 'custom')
    duration = max(1, round((finish - start) * 1000))
    return dict(direction=direction, angle=relative, length=max(.1, math.hypot(dx, dy) * 100),
                duration_ms=duration, lead_ms=min(duration, max(0, round((beat - start) * 1000))),
                release_ms=max(1, round((contact['end'] - finish) * 1000)) if len(contact['note_ids']) == 1 else 1)


def flick_path(contact, flicks, chart, *, direction, angle=0., angle_space='line', length=.16,
               duration_ms=70, lead_ms=25, release_ms=1, note_id=None):
    if direction not in ('up', 'down', 'left', 'right', 'custom') or angle_space not in ('line', 'screen'):
        raise ValueError('Flick 方向无效')
    if not all(isinstance(v, (int, float)) and math.isfinite(v)
               for v in (angle, length, duration_ms, lead_ms, release_ms)):
        raise ValueError('Flick 参数必须是有限数值')
    duration, lead, release = round(duration_ms), round(lead_ms), round(release_ms)
    if not 1 <= duration <= 10000 or not 0 <= lead <= duration or not 1 <= release <= 60000 or not 0 < length <= 2:
        raise ValueError('滑动时长需为 1–10000 ms；提前开始的时间不能超过滑动时长；滑完后停留需为 1–60000 ms；长度须大于 0 且不超过屏幕宽度的 200%')
    chosen = [row for row in flicks if note_id is None or row[0] == note_id]
    if not chosen:
        return None
    strokes = []
    for nid, line, beat in chosen:
        begin = round(beat * 1000) - lead
        finish = begin + duration
        if strokes and begin <= strokes[-1][1]:
            raise ValueError('同一段动作中的 Flick 滑动时段重叠或紧贴，请缩短时长，或选择其中一个 Flick 单独修改')
        degrees = dict(up=90., down=-90., right=0., left=180.).get(direction, angle)
        theta = (0. if direction == 'custom' and angle_space == 'screen' else line.angle @ beat) - math.radians(degrees)
        if not math.isfinite(theta):
            raise ValueError(f'音符 {nid} 的判定线角度无效')
        dx, dy = math.cos(theta) * length, math.sin(theta) * length * chart.width / chart.height
        anchor = point_at(contact['points'], beat)
        # Keep the original judgement position even when note time rounds to
        # a neighboring millisecond in the exported touch sequence.
        u = max(0., min(1., (beat * 1000 - begin) / duration))
        first = [begin / 1000, anchor[0] - dx * u, anchor[1] - dy * u]
        last = [finish / 1000, first[1] + dx, first[2] + dy]
        for p in (first, last):
            if not all(-1e-12 <= v <= 1 + 1e-12 for v in p[1:]):
                raise ValueError(f'音符 {nid} 的滑动超出屏幕请缩短长度、调整方向或先移动判定落点')
            p[1:] = [min(1., max(0., v)) for v in p[1:]]
        strokes.append((begin, finish, first, last))
    shared = len(contact['note_ids']) > 1
    if not shared:
        begin, finish, first, last = strokes[0]
        return dict(start=begin / 1000, end=(finish + release) / 1000, points=[first, last])
    # Shared Hold/Drag contacts keep their other samples and release lifetime.
    # Full geometry/coverage validation still applies to all associated notes.
    points, cursor = [], 0
    old = contact['points']
    for begin, finish, first, last in strokes:
        while cursor < len(old) and old[cursor][0] < begin / 1000:
            points.append(list(old[cursor]))
            cursor += 1
        points.extend((first, last))
        while cursor < len(old) and old[cursor][0] <= finish / 1000:
            cursor += 1
    points.extend(list(p) for p in old[cursor:])
    return dict(start=min(contact['start'], strokes[0][0] / 1000),
                end=max(contact['end'], (strokes[-1][1] + release) / 1000), points=points)
