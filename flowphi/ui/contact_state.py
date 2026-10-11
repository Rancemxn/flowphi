from __future__ import annotations

import copy

from flowphi.charts.model import NoteType
from flowphi.motion import point_at


def chart_notes(chart):
    return [n for line in chart.lines for n in line.notes] if chart else []


def heads(contact, notes):
    """Hold body segments do not introduce a second head."""
    return [nid for nid in contact['note_ids'] if 0 <= nid < len(notes)
            and notes[nid].type in (NoteType.TAP, NoteType.HOLD)
            and contact.get('judgement_times', {}).get(str(nid), notes[nid].seconds)
            >= contact['start'] - .0600001]


def splice_contacts(previous, following):
    """A later segment owns its interval; keep the earlier path on either side."""
    result = copy.deepcopy(previous)
    start, end = following['start'], following['end']
    points = [list(p) for p in previous['points'] if p[0] < start]
    anchor = round(min(previous['end'] - .001, start - .001), 3)
    if anchor >= previous['start'] and (not points or points[-1][0] < anchor):
        points.append([anchor, *point_at(previous['points'], anchor)])
    points.extend(list(p) for p in following['points'])
    if end < previous['end'] - 1e-9:
        last = round(end - .001, 3)
        if points[-1][0] < last:
            points.append([last, *point_at(following['points'], last)])
        points.append([end, *point_at(previous['points'], end)])
        points.extend(list(p) for p in previous['points'] if p[0] > end)
    kinds = (previous['kind'], following['kind'])
    result.update(end=max(previous['end'], end), points=points,
                  note_ids=sorted(set(previous['note_ids']) | set(following['note_ids'])),
                  judgement_times={**previous.get('judgement_times', {}), **following.get('judgement_times', {})},
                  kind=next(k for k in ('flick', 'hold', 'drag', 'tap') if k in kinds),
                  joint_note_coverage=True, touch_mode='press',
                  editor_contacts=[*previous['editor_contacts'], *following['editor_contacts']],
                  shared_contact='authored_continuation')
    if result['end'] != previous['end']:
        result.pop('terminal_hold_release_ms', None)
        result.pop('terminal_hold_release_reason', None)
    return result


def resolve_contacts(document, chart=None):
    notes = chart_notes(chart)
    result, last, groups, problems = [], {}, {}, []
    for i in document.sorted_ids(range(len(document))):
        source = document.contact(i)
        c = dict(source, editor_contacts=[i], down_time=source.get('down_time', source['start']), touch_mode='press')
        key = c['hand'], c['finger']
        previous_index = last.get(key)
        previous = result[previous_index] if previous_index is not None else None
        mode = source.get('touch_mode', 'auto')
        passive = not heads(c, notes) if notes else c['kind'] in ('drag', 'flick')
        adjacent = previous is not None and abs(c['start'] - previous['end']) <= .0010001
        joining = previous is not None and c['start'] > previous['start'] + 1e-9 and (
            mode == 'continue' or (mode == 'auto' and passive and adjacent))
        if joining:
            result[previous_index] = splice_contacts(previous, c)
            groups[i] = previous_index
            continue
        if mode == 'continue':
            problems.append(dict(code='continuation_start', severity='error', contacts=[i], time=c['start'],
                message='找不到同一手指上更早开始的动作，无法继续按住请选择“重新按下”，或调整手指分配'))
        groups[i] = len(result)
        last[key] = len(result)
        result.append(c)
    return result, groups, problems


def clip_contact(contact, time_range):
    """Keep a carry-in touch down at frame zero without inventing a note hit."""
    if time_range is None:
        return copy.deepcopy(contact)
    low, high = time_range
    start, end = max(contact['start'], low), min(contact['end'], high)
    if end <= start:
        return None
    c = copy.deepcopy(contact)
    c['down_time'] = c.get('down_time', c['start'])
    c.setdefault('judgement_context', dict(start=contact['start'], end=contact['end'], points=copy.deepcopy(contact['points'])))
    c.update(start=start, end=end, points=[[start, *point_at(c['points'], start)],
             *[list(p) for p in c['points'] if start < p[0] < end]])
    last = round(end - .001, 3)
    if last > c['points'][-1][0]:
        c['points'].append([last, *point_at(contact['points'], last)])
    return c
