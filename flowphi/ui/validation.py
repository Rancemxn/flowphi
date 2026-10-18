from __future__ import annotations

from collections import defaultdict
import math

from flowphi.motion import FINGER_ORDER, point_at
from flowphi.ui.contact_state import clip_contact, resolve_contacts


def issue(code, message, ids, time, severity='error'):
    return dict(code=code, message=message, contacts=list(ids), time=float(time), severity=severity)


def quick_check(document, indices=None, *, chart=None, time_range=None):
    """Check physical finger occupancy after resolving authored continuations."""
    contacts, groups, problems = resolve_contacts(document, chart)
    targets = set(range(len(contacts))) if indices is None else {groups[i] for i in indices}
    problems = [p for p in problems if (indices is None or set(p['contacts']).intersection(indices))
                and (time_range is None or time_range[0] <= p['time'] < time_range[1])]
    active = defaultdict(list)
    for i, original in enumerate(contacts):
        a = original
        if time_range is not None:
            low, high = max(a['start'], time_range[0]), min(a['end'], time_range[1])
            if high <= low:
                continue
            a = dict(a, start=low, end=high)
        if a['finger'] not in document.allowed_fingers:
            if i in targets:
                problems.append(issue('disabled_finger', '使用了方案中未启用的手指，请改用已启用的手指；如需增加手指，请重新生成指法', a['editor_contacts'], a['start']))
        key = a['hand'], a['finger']
        active[key] = [(j, b) for j, b in active[key] if b['end'] > a['start'] + 1e-8]
        for j, b in active[key]:
            if i not in targets and j not in targets:
                continue
            low, high = max(a['start'], b['start']), min(a['end'], b['end'])
            if high - low > 1e-8:
                problem = issue('finger_overlap',
                    f'同一根手指上的两段动作重叠 {(high - low) * 1000:.3f} ms若要贴屏接续，请将后段设为“继续按住”；需要新点击时请分配其他手指或先抬起',
                    sorted(set(a['editor_contacts'] + b['editor_contacts'])), low)
                problem['end_time'] = high
                problems.append(problem)
        active[key].append((i, a))
    return problems


def head_check(document, notes, time_range=None):
    """Match each Tap/Hold head to a distinct physical DOWN."""
    from flowphi.charts.model import NoteType
    from flowphi.planning.timing import PLANNING_WINDOW_MS
    candidates = {}
    for nid, (_, note) in enumerate(notes):
        if note.type not in (NoteType.TAP, NoteType.HOLD):
            continue
        candidates[nid] = [i for i in document.note_contacts.get(nid, ())
            if abs(document.contact(i).get('down_time', document.contact(i)['start']) -
                   document.contact(i).get('judgement_times', {}).get(str(nid), note.seconds))
            <= PLANNING_WINDOW_MS[note.type] / 1000 + .0011]
    owners = {}
    def match(nid, seen):
        for i in candidates[nid]:
            if i in seen:
                continue
            seen.add(i)
            if i not in owners or match(owners[i], seen):
                owners[i] = nid
                return True
        return False
    for nid in sorted(candidates, key=lambda n: (notes[n][1].seconds, n)):
        match(nid, set())
    matched = set(owners.values())
    return [issue('missing_down',
        f'音符 {nid}（{notes[nid][1].type.name.title()}）需要单独按下，持续按住不能代替新点击请拆开合并动作并分配其他手指，或在音符起点重新按下',
        document.note_contacts.get(nid, []), notes[nid][1].seconds)
        for nid in candidates if nid not in matched and
        (time_range is None or time_range[0] <= notes[nid][1].seconds < time_range[1])]


def edit_check(document, chart=None):
    problems = quick_check(document, chart=chart)
    if chart:
        from flowphi.ui.model import PlanDocument
        contacts, _, _ = resolve_contacts(document, chart)
        physical = PlanDocument(dict(document.baseline, contacts=contacts), copy_baseline=False, allow_empty=True)
        notes = [(line, note) for line in chart.lines for note in line.notes]
        for problem in head_check(physical, notes):
            problem['contacts'] = sorted({j for i in problem['contacts'] for j in contacts[i]['editor_contacts']})
            problems.append(problem)
    return problems


def draft_warnings(document, indices=None):
    """Persist the planner's unresolved-note locations through saving and edits."""
    targets = range(len(document)) if indices is None else sorted(set(indices))
    problems = []
    for i in targets:
        c = document.contact(i)
        pending = c.get('draft_fallback')
        if pending is None:
            continue
        note_ids = ', '.join(map(str, c['note_ids']))
        message = f'音符 {note_ids} 生成初稿时没有找到可行分配，暂用了临时指法和轨迹请复查后点击“检查方案”'
        if pending.get('empty_zone_time') is not None:
            message += ' 这段动作中有时刻没有可用判定区域'
        if i in document.edits:
            message += ' 已有人工修改，可手动检查当前效果'
        problems.append(issue('draft_fallback', message, [i], c['beat'], 'warning'))
    return problems


def movement_warnings(document):
    if not document.baseline['settings'].get('speed_limits', True):
        return []
    previous, result = {}, []
    screen = document.baseline['physical_screen']
    speed = document.baseline['settings'].get('finger_speed', 2.2)
    for i in document.sorted_ids(range(len(document))):
        c = document.contact(i)
        key = c['hand'], c['finger']
        if key in previous:
            j, old = previous[key]
            gap = c['start'] - old['end']
            distance = math.hypot((c['points'][0][1] - old['points'][-1][1]) * screen[0],
                                  (c['points'][0][2] - old['points'][-1][2]) * screen[1])
            if gap >= 0 and distance > speed * max(gap, .001) and distance > .005:
                result.append(issue('fast_transfer', f'手指需在约 {gap * 1000:.0f} ms 内移动 {distance * 1000:.0f} mm，建议生成片段视频查看动作',
                                    [j, i], c['start'], 'warning'))
        previous[key] = i, c
    return result


def validate(document, chart, *, geometry=True, progress=None, time_range=None, resolved=False):
    if not resolved:
        from flowphi.ui.model import PlanDocument
        contacts, _, connection_problems = resolve_contacts(document, chart)
        canonical = PlanDocument(dict(document.baseline, contacts=contacts), chart=document.chart, copy_baseline=False, allow_empty=True)
        report = validate(canonical, chart, geometry=geometry, progress=progress, time_range=time_range, resolved=True)
        for problem in report['issues']:
            problem['contacts'] = sorted({j for i in problem['contacts'] for j in contacts[i]['editor_contacts']})
        report['issues'].extend(p for p in connection_problems if time_range is None or time_range[0] <= p['time'] < time_range[1])
        report['valid'] = not any(p['severity'] == 'error' for p in report['issues'])
        return report
    problems = quick_check(document, chart=chart, time_range=time_range)
    notes = [(line, n) for line in chart.lines for n in line.notes]
    expected = {i for i, (_, n) in enumerate(notes) if time_range is None or
                (n.seconds < time_range[1] and n.seconds + max(n.hold, .001) > time_range[0])}
    actual = set(document.note_contacts)
    if not expected <= actual or actual - set(range(len(notes))):
        problems.append(issue('chart_coverage', '方案与谱面的音符编号不匹配，或有音符未安排请确认关联的原谱面，必要时重新生成指法', [], 0.))
        return dict(valid=False, issues=problems, geometry_checked=False)
    problems.extend(head_check(document, notes, time_range))
    selected = [i for i in range(len(document)) if time_range is None or
                document.contact(i)['start'] < time_range[1] and document.contact(i)['end'] > time_range[0]]
    problems.extend(draft_warnings(document, selected))
    problems.extend(p for p in movement_warnings(document) if time_range is None or time_range[0] <= p['time'] < time_range[1])
    if geometry:
        from flowphi.charts.model import NoteType
        from flowphi.planning.contacts import ContactGeometry, validate_contact
        from flowphi.planning.timing import PLANNING_WINDOW_MS, check_offset
        g = ContactGeometry(chart, physical_screen=document.baseline['physical_screen'],
                            extra_clearance_m=document.baseline.get('comfort', {}).get('extra_block_clearance_m', 0.))
        for done, i in enumerate(selected, 1):
            c = clip_contact(document.contact(i), time_range)
            for nid in c['note_ids']:
                note = notes[nid][1]
                judgement = g.judgement_time(c, nid)
                if (time_range is None or time_range[0] <= judgement < time_range[1]) and abs(judgement - note.seconds) > 1e-9:
                    try:
                        if document.baseline['settings'].get('judgement_windows', False):
                            check_offset(note, judgement)
                        elif note.type != NoteType.DRAG or abs(judgement - note.seconds) > .0155001:
                            raise ValueError('判定时刻超出方案允许的范围')
                    except ValueError as exc:
                        problems.append(issue('judgement_time', str(exc), [i], judgement))
            bad = validate_contact(g, c, c['points'], limit=1)
            if bad:
                problems.append(issue('geometry', '轨迹离开了有效判定区域，或进入遮挡区域请在“轨迹”页调整相关控制点，再检查方案', [i], bad[0]))
            native = chart.block_areas.path_violations(c['points'], c['start'], c['end'])
            if native and not bad:
                problems.append(issue('blocked', '轨迹穿过了遮挡区域请在“轨迹”页调整相关控制点，再检查方案', [i], c['start']))
            if progress and (done % 64 == 0 or done == len(selected)):
                progress(done, len(selected))
        # A long Hold can be split across contacts. Validate the union of their
        # lifetimes rather than requiring every segment to cover the entire Hold.
        for nid, (_, note) in enumerate(notes):
            if nid not in expected:
                continue
            ids = document.note_contacts[nid]
            segments = sorted((document.contact(i)['start'], document.contact(i)['end'], i) for i in ids)
            judgement = g.judgement_time(document.contact(ids[0]), nid)
            check_hit = time_range is None or time_range[0] <= judgement < time_range[1]
            if check_hit and not any(a - .0011 <= judgement < b + .0011 for a, b, _ in segments):
                problems.append(issue('missing_hit', '音符判定时没有手指按住，请检查动作的按下和抬起时间', ids, judgement))
            if note.type == NoteType.FLICK and check_hit:
                width, height = document.baseline['physical_screen']
                window = PLANNING_WINDOW_MS[NoteType.FLICK] / 1000
                distance = 0.
                for i in ids:
                    c = document.contact(i)
                    beat = g.judgement_time(c, nid)
                    motion = c.get('judgement_context', c)
                    low, high = max(motion['start'], beat - window), min(motion['end'] - 1e-7, beat + window)
                    if high <= low:
                        continue
                    times = [low, *(p[0] for p in motion['points'] if low < p[0] < high), high]
                    xy = [point_at(motion['points'], t) for t in times]
                    travelled = sum(math.hypot((b[0] - a[0]) * width, (b[1] - a[1]) * height)
                                    for a, b in zip(xy, xy[1:]))
                    distance = max(distance, travelled)
                if distance < 1e-8:
                    problems.append(issue('flick_motion', 'Flick 判定附近没有滑动，请检查轨迹和抬起时间', ids, judgement))
                elif distance < width * .10:
                    # The planner's conservative 10% target is not a native
                    # game threshold. Authored short swipes remain exportable.
                    problems.append(issue('flick_distance', 'Flick 判定附近的滑动较短；已保留自定义长度，请确认实际效果',
                                          ids, judgement, 'warning'))
            if note.type == NoteType.HOLD:
                cursor = max(note.seconds, judgement, time_range[0] if time_range else -math.inf)
                terminal = min(note.seconds + note.hold, time_range[1] if time_range else math.inf)
                for a, b, i in segments:
                    if a > cursor + .0011:
                        break
                    cursor = max(cursor, b)
                    c = document.contact(i)
                    if c.get('terminal_hold_release_ms') is not None and 0 <= terminal - b <= .0200001:
                        cursor = terminal
                if cursor < terminal - .0011:
                    problems.append(issue('hold_gap', 'Hold 尚未结束，却有一段时间没有手指按住请检查抬起时间或接续动作', ids, cursor))
    return dict(valid=not any(p['severity'] == 'error' for p in problems), issues=problems,
                geometry_checked=geometry, contacts=len(selected), notes=len(expected), time_range=time_range,
                native_ap_validated=False)
