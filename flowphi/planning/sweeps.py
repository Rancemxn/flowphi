import cmath
import copy
import math

from flowphi.charts.model import NoteType
from flowphi.planning.contacts import ContactGeometry, validate_contact
from flowphi.motion import FINGER_ORDER, point_at
from flowphi.planning.timing import check_offset, audit_sweeps


def outward_group(planner, task):
    group = [t for t in planner.tasks if abs(t.note.seconds - task.note.seconds) < .001]
    flicks = [t for t in group if t.note.type == NoteType.FLICK]
    taps = [t for t in group if t.note.type == NoteType.TAP]
    drags = [t for t in group if t.note.type == NoteType.DRAG]
    if len(flicks) != 2 or not 1 <= len(taps) <= 2 or not drags or len(group) != len(flicks) + len(taps) + len(drags):
        return None
    origin = sum(t.note.seconds for t in group) / len(group)
    axis = cmath.exp(1j * (task.line.angle @ origin))
    def along(t):
        return (t.line.pos(origin, t.note.offset) * axis.conjugate()).real
    group.sort(key=along)
    if any(abs(math.sin((t.line.angle @ origin) - (task.line.angle @ origin))) > .02 for t in group):
        return None
    if len(taps) == 2:
        for split in range(1, len(group)):
            branches = []
            for part in (group[:split], group[split:]):
                if sum(t.note.type == NoteType.TAP for t in part) != 1 or sum(t.note.type == NoteType.FLICK for t in part) != 1:
                    break
                if part[-1].note.type == NoteType.TAP: part = list(reversed(part))
                if part[0].note.type != NoteType.TAP or part[-1].note.type != NoteType.FLICK:
                    break
                branches.append(part)
            if len(branches) == 2:
                return origin, axis, branches, group
        return None
    if group[0].note.type != NoteType.FLICK or group[-1].note.type != NoteType.FLICK:
        return None
    center = sum(along(t) for t in taps) / len(taps)
    if not along(group[0]) < center < along(group[-1]):
        return None
    left = [t for t in group if along(t) < center - 1e-7]
    right = [t for t in group if along(t) > center + 1e-7]
    middle = [t for t in group if t not in left and t not in right]
    # One central press starts one sweep; the other starts at its inner Drag.
    left = sorted([*left, *middle], key=along, reverse=True)
    right.sort(key=along)
    if not left or not right or any(sum(t.note.type == NoteType.TAP for t in branch) > 1 for branch in (left, right)):
        return None
    return origin, axis, (left, right), group


def sweep_states(planner, group, state):
    from flowphi.planning.planner import State
    from flowphi.planning.load import score_labels
    origin, axis, branches, tasks = group
    geometry = ContactGeometry(planner.chart, physical_screen=planner.physical, extra_clearance_m=.004)
    result = []
    for hands in (('left', 'right'), ('right', 'left')):
        history = copy.deepcopy(list(state.contacts))
        built = []
        valid = True
        for branch, hand in zip(branches, hands):
            a = branch[0].line.pos(origin, branch[0].note.offset)
            b = branch[-1].line.pos(origin, branch[-1].note.offset)
            direction = 1 if ((b - a) * axis.conjugate()).real > 0 else -1
            previous = next((c for c in reversed(history) if (c['hand'], c['finger']) == (hand, 'index')), None)
            available = previous['end'] + .002 if previous else origin - .018
            # A Hold's extra UP millisecond is not part of its authored tail.
            if previous and previous['kind'] == 'hold':
                notes = [planner.original_tasks[n].note for n in previous['note_ids']]
                tail = max(n.seconds + n.hold for n in notes)
                if all(n.type == NoteType.HOLD for n in notes) and tail <= origin + .0005 and previous['end'] <= tail + .0011:
                    previous['end'] = math.ceil(tail * 1000 - 1e-7) / 1000
                    last = round(previous['end'] - .001, 3)
                    previous['points'] = [p for p in previous['points'] if p[0] < last] + [[last, *point_at(previous['points'], last)]]
                    available = previous['end'] + .002
            start = round(max(origin - .018, available) * 1000) / 1000
            points = []; judgements = {}; schedule = []
            for i, item in enumerate(branch):
                when = round((start + i * .026) * 1000) / 1000
                try: check_offset(item.note, when)
                except ValueError:
                    valid = False; break
                pos = item.line.pos(when, item.note.offset)
                xy = (pos.real / planner.chart.width, pos.imag / planner.chart.height)
                judgements[str(item.id)] = when
                dwell = .036 if item.note.type == NoteType.FLICK else .012
                points.extend([[when, *xy], [round(when + dwell, 3), *xy]])
                schedule.append(dict(note=item.id, kind=item.note.type.name.lower(), seconds=when))
            if not valid: break
            # Pause at the Flick band then accelerate outward, so the terminal
            # Flick has a new gesture instead of relying on an earlier Drag.
            last = branch[-1]
            flick_time = round(points[-1][0] + .006, 3)
            try: check_offset(last.note, flick_time)
            except ValueError:
                valid = False; break
            judgements[str(last.id)] = flick_time
            schedule[-1]['seconds'] = flick_time
            delta = axis * direction * .055 * planner.chart.width
            target = [points[-1][1] + delta.real / planner.chart.width,
                    points[-1][2] + delta.imag / planner.chart.height]
            points.append([round(points[-1][0] + .020, 3), *target])
            end = round(points[-1][0] + .002, 3)
            c = dict(pointer=(0 if hand == 'left' else 5) + FINGER_ORDER.index('index'),
                hand=hand, finger='index', kind='flick', start=start, end=end, beat=flick_time,
                note_ids=[t.id for t in branch], points=points, judgement_times=judgements,
                judgement_time_reason='outward_sweep', judgement_window_schedule=schedule,
                shared_contact='outward_tap_drag_flick', joint_note_coverage=True,
                load_kind='tap' if any(t.note.type == NoteType.TAP for t in branch) else 'flick',
                effort=.055, burst_cost=0., planned=True, continuous_sweep=True)
            if validate_contact(geometry, c, points) or planner.blocks.path_violations(points, start, end):
                valid = False; break
            built.append(c)
        if not valid: continue
        try: audit_sweeps(planner.chart, built)
        except ValueError: continue
        contacts = sorted([*history, *built], key=lambda c: (c['start'], c['pointer']))
        keys = [(h, 'index') for h in ('left', 'right')]
        labels = [keys.index((c['hand'], c['finger'])) for c in contacts]
        cost, decisions = score_labels(contacts, keys, vars(planner.settings), labels)
        for c, decision in zip(contacts, decisions):
            c.update(effort=decision.pop('effort'), burst_cost=decision.pop('burst_cost'), decision=decision)
        result.append(State(tuple(contacts), cost))
    return result
