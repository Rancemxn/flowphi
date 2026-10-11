from flowphi.charts.model import NoteType
import math
from flowphi.motion import point_at

WINDOW_MS = {NoteType.TAP: 80, NoteType.HOLD: 80, NoteType.DRAG: 100, NoteType.FLICK: 140}
# Reserve at least 20 ms for input sampling and comparison strictness.
PLANNING_WINDOW_MS = {NoteType.TAP: 60, NoteType.HOLD: 60, NoteType.DRAG: 80, NoteType.FLICK: 110}


def check_offset(note, judgement):
    offset = (judgement - note.seconds) * 1000
    if abs(offset) > PLANNING_WINDOW_MS[note.type] + 1e-6:
        raise ValueError(f'Judgement offset outside planning window: {offset:.6f} ms')
    return dict(offset_ms=offset, window_ms=WINDOW_MS[note.type],
                planning_window_ms=PLANNING_WINDOW_MS[note.type],
                native_ap_validated=False)


def audit_sweeps(chart, contacts):
    notes = dict(enumerate((line, n) for line in chart.lines for n in line.notes))
    results = []
    for c in contacts:
        if not c.get('continuous_sweep'): continue
        matched = {n: [] for n in c['note_ids']}
        for phase_ms in range(16):
            phase = phase_ms / 1000
            first = math.ceil((c['start'] - phase) * 60 - 1e-8)
            last = math.ceil((c['end'] - phase) * 60 - 1e-8)
            previous = point_at(c['points'], c['start'])
            d0 = (0., 0.); stopped = True; seen = {}
            for frame in range(first, last + 1):
                when = phase + frame / 60
                xy = point_at(c['points'], min(when, c['end']))
                d1 = ((xy[0] - previous[0]) * chart.width / chart.height * 10, (xy[1] - previous[1]) * 10)
                length = math.hypot(*d0)
                relative = sum(a * b for a, b in zip(d0, d1)) / length if length > .1 else 0.
                new_flick = False
                if relative < .06 or stopped:
                    new_flick = math.hypot(*d1) >= .3
                    stopped = not new_flick
                for nid in c['note_ids']:
                    if nid in seen: continue
                    line, note = notes[nid]
                    delta = when - note.seconds
                    if abs(delta) >= WINDOW_MS[note.type] / 1000: continue
                    center = line.pos(when, note.offset); angle = line.angle @ when
                    dx = xy[0] * chart.width - center.real; dy = xy[1] * chart.height - center.imag
                    along = abs(dx * math.cos(angle) + dy * math.sin(angle)) / chart.width
                    half = .106875 if note.type == NoteType.TAP else .118125
                    if along >= half: continue
                    if note.type == NoteType.TAP and frame != first: continue
                    if note.type == NoteType.FLICK and not new_flick: continue
                    if note.type == NoteType.HOLD: raise ValueError('A sweep must not replace a Hold')
                    seen[nid] = delta * 1000
                d0 = d1; previous = xy
            missing = set(c['note_ids']) - set(seen)
            if missing: raise ValueError(f'Sweep misses notes {sorted(missing)} at 60 Hz phase {phase_ms} ms')
            for nid, delta in seen.items(): matched[nid].append(delta)
        results.append(dict(notes=c['note_ids'], hand=c['hand'], finger=c['finger'], fps=60, phases=16,
            offset_ranges_ms={str(n): [min(ds), max(ds)] for n, ds in matched.items()},
            new_flick_gesture='passed', native_ap_validated=False))
    return results
