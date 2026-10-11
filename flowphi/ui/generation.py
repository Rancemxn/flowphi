from __future__ import annotations

from collections import Counter
import cmath
import math
import time

from flowphi.planning.planner import Planner, PlanningError, compact_stationary
from flowphi.charts.model import NoteType
from flowphi.ui.model import TimeIndex
from flowphi.motion import FINGER_ORDER, VERSION


def pending_contact(planner, task, reasons):
    end = task.end if task.note.type in (NoteType.HOLD, NoteType.FLICK) else task.start + (
        12 if task.note.type == NoteType.DRAG else 24)
    times = sorted({task.start, task.beat, end, *range(task.start, end, planner.settings.sample_ms)})
    if planner.blocks:
        times = sorted(set(times) | {ms for t in planner.blocks.key_times
                       for ms in (math.floor(t * 1000) - 1, math.floor(t * 1000), math.ceil(t * 1000))
                       if task.start <= ms <= end})
    points, empty_at = [], None
    for ms in times:
        when = planner.target_time(task, ms)
        target = task.line.pos(when, task.note.offset)
        if task.note.type == NoteType.FLICK:
            normal = cmath.exp(1j * (task.line.angle @ task.note.seconds)) * 1j
            target += normal * planner.screen.width * planner.settings.flick_distance * (ms - task.beat) / (end - task.start)
        if not math.isfinite(target.real) or not math.isfinite(target.imag):
            raise PlanningError(f'Non-finite note position for note {task.id} at {ms / 1000:.3f}s')
        xy = planner.project(planner.zone(task, ms), (target.real, target.imag))
        if xy is None:
            if empty_at is None:
                empty_at = ms / 1000
            xy = target.real, target.imag
        x = min(.99, max(.01, xy[0] / planner.screen.width))
        y = min(.99, max(.01, xy[1] / planner.screen.height))
        points.append([ms / 1000, x, y])
    side, finger = planner.keys[0]
    record = dict(pointer=(0 if side == -1 else 5) + FINGER_ORDER.index(finger),
                  hand='left' if side == -1 else 'right', finger=finger,
                  start=task.start / 1000, end=(end + 1) / 1000, beat=task.beat / 1000,
                  kind=task.note.type.name.lower(), note_ids=[task.id], points=points,
                  effort=.018 if task.note.type == NoteType.DRAG else .055, burst_cost=0.,
                  planned=False, draft_fallback=dict(rejections=reasons, empty_zone_time=empty_at),
                  decision=dict(degraded=['unplanned_draft']))
    record.update(planner.timing_metadata(task))
    return record


def assign_pending(contacts, pending, keys):
    if not pending:
        return
    all_contacts = contacts + pending
    assigned = set(range(len(contacts)))
    index = TimeIndex((c['start'], c['end'], i) for i, c in enumerate(all_contacts))
    for i in range(len(contacts), len(all_contacts)):
        c = all_contacts[i]
        overlap = Counter()
        for j in index.query(c['start'], c['end'] - 1e-9):
            if j not in assigned:
                continue
            other = all_contacts[j]
            overlap[other['hand'], other['finger']] += min(c['end'], other['end']) - max(c['start'], other['start'])
        x = c['points'][0][1]
        side, finger = min(keys, key=lambda key: (
            overlap['left' if key[0] == -1 else 'right', key[1]],
            max(0., -key[0] * (x - .5)), FINGER_ORDER.index(key[1])))
        c.update(hand='left' if side == -1 else 'right', finger=finger,
                 pointer=(0 if side == -1 else 5) + FINGER_ORDER.index(finger))
        assigned.add(i)


def generate_draft(chart, settings, profile, console=None, *, progress=None):
    started = time.monotonic()
    planner = Planner(chart, settings, profile)
    pending = []

    def retain(task, reasons):
        pending.append(pending_contact(planner, task, reasons))

    contacts = planner.run(console, on_unplanned=retain, progress=progress)
    # A later shared sweep can recover a previously unresolved note.
    covered = {nid for c in contacts for nid in c['note_ids']}
    pending = [c for c in pending if c['note_ids'][0] not in covered]
    if not contacts and not pending:
        raise PlanningError('No notes to plan')
    contacts = planner.join_drags(contacts)
    assign_pending(contacts, pending, planner.keys)
    contacts.extend(pending)
    if {nid for c in contacts for nid in c['note_ids']} != set(planner.original_tasks):
        raise PlanningError('Draft does not cover all chart notes')
    contacts.sort(key=lambda c: (c['start'], c['pointer']))
    for c in contacts:
        if c['kind'] != 'flick':
            c['points'] = compact_stationary(c['points'])
    # Derived motion and PSAP binding are produced only by the export compiler.
    return dict(version=VERSION, algorithm=5, timebase='chart_seconds', coordinates='normalized_screen',
                contacts=contacts, hand_rest=[], profile=profile, physical_screen=list(planner.physical),
                settings=vars(settings), palm_motion='v4',
                diagnostics=dict(notes=len(planner.tasks), contacts=len(contacts),
                    joined_drags=sum(len(c['note_ids']) - 1 for c in contacts),
                    seconds=time.monotonic() - started, rejections=dict(planner.rejected),
                    draft_fallback_notes=[nid for c in pending for nid in c['note_ids']],
                    degraded=[dict(note_ids=c['note_ids'], time=c['start'], reasons=c['decision']['degraded'])
                              for c in contacts if c['decision']['degraded']],
                    finger_usage=dict(Counter(f'{c["hand"]}:{c["finger"]}' for c in contacts)),
                    native_ap_validated=False))
