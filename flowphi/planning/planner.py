from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import copy
import cmath
import hashlib
import math
import time

from shapely.geometry import Point, box
from shapely.ops import nearest_points

from flowphi.charts.model import Note, NoteType
from flowphi.planning.geometry import JudgeArea, JUDGE_HALF_TAP, JUDGE_HALF_DRAG
from flowphi.planning.touch import ScreenUtil, TouchAction, VirtualTouchEvent, dump_data
from flowphi.motion import (FINGER_ORDER, VERSION, default_profile, finish_motion,
                            point_at, travel_time, world_xy)


@dataclass(frozen=True)
class Settings:
    beam_width: int = 8
    candidates_per_finger: int = 3
    sample_ms: int = 20
    screen_width_m: float = .28
    finger_speed: float = 2.2
    finger_acceleration: float = 28.
    wrist_speed: float = .9
    wrist_acceleration: float = 9.
    fatigue_seconds: float = 4.
    recovery_seconds: float = .65
    allow_degraded: bool = True
    fingers: tuple[str, ...] = ('index', 'middle', 'ring', 'thumb', 'little')
    visual_radius: float = .012
    hold_margin: float = .004
    flick_distance: float = .16
    fingering_objective: str = 'balanced'
    speed_limits: bool = True
    judgement_windows: bool = False

    def __post_init__(self):
        if self.fingering_objective not in ('balanced', 'load'):
            raise ValueError('Unknown fingering objective')
        if not (1 <= self.beam_width <= 32 and 1 <= self.candidates_per_finger <= 8 and 1 <= self.sample_ms <= 25):
            raise ValueError('Invalid draft search/sample budget')
        if any(not math.isfinite(x) or x <= 0 for x in (self.screen_width_m, self.finger_speed,
                self.finger_acceleration, self.wrist_speed, self.wrist_acceleration,
                self.fatigue_seconds, self.recovery_seconds)):
            raise ValueError('Invalid draft movement/load settings')
        if not self.fingers or any(f not in FINGER_ORDER for f in self.fingers):
            raise ValueError('Invalid enabled fingers')
        # Editor job requests round-trip through JSON, where tuples become lists.
        object.__setattr__(self, 'fingers', tuple(self.fingers))
        if not 0 < self.visual_radius <= .03:
            raise ValueError('Invalid visual contact radius')
        if not 0 <= self.hold_margin <= .02:
            raise ValueError('Invalid Hold judgement margin')
        if not .10 <= self.flick_distance <= .25:
            raise ValueError('Invalid Flick distance')


class PlanningError(ValueError):
    pass


@dataclass(frozen=True)
class Task:
    id: int
    note: object
    line: object
    start: int
    end: int
    beat: int


@dataclass(frozen=True)
class State:
    contacts: tuple = ()
    score: float = 0.


COMFORT = dict(index=5.5, middle=5., ring=3.6, thumb=3.2, little=2.8)
PEAK = dict(index=13., middle=11., ring=8., thumb=7., little=6.)
PREFERENCE = dict(index=0., middle=.18, ring=.5, thumb=.35, little=.65)


def load_at(contacts, t, hand, finger=None, settings=None):
    """Analytic decay/integration: MOVE sampling and video fps never count as clicks."""
    settings = settings or Settings()
    fatigue, debt = 0., 0.
    tau, recovery = settings.fatigue_seconds, settings.recovery_seconds
    for c in contacts:
        if c['hand'] != hand or (finger is not None and c['finger'] != finger) or c['start'] > t:
            continue
        age = t - c['start']
        pulse = c.get('effort', .055)
        fatigue += pulse * math.exp(-age / tau)
        debt += c.get('burst_cost', 0.) * math.exp(-age / recovery)
        if c['kind'] == 'hold':
            duration = min(t, c['end']) - c['start']
            # Integral of sustained work with exponential recovery after release.
            fatigue += .11 * tau * (1 - math.exp(-duration / tau)) * math.exp(-max(0., t - c['end']) / tau)
    if finger is None:
        fatigue *= .5
        debt *= .55
    return min(1., fatigue), max(0., 1 - debt)


class Planner:
    def __init__(self, chart, settings, profile):
        self.chart, self.settings, self.profile = chart, settings, profile
        from flowphi.charts.blocks import BlockAreas
        self.blocks = getattr(chart, 'block_areas', None) or BlockAreas()
        self.screen = ScreenUtil(chart.width, chart.height)
        self.physical = (settings.screen_width_m, settings.screen_width_m * chart.height / chart.width)
        self.keys = [(side, finger) for side in (-1, 1) for finger in settings.fingers
                     if f'{side}:{finger}' in profile['fingers']]
        if not self.keys:
            raise ValueError('Hand profile has no enabled fingers')
        w, h = chart.width, chart.height
        self.screen_poly = box(w * .01, h * .01, w * .99, h * .99)
        self.pause_poly = box(0, 0, w * .1, h * .1).union(box(w * .9, 0, w, h * .1))
        self.zones, self.paths = {}, {}
        self.corridor_paths = {}
        self.terminal_paths = {}
        self.judgement_times = {}
        self.rejected = Counter()
        self.tasks = []
        for line in chart.lines:
            for note in line.notes:
                if not all(math.isfinite(v) for v in (note.seconds, note.hold, note.offset.real, note.offset.imag)) or note.hold < 0:
                    raise PlanningError('Non-finite note or negative Hold duration')
                if note.type not in (NoteType.TAP, NoteType.DRAG, NoteType.HOLD, NoteType.FLICK):
                    raise PlanningError(f'Unsupported note type: {note.type}')
                beat = round(note.seconds * 1000)
                start = beat - 25 if note.type == NoteType.FLICK else beat
                end = round((note.seconds + note.hold) * 1000) if note.type == NoteType.HOLD else (
                    beat + 45 if note.type == NoteType.FLICK else beat + 70)
                self.tasks.append(Task(len(self.tasks), note, line, start, max(start + 1, end), beat))
        self.original_tasks = {task.id: task for task in self.tasks}
        for i, task in enumerate(self.tasks):
            if task.note.type != NoteType.DRAG or not self.zone(task, task.start).is_empty:
                continue
            # The compatible timing search permits 15 ms around an
            # invisible judgement instant. Keep the rendered chart untouched.
            for delta in (d * s for d in range(1, 16) for s in (-1, 1)):
                beat = task.beat + delta
                shifted = Note(task.note.type, beat / 1000, task.note.hold, task.note.offset)
                candidate = Task(task.id, shifted, task.line, beat, beat + 70, beat)
                if not self.zone(candidate, beat).is_empty:
                    self.tasks[i] = candidate
                    self.judgement_times[task.id] = shifted.seconds
                    self.zones = {key: value for key, value in self.zones.items() if key[0] != task.id}
                    break
        self.tasks.sort(key=lambda n: (n.start, n.id))

    def target_time(self, task, ms):
        # Taps stay at their judgement position; sustained contacts follow the note.
        # The Hold tail is sampled just before release, not after a line's cut.
        if task.note.type == NoteType.HOLD:
            return max(task.note.seconds, min(ms / 1000, task.note.seconds + task.note.hold - .001))
        if task.note.type == NoteType.DRAG:
            return task.note.seconds if task.id in self.judgement_times else max(task.note.seconds, ms / 1000)
        return task.note.seconds

    def zone(self, task, ms):
        key = (task.id, task.note.seconds, ms)
        if key not in self.zones:
            sec = self.target_time(task, ms)
            rot = cmath.exp(1j * (task.line.angle @ sec))
            center = task.line.pos(sec, task.note.offset)
            half = JUDGE_HALF_DRAG if task.note.type in (NoteType.DRAG, NoteType.FLICK) else JUDGE_HALF_TAP
            legal = JudgeArea(center, rot, self.screen.width, self.screen.height, half).get_valid_poly(
                self.screen_poly, self.pause_poly)
            if self.blocks:
                forbidden = self.blocks.forbidden(ms / 1000)
                if not forbidden.is_empty:
                    legal = legal.difference(forbidden.buffer(self.screen.width * .0002))
            if task.note.type == NoteType.HOLD and not legal.is_empty:
                inset = legal.buffer(-self.screen.width * self.settings.hold_margin)
                if not inset.is_empty:
                    legal = inset
            if legal.is_empty:
                self.zones[key] = legal
            else:
                # Game judgement strips can span the whole screen. Animation must
                # hit the visible note, with only a few millimetres of local freedom.
                visible = self.project(legal, (center.real, center.imag))
                radius = self.screen.width * self.settings.visual_radius
                if task.note.type == NoteType.FLICK:
                    radius += self.screen.width * self.settings.flick_distance * abs(ms - task.beat) / 70
                self.zones[key] = legal.intersection(Point(*visible).buffer(radius))
        return self.zones[key]

    def project(self, zone, seed):
        if zone.is_empty:
            return None
        p = Point(*seed)
        if zone.covers(p):
            return seed
        nearest = nearest_points(zone, p)[0]
        return nearest.x, nearest.y

    def offset(self, side, finger):
        return self.profile['fingers'][f'{side}:{finger}']['offset']

    def anchor(self, contact, t):
        xy = world_xy(point_at(contact['points'], t), self.physical)
        side = -1 if contact['hand'] == 'left' else 1
        offset = self.offset(side, contact['finger'])
        return xy[0] - offset[0], xy[1] - offset[1]

    def seeds(self, task, state, side, finger):
        w, h = self.screen.width, self.screen.height
        current = task.line.pos(task.note.seconds, task.note.offset)
        same = [c for c in state.contacts if c['hand'] == ('left' if side == -1 else 'right')]
        last = next((c for c in reversed(same) if c['finger'] == finger), None)
        offset = self.offset(side, finger)
        # Keep a hand's other fingertips near its existing wrist, with distinct pad positions.
        if same:
            a = self.anchor(same[-1], task.start / 1000)
            coordinated = ((a[0] + offset[0]) / self.physical[0] + .5,
                           .5 - (a[1] + offset[1]) / self.physical[1])
        else:
            coordinated = (.5 + side * .24 + offset[0] / self.physical[0], .60)
        targets = [(current.real, current.imag), (coordinated[0] * w, coordinated[1] * h)]
        if last:
            targets.append((last['points'][-1][1] * w, last['points'][-1][2] * h))
        zone = self.zone(task, task.beat if task.note.type == NoteType.FLICK else task.start)
        if task.note.type in (NoteType.HOLD, NoteType.DRAG):
            point = self.project(zone, (current.real, current.imag))
            return [point] if point is not None else []
        result = []
        for seed in targets:
            p = self.project(zone, seed)
            if p is not None and all(math.dist(p, q) > w * .005 for q in result):
                result.append(p)
        return result

    def path(self, task, seed, end, direction=1):
        key = (task.id, task.note.seconds, task.start, tuple(round(v, 5) for v in seed), end, direction)
        if key in self.paths:
            return self.paths[key]
        times = sorted({task.start, end, task.beat, *range(task.start, end, self.settings.sample_ms)})
        if self.blocks:
            times = sorted(set(times) | {ms for t in self.blocks.key_times
                           for ms in (math.floor(t * 1000) - 1, math.floor(t * 1000), math.ceil(t * 1000))
                           if task.start <= ms <= end})
        points, pos = [], seed
        normal = cmath.exp(1j * (task.line.angle @ task.note.seconds)) * 1j
        if abs(direction) == 2:
            normal *= -1j
        normal *= 1 if direction > 0 else -1
        flick_distance = self.screen.width * self.settings.flick_distance
        origin = task.line.pos(self.target_time(task, task.start), task.note.offset)
        for ms in times:
            if task.note.type == NoteType.FLICK:
                u = (ms - task.beat) / (end - task.start)
                target = (seed[0] + normal.real * flick_distance * u,
                          seed[1] + normal.imag * flick_distance * u)
            elif task.note.type in (NoteType.HOLD, NoteType.DRAG):
                center = task.line.pos(self.target_time(task, ms), task.note.offset)
                target = (center.real, center.imag)
            else:
                center = task.line.pos(self.target_time(task, ms), task.note.offset)
                target = (seed[0] + center.real - origin.real, seed[1] + center.imag - origin.imag)
            pos = self.project(self.zone(task, ms), target)
            if pos is None:
                self.paths[key] = None
                return None
            points.append([ms / 1000, pos[0] / self.screen.width, pos[1] / self.screen.height])
        # A segment between legal endpoints may still pass through a block. This
        # is a hard legality check, including Tap dwell and early Flick motion;
        # allow_degraded never relaxes it.
        if self.blocks.path_violations(points, task.start / 1000, end / 1000):
            self.paths[key] = None
            return None
        # Reject a clipped Flick that barely moves; native DPI-specific validation remains separate.
        if task.note.type == NoteType.FLICK and math.dist(world_xy(points[0][1:], self.physical),
                world_xy(points[-1][1:], self.physical)) < self.physical[0] * .10:
            points = None
        self.paths[key] = points
        return points

    def constraints(self, record, state, side, finger):
        """Return physical violations and normalized movement/pose costs."""
        t, end = record['start'], record['end']
        if any(c['hand'] == record['hand'] and c['finger'] == finger and
               max(t, c['start']) < min(end, c['end']) - 1e-9 for c in state.contacts):
            return ['occupied'], 0., 0.
        previous = next((c for c in reversed(state.contacts) if c['hand'] == record['hand'] and c['finger'] == finger), None)
        violations, movement, posture = [], 0., 0.
        xy = world_xy(record['points'][0][1:], self.physical)
        if previous:
            if previous['end'] > t + 1e-9:
                return ['occupied'], 0., 0.
            gap = t - previous['end']
            distance = math.dist(world_xy(previous['points'][-1][1:], self.physical), xy)
            required = max(.014, travel_time(distance, self.settings.finger_speed, self.settings.finger_acceleration))
            movement = (required / max(gap, .001)) ** 2 + distance / .3
            if gap < required:
                violations.append('finger_transfer')
        a = self.anchor(record, t)
        hand_previous = next((c for c in reversed(state.contacts) if c['hand'] == record['hand']), None)
        if hand_previous and hand_previous['end'] < t:
            distance = math.dist(self.anchor(hand_previous, hand_previous['end']), a)
            required = travel_time(max(0., distance - .025), self.settings.wrist_speed, self.settings.wrist_acceleration)
            available = t - hand_previous['end']
            movement += (required / max(.02, available)) ** 2 * .35
            if required > available:
                violations.append('wrist_transfer')
        for first, second in zip(record['points'], record['points'][1:]):
            speed = math.dist(world_xy(first[1:], self.physical), world_xy(second[1:], self.physical)) / (second[0] - first[0])
            if speed > self.settings.finger_speed:
                violations.append('contact_speed')
                break
        for other in reversed(state.contacts):
            if other['end'] < t - .4:
                # Long Holds can precede shorter contacts, so cannot break here.
                continue
            low, high = max(t, other['start']), min(end, other['end'])
            if high <= low:
                continue
            times = sorted({low, high - 1e-7, *(p[0] for p in record['points'] if low <= p[0] < high),
                            *(p[0] for p in other['points'] if low <= p[0] < high)})
            times += [(a + b) / 2 for a, b in zip(times, times[1:])]
            for when in times:
                p = world_xy(point_at(record['points'], when), self.physical)
                q = world_xy(point_at(other['points'], when), self.physical)
                distance = math.dist(p, q)
                same = other['hand'] == record['hand']
                if distance < (.013 if same else .018):
                    violations.append('pad_collision')
                    break
                anchors = math.dist(self.anchor(record, when), self.anchor(other, when))
                if same:
                    posture = max(posture, (anchors / .05) ** 2)
                    other_side = -1 if other['hand'] == 'left' else 1
                    expected = self.offset(side, finger)[0] - self.offset(other_side, other['finger'])[0]
                    if (p[0] - q[0]) * expected < -.00012 and abs(p[1] - q[1]) < .04:
                        # Crossing fingers can be layered in height by the pose solver.
                        posture = max(posture, .5)
                    if anchors > .065 * self.profile['scale'] / .27:
                        violations.append('hand_span')
                        break
                elif anchors < .055 * self.profile['scale'] / .27:
                    # Wrist proximity is a pose preference, not a ban on crossed hands.
                    posture = max(posture, .5)
        # Coarse swept fingertip check during transfer, not just at endpoints.
        if previous and 0 < t - previous['end'] < .65:
            start = previous['end']
            a = world_xy(previous['points'][-1][1:], self.physical)
            for other in state.contacts:
                low, high = max(start, other['start']), min(t, other['end'])
                if high <= low or other is previous:
                    continue
                for step in range(1, 6):
                    when = low + (high - low) * step / 6
                    u = (when - start) / (t - start)
                    p = (a[0] + (xy[0] - a[0]) * u, a[1] + (xy[1] - a[1]) * u, .018 * math.sin(math.pi * u))
                    q = (*world_xy(point_at(other['points'], when), self.physical), 0.)
                    if math.dist(p, q) < .012:
                        # The baked transfer can pass above an occupied contact.
                        posture = max(posture, .5)
        if not self.settings.speed_limits:
            violations = [v for v in violations if v not in
                          ('finger_transfer', 'wrist_transfer', 'contact_speed')]
        return sorted(set(violations)), movement, posture

    def corridor_path(self, task, end):
        """Route a sustained contact through its legal band if center tracking fails.

        A moving obstacle can cut the direct line between legal center-following
        samples. Search the actual judgement corridor rather than treating that
        one failed trajectory as an impossible note.
        """
        key = (task.id, task.note.seconds, task.start, end)
        if key in self.corridor_paths:
            return self.corridor_paths[key]
        from flowphi.planning.contacts import ContactGeometry, refine_contact, route_contact, validate_contact
        geometry = ContactGeometry(self.chart, physical_screen=self.physical, extra_clearance_m=.004)
        times = sorted({task.start, end, *range(task.start, end, self.settings.sample_ms)})
        points = []
        for ms in times:
            center = task.line.pos(self.target_time(task, ms), task.note.offset)
            points.append([ms / 1000, center.real / self.screen.width, center.imag / self.screen.height])
        contact = dict(kind=task.note.type.name.lower(), note_ids=[task.id], start=task.start / 1000,
                       end=(end + 1) / 1000, beat=task.beat / 1000, points=points)
        contact.update(self.timing_metadata(task))
        result = None
        extra = set()
        for solver in (refine_contact, route_contact):
            for attempt in range(3):
                candidate, _ = (solver(geometry, contact, extra_times=extra) if solver is refine_contact
                               else solver(geometry, contact, self.physical, extra_times=extra))
                if candidate is None:
                    break
                bad = validate_contact(geometry, contact, candidate, limit=100)
                if not bad and not self.blocks.path_violations(candidate, contact['start'], contact['end']):
                    result = candidate
                    break
                extra.update(bad)
            if result is not None:
                break
        self.corridor_paths[key] = result
        return result

    def choices(self, task, state, degraded=False):
        result = []
        t = task.start / 1000
        hands = {hand: load_at(state.contacts, t, hand, settings=self.settings) for hand in ('left', 'right')}
        for side, finger in self.keys:
            hand = 'left' if side == -1 else 'right'
            previous = next((c for c in reversed(state.contacts) if c['hand'] == hand and c['finger'] == finger), None)
            if previous and previous['end'] > t + 1e-9:
                self.rejected['occupied'] += 1
                continue
            fatigue, burst = load_at(state.contacts, t, hand, finger, self.settings)
            hand_fatigue, hand_burst = hands[hand]
            cps = 1 / max(.001, t - previous['start']) if previous else 0.
            comfort = COMFORT[finger] * (1 - .3 * fatigue - .15 * hand_fatigue)
            limit = PEAK[finger] * (.65 + .35 * burst) * (1 - .2 * hand_fatigue)
            over = cps > limit and task.note.type in (NoteType.TAP, NoteType.HOLD)
            if over and not degraded and self.settings.fingering_objective != 'load':
                self.rejected['burst_capacity'] += 1
                continue
            burst_cost = .10 * max(0., cps / comfort - 1) if task.note.type == NoteType.TAP else 0.
            # Keep most of a beat interval available for the next stroke. A 75 ms
            # dwell on ordinary fast taps artificially forced auxiliary fingers.
            dwell = round(max(.024, min(.060, .18 / max(1., cps))) * 1000)
            if task.note.type == NoteType.DRAG:
                # A Drag is a pass-through contact, not a Tap that must dwell.
                # Short successive Drags are joined into a continuous stroke below.
                dwell = 12
            end = task.end if task.note.type in (NoteType.HOLD, NoteType.FLICK) else task.start + dwell
            base_after_f, base_after_b = load_at(state.contacts, (end + 1) / 1000, hand, finger, self.settings)
            candidates = []
            seeds = self.seeds(task, state, side, finger)
            if not seeds:
                self.rejected['judge_geometry'] += 1
                continue
            for seed in seeds:
                directions = (1, -1, 2, -2) if task.note.type == NoteType.FLICK else (1,)
                for direction in directions:
                    effective_end = end
                    points = self.path(task, seed, end, direction)
                    if points is None and task.note.type in (NoteType.HOLD, NoteType.DRAG, NoteType.TAP):
                        points = self.corridor_path(task, end)
                    if points is None and task.note.type == NoteType.HOLD:
                        if task.id not in self.terminal_paths:
                            self.terminal_paths[task.id] = (None, end)
                            # Only an empty terminal zone permits this animation
                            # grace. Earlier geometry remains fully validated.
                            from flowphi.planning.contacts import ContactGeometry
                            g = ContactGeometry(self.chart, physical_screen=self.physical, extra_clearance_m=.004)
                            c = dict(kind='hold', note_ids=[task.id], start=t, end=(end + 1) / 1000)
                            if g.zone(c, end / 1000).is_empty:
                                earliest = math.ceil((task.note.seconds + task.note.hold - .020) * 1000) - 1
                                for shortened in range(end - 1, max(task.start, earliest) - 1, -1):
                                    if g.zone(c, (shortened + 1) / 1000 - 1e-7).is_empty:
                                        continue
                                    trial = self.corridor_path(task, shortened)
                                    if trial is not None:
                                        self.terminal_paths[task.id] = (trial, shortened)
                                        break
                        points, effective_end = self.terminal_paths[task.id]
                    if points is None:
                        self.rejected['judge_geometry'] += 1
                        continue
                    record = dict(pointer=(0 if side == -1 else 5) + FINGER_ORDER.index(finger),
                                  hand=hand, finger=finger, start=t, end=(effective_end + 1) / 1000,
                                  points=points, kind=task.note.type.name.lower(), note_ids=[task.id],
                                  beat=task.beat / 1000, burst_cost=burst_cost,
                                  effort=.018 if task.note.type == NoteType.DRAG else .055,
                                  planned=True)
                    record.update(self.timing_metadata(task))
                    if effective_end != end:
                        record['terminal_hold_release_ms'] = (task.note.seconds + task.note.hold - record['end']) * 1000
                        record['terminal_hold_release_reason'] = 'empty legal zone within final 20 ms'
                    violations, movement, posture = self.constraints(record, state, side, finger)
                    if over:
                        violations.append('burst_capacity')
                    if violations and not degraded and self.settings.fingering_objective != 'load':
                        self.rejected.update(violations)
                        continue
                    if 'occupied' in violations:
                        continue
                    x, y = points[0][1:]
                    center = task.line.pos(task.note.seconds, task.note.offset)
                    visual = ((x - center.real / self.screen.width) ** 2 +
                              (y - center.imag / self.screen.height) ** 2) * 80.
                    side_cost = max(0., -.03 - side * (x - .5)) ** 2 * 12
                    same_style = .08 if previous and task.note.type == NoteType.TAP and cps < 2 else 0.
                    if (previous and task.note.type == NoteType.DRAG and previous['kind'] == 'drag'
                            and t - previous['end'] <= .14
                            and math.dist(world_xy(previous['points'][-1][1:], self.physical),
                                          world_xy(points[0][1:], self.physical)) < .055):
                        # Safe Drag continuation avoids an unnecessary handover/lift; final bridging is checked below.
                        same_style += .4
                    if self.settings.fingering_objective != 'load':
                        record['effort'] += .02 * min(3., math.sqrt(movement))
                    own_f, own_b = load_at((record,), record['end'], hand, finger, self.settings)
                    after_f, after_b = min(1., base_after_f + own_f), max(0., base_after_b - (1 - own_b))
                    costs = dict(preference=PREFERENCE[finger], fatigue=.8 * fatigue + 1.5 * max(0., after_f - fatigue),
                                 burst=2 * burst_cost + .2 * (1 - after_b),
                                 hand=.6 * hand_fatigue + .3 * (1 - hand_burst),
                                 movement=.35 * movement, posture=.8 * posture,
                                 visual=visual, side=side_cost, habit=-same_style,
                                 degraded=1000. * len(violations))
                    if self.settings.fingering_objective == 'load':
                        for name in ('preference', 'movement', 'posture', 'visual', 'side', 'habit', 'degraded'):
                            costs[name] = 0.
                    record['decision'] = dict(costs=costs, fatigue_before=fatigue, fatigue_after=after_f,
                        burst_before=burst, burst_after=after_b, hand_fatigue=hand_fatigue,
                        finger_cps=cps, degraded=violations)
                    score = sum(costs.values())
                    candidates.append((score, record))
            candidates.sort(key=lambda x: (x[0], x[1]['points'][0][1:]))
            result.extend(candidates[:self.settings.candidates_per_finger])
        return result

    def run(self, console=None, *, on_unplanned=None, progress=None):
        """Plan contacts, optionally reporting unsolved notes to a draft editor.

        A reported note stays outside the search state so an unverified draft
        trajectory cannot influence sharing or occupancy of later contacts.
        Callers using this hook must retain the reported notes themselves.
        """
        beam = [State()]
        drag_representatives = {}
        swept = set()
        for index, task in enumerate(self.tasks):
            if progress:
                progress(index, len(self.tasks), task)
            if task.id in swept:
                continue
            if self.settings.judgement_windows and self.settings.fingers == ('index',):
                from flowphi.planning.sweeps import outward_group, sweep_states
                group = outward_group(self, task)
                if group:
                    states = [new for state in beam for new in sweep_states(self, group, state)]
                    if states:
                        beam = sorted(states, key=lambda s: s.score)[:self.settings.beam_width]
                        swept.update(t.id for t in group[-1])
                        if console: console.print(f'OUTWARD_SWEEP {task.beat / 1000:.3f}s notes={len(group[-1])}')
                        continue
            # Exact stacked Drags share one continuous touch. Keep every note ID,
            # but do not demand an additional anatomical finger for each copy.
            # Same line/time/offset guarantees the entire moving path is identical.
            if task.note.type == NoteType.DRAG:
                key = (id(task.line), task.note.seconds, task.note.offset)
                representative = drag_representatives.get(key)
                if representative is not None:
                    beam = [State(tuple(dict(c, note_ids=[*c['note_ids'], task.id])
                                        if representative in c['note_ids'] else c
                                        for c in state.contacts), state.score) for state in beam]
                    continue
            rejected_before = self.rejected.copy()
            expanded = []
            for state in beam:
                for cost, record in self.choices(task, state):
                    expanded.append(State((*state.contacts, record), state.score + cost))
            if not expanded and self.settings.allow_degraded:
                for state in beam:
                    for cost, record in self.choices(task, state, degraded=True):
                        expanded.append(State((*state.contacts, record), state.score + cost))
            if not expanded:
                for state in beam:
                    expanded.extend(self.share_flick_hold(task, state))
                    expanded.extend(self.share_drag(task, state))
                    expanded.extend(self.flick_on_contact(task, state))
                    if not self.settings.speed_limits:
                        expanded.extend(self.flick_after_hold(task, state))
                expanded = [s for s in expanded if self.contact_occupancy_valid(s.contacts)]
            if not expanded:
                if self.settings.judgement_windows:
                    for state in beam:
                        expanded.extend(self.window_choices(task, state))
            if not expanded:
                reasons = dict(self.rejected - rejected_before)
                if on_unplanned is None:
                    raise PlanningError(f'No feasible handcam fingering for note {task.id} at {task.beat / 1000:.3f}s; '
                                        f'rejections for this note: {reasons}')
                on_unplanned(task, reasons)
                if console:
                    console.print(f'flowphi: note {task.id} at {task.beat / 1000:.3f}s retained for editing; '
                                  f'rejections for this note: {reasons}')
                continue
            if task.note.type == NoteType.DRAG:
                drag_representatives[key] = task.id
            expanded.sort(key=lambda n: n.score)
            # Retain different fingering histories, not eight tiny variations of one fingertip.
            signatures, beam = Counter(), []
            for node in expanded:
                node = State(tuple(sorted(node.contacts, key=lambda c: (c['start'], c['pointer']))), node.score)
                signature = tuple((c['hand'], c['finger']) for c in node.contacts[-3:])
                if signatures[signature] >= 2:
                    continue
                signatures[signature] += 1
                beam.append(node)
                if len(beam) == self.settings.beam_width:
                    break
            if console and (index % 50 == 0 or index + 1 == len(self.tasks)):
                console.print(f'flowphi: {index + 1}/{len(self.tasks)} notes, {len(beam)} candidates')
        if progress:
            progress(len(self.tasks), len(self.tasks), None)
        return list(min(beam, key=lambda n: n.score).contacts)

    @staticmethod
    def contact_occupancy_valid(contacts):
        last = {}
        for c in sorted(contacts, key=lambda c: c['start']):
            key = c['hand'], c['finger']
            if last.get(key, -math.inf) > c['start'] + 1e-9: return False
            last[key] = c['end']
        return True

    def timing_metadata(self, task):
        original = self.original_tasks[task.id].note.seconds
        if abs(task.note.seconds - original) < 1e-9:
            return {}
        return dict(judgement_times={str(task.id): task.note.seconds},
                    judgement_time_reason='bounded_window' if self.settings.judgement_windows else 'invisible_drag')

    def window_choices(self, task, state):
        """Try free-finger times inside the documented judgment windows.

        Every candidate keeps the original chart and Hold tail. The exported
        note-time mapping is validated again against the original chart.
        """
        from flowphi.planning.timing import PLANNING_WINDOW_MS
        original = self.original_tasks[task.id]
        limit = PLANNING_WINDOW_MS[task.note.type]
        origin = original.note.seconds
        last = {}
        for c in state.contacts:
            key = (c['hand'], c['finger'])
            last[key] = max(last.get(key, 0.), c['end'])
        beats = {round(origin * 1000) + delta for delta in range(5, limit + 1, 5)}
        for end in last.values():
            # Separate UP and the next DOWN, and retain 25 ms of Flick approach.
            beats.add(math.ceil((end + .002) * 1000) + (25 if task.note.type == NoteType.FLICK else 0))
        result = []
        for beat in sorted(beats, key=lambda t: abs(t / 1000 - origin)):
            if abs(beat / 1000 - origin) > limit / 1000 + 1e-9 or beat == task.beat:
                continue
            start = beat - 25 if task.note.type == NoteType.FLICK else beat
            end = original.end if task.note.type == NoteType.HOLD else beat + (45 if task.note.type == NoteType.FLICK else 70)
            if end <= start:
                continue
            note = Note(task.note.type, beat / 1000,
                      max(0., origin + original.note.hold - beat / 1000) if task.note.type == NoteType.HOLD else task.note.hold,
                      task.note.offset)
            candidate = Task(task.id, note, task.line, start, end, beat)
            for cost, c in self.choices(candidate, state, degraded=True):
                ordered = tuple(sorted((*state.contacts, c), key=lambda old: (old['start'], old['pointer'])))
                result.append(State(ordered, state.score + cost))
            if result:
                break
        return result

    def flick_after_hold(self, task, state):
        """A completed Hold may release at its tail before a same-beat Flick.

        Only discard the planner's extra UP millisecond after the actual Hold.
        Start the Flick at its original judgement time, retaining a real swipe.
        """
        if task.note.type != NoteType.FLICK or task.start >= task.beat:
            return []
        result = []
        notes = {nid: item.note for nid, item in self.original_tasks.items()}
        for index, old in enumerate(state.contacts):
            if old['kind'] != 'hold' or not task.start / 1000 < old['end'] <= task.beat / 1000 + .0011:
                continue
            # Note IDs refer to the original line/note traversal, not sorted tasks.
            if any(notes[n].type != NoteType.HOLD for n in old['note_ids']):
                continue
            release = math.ceil(max(notes[n].seconds + notes[n].hold for n in old['note_ids']) * 1000 - 1e-7)
            if release > task.beat or release / 1000 <= old['start']:
                continue
            trimmed = copy.deepcopy(old)
            trimmed['end'] = release / 1000
            last = round(trimmed['end'] - .001, 3)
            trimmed['points'] = [p for p in old['points'] if p[0] < last] + [[last, *point_at(old['points'], last)]]
            trimmed['hold_tail_dwell_trim_ms'] = (old['end'] - trimmed['end']) * 1000
            contacts = list(state.contacts); contacts[index] = trimmed
            delayed = replace(task, start=task.beat)
            for cost, c in self.choices(delayed, State(tuple(contacts), state.score), degraded=True):
                if (c['hand'], c['finger']) != (old['hand'], old['finger']):
                    continue
                c['flick_start_delayed_ms'] = task.beat - task.start
                result.append(State((*contacts, c), state.score + cost))
        return result

    def share_flick_hold(self, task, state):
        """One same-beat Tap or Hold may share a Flick's pressed fingertip.

        Preserve the swipe and validate the intersection of both judgement
        bands. This never allocates an extra finger or skips either note.
        """
        if task.note.type not in (NoteType.HOLD, NoteType.TAP):
            return []
        from flowphi.planning.contacts import ContactGeometry, refine_contact, validate_contact
        geometry = ContactGeometry(self.chart, physical_screen=self.physical, extra_clearance_m=.004)
        result = []
        for index, old in enumerate(state.contacts):
            if old['kind'] != 'flick' or abs(old['beat'] - task.beat / 1000) > .0011 or old['end'] <= task.start / 1000:
                continue
            if any(geometry.notes[n][1].type in (NoteType.HOLD, NoteType.TAP) for n in old['note_ids']):
                continue
            seed = self.project(self.zone(task, task.start),
                                (old['points'][-1][1] * self.screen.width, old['points'][-1][2] * self.screen.height))
            if seed is None:
                continue
            end = task.end if task.note.type == NoteType.HOLD else task.start + 12
            tail = self.path(task, seed, end)
            if tail is None:
                tail = self.corridor_path(task, end)
            if tail is None:
                continue
            merged = copy.deepcopy(old)
            merged.update(note_ids=[*old['note_ids'], task.id], end=max(old['end'], (end + 1) / 1000),
                          joint_note_coverage=True, shared_contact='same_beat_flick_press')
            merged['points'] = old['points'] + [p for p in tail if p[0] > old['points'][-1][0]]
            points, _ = refine_contact(geometry, merged)
            if points is None or validate_contact(geometry, merged, points):
                continue
            if self.blocks.path_violations(points, merged['start'], merged['end']):
                continue
            merged['points'] = points
            contacts = list(state.contacts)
            contacts[index] = merged
            result.append(State(tuple(contacts), state.score))
        return result

    def share_drag(self, task, state):
        """Keep an occupied touch down when it can also cover the next Drag."""
        if task.note.type != NoteType.DRAG:
            return []
        from flowphi.planning.contacts import ContactGeometry, refine_contact, validate_contact
        geometry = ContactGeometry(self.chart, physical_screen=self.physical, extra_clearance_m=.004)
        result = []
        for index, old in enumerate(state.contacts):
            if not old['start'] <= task.beat / 1000 < old['end']:
                continue
            merged = copy.deepcopy(old)
            merged.update(note_ids=[*old['note_ids'], task.id], joint_note_coverage=True,
                          end=max(old['end'], (task.beat + 12) / 1000), shared_contact='covered_drag')
            if task.id in self.judgement_times:
                merged.setdefault('judgement_times', {})[str(task.id)] = self.judgement_times[task.id]
            if old['kind'] == 'tap':
                merged['kind'] = 'drag'
            if merged['end'] > old['end']:
                merged['points'].append([merged['end'] - .001, *old['points'][-1][1:]])
            window = dict(merged, note_ids=[task.id], kind='drag',
                        start=max(old['start'], task.note.seconds - .0011),
                        end=min(old['end'], task.note.seconds + .011))
            if merged['end'] == old['end'] and not validate_contact(geometry, window, old['points']):
                contacts = list(state.contacts); contacts[index] = merged
                result.append(State(tuple(contacts), state.score))
                continue
            extra = set()
            for attempt in range(3):
                points, _ = refine_contact(geometry, merged, extra_times=extra)
                if points is None:
                    break
                bad = validate_contact(geometry, merged, points, limit=100)
                if not bad and not self.blocks.path_violations(points, merged['start'], merged['end']):
                    merged['points'] = points
                    contacts = list(state.contacts); contacts[index] = merged
                    result.append(State(tuple(contacts), state.score))
                    break
                extra.update(bad)
        return result

    def flick_on_contact(self, task, state):
        """Slide an already pressed fingertip, retaining any live Hold bands."""
        if task.note.type != NoteType.FLICK:
            return []
        from flowphi.planning.contacts import ContactGeometry, refine_contact, validate_contact
        geometry = ContactGeometry(self.chart, physical_screen=self.physical, extra_clearance_m=.004)
        result = []
        for index, old in enumerate(state.contacts):
            if not old['start'] <= task.start / 1000 < old['end']:
                continue
            side = -1 if old['hand'] == 'left' else 1
            for seed in self.seeds(task, state, side, old['finger']):
                for direction in (1, -1, 2, -2):
                    stroke = self.path(task, seed, task.end, direction)
                    if stroke is None:
                        continue
                    merged = copy.deepcopy(old)
                    merged.update(note_ids=[*old['note_ids'], task.id], kind='flick', beat=task.beat / 1000,
                                  end=max(old['end'], (task.end + 1) / 1000), joint_note_coverage=True,
                                  shared_contact='flick_on_held_contact')
                    merged['points'] = ([p for p in old['points'] if p[0] < stroke[0][0]] + stroke +
                                      [p for p in old['points'] if p[0] > stroke[-1][0]])
                    points, _ = refine_contact(geometry, merged)
                    if points is None or validate_contact(geometry, merged, points):
                        continue
                    if self.blocks.path_violations(points, merged['start'], merged['end']):
                        continue
                    merged['points'] = points
                    contacts = list(state.contacts); contacts[index] = merged
                    result.append(State(tuple(contacts), state.score))
                    break
                if result:
                    break
        return result

    def join_drags(self, contacts):
        """Keep safe short Drag chains down; validate the newly occupied connecting interval."""
        result, last = [], {}
        for c in contacts:
            key = c['hand'], c['finger']
            index = last.get(key)
            previous = result[index] if index is not None else None
            if (previous and previous['kind'] == c['kind'] == 'drag'
                    and 0 < c['start'] - previous['end'] <= .14
                    and sum(self.original_tasks[nid].note.type in (NoteType.TAP, NoteType.HOLD)
                            for nid in set(previous['note_ids'] + c['note_ids'])) <= 1):
                merged = dict(previous, end=c['end'], points=previous['points'] + c['points'],
                              note_ids=previous['note_ids'] + c['note_ids'],
                              judgement_times={**previous.get('judgement_times', {}), **c.get('judgement_times', {})},
                              effort=previous['effort'] + c['effort'])
                others = tuple(other for other in contacts if other is not c and other is not previous
                               and not set(other['note_ids']).intersection(merged['note_ids']))
                side = -1 if c['hand'] == 'left' else 1
                errors, _, _ = self.constraints(merged, State(others), side, c['finger'])
                if not errors:
                    result[index] = merged
                    continue
            last[key] = len(result)
            result.append(c)
        return sorted(result, key=lambda c: (c['start'], c['pointer']))


def compact_stationary(points):
    """Remove redundant constant MOVE samples, preserving every transition's endpoints."""
    if all(p[1:] == points[0][1:] for p in points):
        return points[:1]
    result = []
    for p in points:
        if len(result) > 1 and result[-2][1:] == result[-1][1:] == p[1:]:
            result[-1] = p
        else:
            result.append(p)
    return result


def plan(chart, config=None, console=None, *, settings=None, profile=None, view_width=None, progress=None):
    started = time.monotonic()
    settings, profile = settings or Settings(), profile or default_profile()
    planner = Planner(chart, settings, profile)
    contacts = planner.run(console, progress=progress)
    if not contacts:
        raise PlanningError('No notes to plan')
    # Emit lifecycle order exactly as PSAP will be read, including same-time chords.
    contacts.sort(key=lambda c: (c['start'], c['pointer']))
    contacts = planner.join_drags(contacts)
    for c in contacts:
        if c['kind'] != 'flick':
            c['points'] = compact_stationary(c['points'])
    rests = finish_motion(contacts, profile, planner.physical, view_width or settings.screen_width_m / .82)
    frames = defaultdict(list)
    for c in contacts:
        for i, (t, x, y) in enumerate(c['points']):
            frames[round(t * 1000)].append(VirtualTouchEvent(complex(x * chart.width, y * chart.height),
                TouchAction.DOWN if i == 0 else TouchAction.MOVE, c['pointer']))
        x, y = c['points'][-1][1:]
        frames[round(c['end'] * 1000)].append(VirtualTouchEvent(complex(x * chart.width, y * chart.height),
            TouchAction.UP, c['pointer']))
    answer = [(ms, sorted(events, key=lambda e: (0 if e.action == TouchAction.UP else 1, e.pointer_id)))
              for ms, events in sorted(frames.items())]
    content = dump_data(planner.screen, answer)
    # Roundtrip normalization removes float multiply/divide discrepancies before strict binding checks.
    for c in contacts:
        c['points'] = [[t, x * chart.width / chart.width, y * chart.height / chart.height] for t, x, y in c['points']]
    metadata = dict(version=VERSION, algorithm=5, timebase='chart_seconds', coordinates='normalized_screen',
        psap_sha256=hashlib.sha256(content).hexdigest(), contacts=contacts, hand_rest=rests, profile=profile,
        physical_screen=list(planner.physical), settings=vars(settings), palm_motion='v4',
        diagnostics=dict(notes=len(planner.tasks), contacts=len(contacts),
            joined_drags=sum(len(c['note_ids']) - 1 for c in contacts), seconds=time.monotonic() - started,
            rejections=dict(planner.rejected), degraded=[dict(note_ids=c['note_ids'], time=c['start'],
                reasons=c['decision']['degraded']) for c in contacts if c['decision']['degraded']],
            finger_usage=dict(Counter(f'{c["hand"]}:{c["finger"]}' for c in contacts)),
            native_ap_validated=False))
    return planner.screen, answer, metadata
