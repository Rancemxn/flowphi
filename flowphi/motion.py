from __future__ import annotations

import bisect
import copy
import hashlib
import math

from flowphi.pose import motion_style, validate_pose

VERSION = 1
FINGER_ORDER = ('index', 'middle', 'ring', 'thumb', 'little')


class IntervalIndex:
    def __init__(self, intervals):
        self.rows = sorted((start, end, i, value) for i, (start, end, value) in enumerate(intervals))
        self.starts = [row[0] for row in self.rows]
        self.ends = []
        maximum = -math.inf
        for _, end, _, _ in self.rows:
            maximum = max(maximum, end)
            self.ends.append(maximum)
        self.cached_time, self.cached_rows = None, []

    def at(self, t):
        if self.cached_time != t:
            low = bisect.bisect_left(self.ends, t)
            high = bisect.bisect_right(self.starts, t)
            rows = [row for row in self.rows[low:high] if row[1] >= t]
            rows.sort(key=lambda row: row[2])
            self.cached_time, self.cached_rows = t, [row[3] for row in rows]
        return self.cached_rows


class MotionIndex:
    def __init__(self, contacts, rests=()):
        self.hands = {hand: [c for c in contacts if c['hand'] == hand] for hand in ('left', 'right')}
        self.guides, self.rests = {}, {}
        for hand, events in self.hands.items():
            self.guides[hand] = IntervalIndex((c.get('prepare', c['start'] - .2),
                c.get('release_until', c['end'] + .15), c) for c in events)
            self.rests[hand] = IntervalIndex((-math.inf if g['intro'] else g['start'],
                math.inf if g['outro'] else g['end'], g) for g in rests if g['hand'] == hand)


def smooth(u):
    u = min(1., max(0., u))
    return u ** 3 * (10 + u * (-15 + 6 * u))


def blend(a, b, u):
    return tuple(x + (y - x) * u for x, y in zip(a, b))


def transfer_progress(u):
    u = min(1., max(0., u))
    ramp = .2
    if u < ramp:
        v = u / ramp
        return ramp * (v ** 3 - .5 * v ** 4) / (1 - ramp)
    if u > 1 - ramp:
        return 1 - transfer_progress(1 - u)
    return (u - ramp / 2) / (1 - ramp)


def stroke_shape(u, job=None):
    if job and job.get('palm_lift_mode') in ('accent', 'gentle'):
        u = min(1., max(0., u))
        peak = job.get('stroke_peak_phase', .62)
        phase = u / peak if u <= peak else (1 - u) / (1 - peak)
        # Spend more of the gap preparing, then make a quicker downstroke.
        # Both ends still meet the authored contact exactly, at zero height.
        return math.sin(math.pi * .5 * phase) ** 2
    return smooth(u / .28) * smooth((1 - u) / .32)


def stroke_lift(following, gap, job):
    factor = job.get('pose_lift_scale', 1.)
    if job.get('palm_lift_mode') == 'gentle':
        density = smooth((.22 - gap) / .16)
        # Bound the planned stroke before solving, so a short gap cannot leave
        # the wrist chasing a peak or snapping down at the next contact.
        speed = job.get('palm_strike_speed', .9)
        ratio = max(.01, job.get('palm_lift_ratio', .70))
        low = job.get('palm_lift_low', .030)
        high = job.get('palm_lift_high', .032)
        return factor * min(job.get('lift_height', .034), low + (high - low) * density, speed * gap / (math.pi * ratio))
    if job.get('palm_lift_mode') == 'accent':
        density = smooth((.22 - gap) / .16)
        return factor * min(job.get('lift_height', .056), .040 + .014 * density)
    if job.get('finger_motion') == 'whole_finger':
        return factor * min(job.get('lift_height', .028), max(following['lift'] * 1.6, .018 * min(1., gap / .12)))
    return factor * min(job.get('lift_height', .025), following['lift'])


def travel_time(distance, speed=1.5, acceleration=18.):
    """Minimum duration for a rest-to-rest quintic, including its peak derivatives."""
    return max(1.875 * distance / speed, math.sqrt(5.773503 * distance / acceleration))


def world_xy(point, screen):
    return ((point[0] - .5) * screen[0], (.5 - point[1]) * screen[1])


def palm_height(scale):
    return .055 * scale / .29


def palm_offsets(profile, contact_height=.0005):
    result = {}
    height = palm_height(profile['scale'])
    for name, finger in profile['fingers'].items():
        offset, base = finger['offset'], finger['base']
        dx, dy = offset[0] - base[0], offset[1] - base[1]
        radius = math.sqrt(max(1e-6, (finger['length'] * .92) ** 2 -
                               (height + base[2] - contact_height) ** 2))
        ratio = min(1., radius / max(1e-6, math.hypot(dx, dy)))
        side, finger_name = name.split(':')
        result[int(side), finger_name] = (base[0] + dx * ratio, base[1] + dy * ratio, offset[2])
    return result


def point_at(points, t):
    i = max(0, bisect.bisect_right(points, t, key=lambda p: p[0]) - 1)
    a = points[i]
    if i + 1 == len(points) or t <= a[0]:
        return tuple(a[1:])
    b = points[i + 1]
    return blend(a[1:], b[1:], (t - a[0]) / (b[0] - a[0]))


def default_profile(scale=.27):
    factor = scale / .27
    result = {}
    for side in (-1, 1):
        for finger, x, y, length in (('index', -.025, .12, .09), ('middle', 0, .135, .10),
                                     ('ring', .023, .125, .095), ('little', .043, .10, .075),
                                     ('thumb', .0015, .098, .088)):
            result[f'{side}:{finger}'] = dict(offset=[side * x * factor, y * factor, -.045 * factor],
                base=[side * x * factor, .065 * factor, -.005 * factor], length=length * factor)
            if finger == 'thumb':
                result[f'{side}:{finger}']['base'] = [-side * .030 * factor, .0324 * factor, -.0167 * factor]
    return dict(version=1, scale=scale, fingers=result, source='approximate fallback; ten fingers')


def attach_plan(content, contacts, plan, time_offset=0.):
    if plan.get('version') != VERSION or plan.get('algorithm') != 5:
        raise ValueError('Unsupported handcam motion plan version/algorithm')
    if plan.get('timebase') != 'chart_seconds' or plan.get('coordinates') != 'normalized_screen':
        raise ValueError('Unsupported handcam plan timebase/coordinates')
    if 'pose' in plan:
        validate_pose(plan['pose'])
    if hashlib.sha256(content).hexdigest() != plan.get('psap_sha256'):
        raise ValueError('Motion plan belongs to a different PSAP')
    screen = plan.get('physical_screen', [])
    if len(screen) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in screen):
        raise ValueError('Invalid physical screen in motion plan')
    for gap in plan.get('hand_rest', []):
        knots = gap.get('knots', [])
        if (gap.get('hand') not in ('left', 'right') or len(knots) < 2
                or any(len(p) != 4 or not all(math.isfinite(v) for v in p) for p in knots)
                or any(a[0] >= b[0] for a, b in zip(knots, knots[1:]))):
            raise ValueError('Invalid wrist rest trajectory in motion plan')
    for guide in plan.get('pose_guides', []):
        if (guide.get('hand') not in ('left', 'right') or
                any(not isinstance(guide.get(k), (int, float)) or not math.isfinite(guide[k])
                    for k in ('prepare', 'start', 'end', 'release', 'yaw')) or
                not guide['prepare'] <= guide['start'] < guide['end'] <= guide['release'] or abs(guide['yaw']) > .65):
            raise ValueError('Invalid authored palm orientation guide')
    for name, value in plan.get('pose_style', {}).items():
        maximum = {'palm_lift_low': .1, 'palm_lift_high': .1, 'palm_strike_speed': 5., 'transfer_sway': .05, 'finger_lateral_limit': .5}.get(name)
        if maximum is None or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError('Invalid authored pose style')
    records = plan.get('contacts', [])
    if len(records) != len(contacts):
        raise ValueError('Motion plan contact count differs from PSAP')
    result, occupied = [], {}
    for source, record in zip(contacts, records):
        key = (record.get('hand'), record.get('finger'))
        if key[0] not in ('left', 'right') or key[1] not in FINGER_ORDER:
            raise ValueError('Invalid physical finger in motion plan')
        if (any(not isinstance(record.get(name), (float, int)) or not math.isfinite(record[name])
                for name in ('prepare', 'release_until', 'lift'))
                or record['prepare'] > record['start'] or record['release_until'] < record['end']
                or not 0 <= record['lift'] <= .1):
            raise ValueError('Invalid finger preparation/release in motion plan')
        if (source['pointer'] != record.get('pointer')
                or source['start'] != record['start'] + time_offset
                or source['end'] != record['end'] + time_offset
                or len(source['points']) != len(record['points'])
                or any(a[0] != b[0] + time_offset or a[1:] != b[1:]
                       for a, b in zip(source['points'], record['points']))):
            raise ValueError('Motion plan lifecycle/trajectory differs from PSAP')
        if source['end'] <= source['start'] or occupied.get(key, -math.inf) > source['start'] + 1e-9:
            raise ValueError('Motion plan overlaps or fails to release a physical finger')
        occupied[key] = source['end']
        merged = copy.deepcopy(record)
        merged.update(source)
        for name in ('prepare', 'release_until', 'beat'):
            if name in merged:
                merged[name] += time_offset
        merged['planned'] = True
        result.append(merged)
    return result


def finger_pose(events, t, rest, job, key=None):
    i = bisect.bisect_right(events, t, key=lambda c: c['start']) - 1
    previous = events[i] if i >= 0 else None
    following = events[i + 1] if i + 1 < len(events) else None
    h = job['contact_height']
    if previous and t < previous['end']:
        return (*world_xy(point_at(previous['points'], t), job['screen']), h), 1., True
    if key is None and events:
        key = (events[0]['hand'], events[0]['finger'])
    if key is not None:
        job = motion_style(job, *key)
    if previous and following and following['start'] - previous['end'] <= job.get('stroke_gap_limit', .75):
        a = world_xy(previous['points'][-1][1:], job['screen'])
        b = world_xy(following['points'][0][1:], job['screen'])
        gap = following['start'] - previous['end']
        u = min(1., max(0., (t - previous['end']) / gap))
        xy = blend(a, b, transfer_progress(u) if job.get('palm_lift_mode') == 'gentle' else smooth(u))
        sway = job.get('transfer_sway', 0.) * (1 - .65 * smooth((.22 - gap) / .16))
        side = -1 if following['hand'] == 'left' else 1
        xy = (xy[0] + side * sway * stroke_shape(u, job), xy[1])
        # One continuous local stroke, with exact release and landing boundaries.
        lift = stroke_lift(following, gap, job)
        if job.get('finger_motion') == 'whole_finger':
            # Leave the surface promptly, retain a readable raised phase, then
            # return before the next DOWN. Contact times remain unchanged.
            height = lift * stroke_shape(u, job)
        else:
            height = lift * (math.sin(math.pi * smooth(u)) if following.get('lift_curve') == 'v4'
                             else math.sin(math.pi * u) ** 2)
        return (*xy, h + height), 1., False
    if following and t >= following['prepare']:
        u = (t - following['prepare']) / max(1e-6, following['start'] - following['prepare'])
        target = world_xy(following['points'][0][1:], job['screen'])
        xy = blend(rest[:2], target, smooth(u))
        lift = min(job.get('lift_height', .025), following['lift']) * job.get('pose_lift_scale', 1.)
        z = rest[2] + (h - rest[2]) * smooth(u) + .25 * lift * math.sin(math.pi * u) ** 2
        return (*xy, z), smooth(min(1., u / .6)), False
    if previous and t < previous['release_until']:
        u = (t - previous['end']) / max(1e-6, previous['release_until'] - previous['end'])
        a = (*world_xy(previous['points'][-1][1:], job['screen']), h)
        return blend(a, rest, smooth(u)), 1 - smooth(u), False
    return tuple(rest), 0., False


def rest_intervals(contacts, profile, screen, view_width):
    result = []
    offsets = palm_offsets(profile)
    height = palm_height(profile['scale'])
    for side, hand in ((-1, 'left'), (1, 'right')):
        events = sorted((c for c in contacts if c['hand'] == hand), key=lambda c: c['start'])
        clusters = []
        for c in events:
            if clusters and c['start'] <= clusters[-1][1]:
                clusters[-1][1] = max(clusters[-1][1], c['end'])
                clusters[-1][2].append(c)
            else:
                clusters.append([c['start'], c['end'], [c]])
        def anchor(c, end=False):
            xy = world_xy(c['points'][-1 if end else 0][1:], screen)
            offset = offsets[side, c['finger']]
            return [xy[0] - offset[0], xy[1] - offset[1], height]
        reach_y = max((v['offset'][1] for k, v in profile['fingers'].items() if k.startswith(f'{side}:')),
                      default=.18 * profile['scale'] / .27)
        # Leave through the bottom edge; preserve the hand's horizontal region.
        off = [side * .085, -view_width * 9 / 32 - reach_y - .04, height + .008 * profile['scale'] / .27]
        for i in range(len(clusters) + 1):
            past = clusters[i - 1] if i else None
            future = clusters[i] if i < len(clusters) else None
            start = past[1] if past else (future[0] - 5 if future else -5.)
            end = future[0] if future else start + 5.
            a = anchor(max(past[2], key=lambda c: c['end']), True) if past else off
            b = anchor(future[2][0]) if future else off
            gap = end - start
            mode = 'hover'
            parked = None
            for name, target, threshold in (
                    ('withdraw', off, 2.8),
                    ('lower', [(a[0] + b[0]) / 2, -view_width * 9 / 32 - reach_y - .025,
                               off[2]], 1.0)):
                outward = max(.24, travel_time(math.dist(a, target), .65, 4.))
                inward = max(.32, travel_time(math.dist(target, b), .65, 4.))
                if (not past or not future or gap >= threshold) and gap >= outward + inward + .22:
                    mode, parked = name, target
                    break
            if parked is None:
                knots = [[start, *a], [end, *b]]
            else:
                knots = [[start, *a], [start + outward, *parked], [end - inward, *parked], [end, *b]]
            result.append(dict(hand=hand, start=start, end=end, mode=mode, knots=knots,
                               intro=past is None, outro=future is None,
                               from_finger=max(past[2], key=lambda c: c['end'])['finger'] if past else None,
                               to_finger=future[2][0]['finger'] if future else None))
    return result


def wrist_pose(contacts, rests, side, t, job, offsets, neutral_offsets=None, index=None):
    job = motion_style(job, side)
    hand = 'left' if side == -1 else 'right'
    events = index.hands[hand] if index else [c for c in contacts if c['hand'] == hand]
    gap = next((g for g in (index.rests[hand].at(t) if index else rests) if g['hand'] == hand and
                (g['intro'] or t >= g['start']) and (g['outro'] or t < g['end'])), None)
    if gap:
        knots = [list(p) for p in gap['knots']]
        if neutral_offsets is not None:
            # Endpoints follow the rotated palm; the parked point stays below
            # the screen. Blend this correction over the same return interval.
            for knot_index, name in ((0, 'from_finger'), (-1, 'to_finger')):
                if gap.get(name):
                    key = side, gap[name]
                    for axis in (0, 1):
                        knots[knot_index][axis + 1] += neutral_offsets[key][axis] - offsets[key][axis]
        i = max(0, bisect.bisect_right(knots, t, key=lambda p: p[0]) - 1)
        if i == len(knots) - 1:
            return tuple(knots[-1][1:])
        a, b = knots[i:i + 2]
        u = (t - a[0]) / (b[0] - a[0])
        progress = transfer_progress(u) if job.get('palm_lift_mode') == 'gentle' and gap['mode'] == 'hover' else smooth(u)
        position = blend(a[1:], b[1:], progress)
        if gap['mode'] == 'hover':
            duration = gap['end'] - gap['start']
            sway = job.get('transfer_sway', 0.) * (1 - .65 * smooth((.22 - duration) / .16))
            position = (position[0] + side * sway * stroke_shape(u, job), *position[1:])
        if job.get('palm_motion') == 'v4' and gap['mode'] == 'hover':
            following = next((c for c in events if c['start'] >= gap['end'] - 1e-7), None)
            duration = gap['end'] - gap['start']
            if job.get('palm_lift_ratio') is not None and following and duration <= job.get('stroke_gap_limit', .75):
                lift = stroke_lift(following, duration, job) * job['palm_lift_ratio']
                if job.get('palm_lift_mode') not in ('accent', 'gentle'):
                    # Legacy restrained mode reserves speed for XY movement.
                    lift = min(lift, .8 * job.get('wrist_speed', .9) * .28 * duration / 1.875)
                height = lift * stroke_shape(u, job)
            else:
                lift = min(job.get('lift_height', .025), following.get('lift', .012) if following else .012) * job.get('pose_lift_scale', 1.)
                height = .30 * lift * math.sin(math.pi * smooth(u))
            position = (*position[:2], position[2] + height)
        return position
    anchors = []
    candidates = index.guides[hand].at(t) if index else events
    active_events = [c for c in candidates if c['start'] <= t < c['end']]
    guided = active_events or candidates
    if active_events:
        # A held finger must not prevent the palm from preparing the other
        # fingers for a chord. Reach projection in the baker keeps the held pad
        # fixed while the wrist approaches the next available finger's anchor.
        active_fingers = {c['finger'] for c in active_events}
        upcoming = {}
        for c in candidates:
            if c['finger'] not in active_fingers and c['start'] > t and c.get('prepare', c['start'] - .2) <= t:
                old = upcoming.get(c['finger'])
                if old is None or c['start'] < old['start']:
                    upcoming[c['finger']] = c
        guided = [*active_events, *upcoming.values()]
    for c in guided:
        prepare = c.get('prepare', c['start'] - .2)
        release = c.get('release_until', c['end'] + .15)
        if prepare <= t <= release:
            weight = smooth((t - prepare) / max(1e-6, c['start'] - prepare)) if t < c['start'] else (
                1 - smooth((t - c['end']) / max(1e-6, release - c['end'])) if t >= c['end'] else 1.)
            xy = world_xy(point_at(c['points'], t), job['screen'])
            offset = offsets[side, c['finger']]
            anchors.append(((xy[0] - offset[0], xy[1] - offset[1]), weight))
    total = sum(w for _, w in anchors)
    height = palm_height(job['hand_scale']) if job.get('palm_motion') == 'v4' else .040 * job['hand_scale'] / .27
    if not total:
        return side * .09, -.16, height
    return (sum(p[0] * w for p, w in anchors) / total,
            sum(p[1] * w for p, w in anchors) / total, height)


def finish_motion(contacts, profile, screen, view_width):
    for hand in ('left', 'right'):
        for finger in FINGER_ORDER:
            events = [c for c in contacts if c['hand'] == hand and c['finger'] == finger]
            activity = 0.
            for i, c in enumerate(events):
                previous = events[i - 1] if i else None
                following = events[i + 1] if i + 1 < len(events) else None
                intervals = [other['start'] - c['start'] for other in (following,) if other]
                if previous:
                    intervals.append(c['start'] - previous['start'])
                spacing = min(intervals, default=1.)
                density = max(0., min(1., (.35 - spacing) / .25))
                activity = max(density, activity * math.exp(-max(.001, spacing) / .4))
                distance = math.dist(world_xy(previous['points'][-1][1:], screen),
                                     world_xy(c['points'][0][1:], screen)) if previous else .08
                gap = c['start'] - previous['end'] if previous else .3
                # Fourth-round stroke, only 12% larger, still shortened for bursts.
                c['lift'] = 1.12 * min(.025, .008 + .18 * distance) * min(1., max(0., gap) / .12)
                c['lift_curve'] = 'v4'
                prepare = .28 + .06 * (1 - activity)
                earliest = previous['end'] if previous else -math.inf
                if previous and c['start'] - previous['end'] > .75:
                    earliest += min(.18, (c['start'] - previous['end']) * .30)
                c['prepare'] = max(earliest, c['start'] - prepare)
            for i, c in enumerate(events):
                following = events[i + 1] if i + 1 < len(events) else None
                c['release_until'] = min(following['prepare'] if following else math.inf, c['end'] + .18)
    return rest_intervals(contacts, profile, screen, view_width)
