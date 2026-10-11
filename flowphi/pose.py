from __future__ import annotations

from bisect import bisect_right
import copy
import math


HANDS = ('left', 'right')
FINGERS = ('index', 'middle', 'ring', 'little', 'thumb')
TARGETS = ('hand', *FINGERS)
HAND_CHANNELS = {
    'x': (-.2, .2, 0.), 'y': (-.2, .2, 0.), 'z': (-.06, .15, 0.),
    'yaw': (-math.pi, math.pi, 0.), 'pitch': (-.8, .8, 0.), 'roll': (-.8, .8, 0.),
    'spread': (-.5, .7, 0.), 'curl': (-.6, 1.2, 0.),
    'lift_scale': (0., 3., 1.), 'peak_shift': (-.4, .3, 0.),
}
FINGER_CHANNELS = {
    'curl': (-.8, 1.4, 0.), 'middle_curl': (-.8, 1.4, 0.), 'tip_curl': (-.8, 1.4, 0.),
    'spread': (-.65, .65, 0.), 'raise': (-.02, .06, 0.),
    'lift_scale': (0., 3., 1.), 'peak_shift': (-.4, .3, 0.),
}


def channels(target):
    return HAND_CHANNELS if target == 'hand' else FINGER_CHANNELS


def neutral_values(target):
    return {name: bounds[2] for name, bounds in channels(target).items()}


def empty_pose():
    return dict(version=1, defaults={}, clips=[])


def number(value, low, high):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and low <= value <= high


def validate_values(values, target):
    limits = channels(target)
    if not isinstance(values, dict) or set(values) - limits.keys():
        raise ValueError('姿态包含不支持的参数')
    if any(not number(value, *limits[name][:2]) for name, value in values.items()):
        raise ValueError('姿态参数超出允许范围')


def validate_pose(pose):
    if not isinstance(pose, dict) or pose.get('version') != 1 or set(pose) - {'version', 'defaults', 'clips'}:
        raise ValueError('不支持此姿态轨道格式')
    defaults, clips = pose.get('defaults', {}), pose.get('clips', [])
    if not isinstance(defaults, dict) or not isinstance(clips, list) or len(clips) > 512:
        raise ValueError('姿态轨道数据无效或片段超过 512 个')
    for key, values in defaults.items():
        if key not in {f'{hand}:{target}' for hand in HANDS for target in TARGETS}:
            raise ValueError('姿态默认值包含未知的手或手指')
        validate_values(values, key.split(':')[1])
    seen, total = set(), 0
    for clip in clips:
        if not isinstance(clip, dict) or set(clip) - {'id', 'hand', 'target', 'start', 'end', 'fade_in', 'fade_out', 'curve', 'enabled', 'keys'}:
            raise ValueError('姿态片段包含不支持的字段')
        identity = clip.get('id')
        if not isinstance(identity, str) or not identity or len(identity) > 128 or identity in seen:
            raise ValueError('姿态片段编号无效或重复')
        seen.add(identity)
        if clip.get('hand') not in HANDS or clip.get('target') not in TARGETS:
            raise ValueError('姿态片段的手或手指无效')
        if (not number(clip.get('start'), -3600, 86400) or not number(clip.get('end'), -3600, 86400)
                or clip['end'] - clip['start'] < .001 - 1e-9):
            raise ValueError('姿态片段结束时刻必须晚于开始时刻')
        if any(not number(clip.get(name, 0.), 0., clip['end'] - clip['start']) for name in ('fade_in', 'fade_out')):
            raise ValueError('姿态过渡时长不能超过片段长度')
        if clip.get('curve', 'smooth') not in ('smooth', 'linear') or not isinstance(clip.get('enabled', True), bool):
            raise ValueError('姿态片段的过渡方式无效')
        keys = clip.get('keys')
        if not isinstance(keys, list) or not keys or len(keys) > 2000:
            raise ValueError('每个姿态片段需要 1–2000 个关键帧')
        previous = -math.inf
        for key in keys:
            if not isinstance(key, dict) or set(key) != {'time', 'values'}:
                raise ValueError('姿态关键帧格式无效')
            if not number(key['time'], clip['start'], clip['end']) or key['time'] <= previous:
                raise ValueError('姿态关键帧必须按时间排列，且位于片段内')
            validate_values(key['values'], clip['target'])
            previous = key['time']
        total += len(keys)
    if total > 10000:
        raise ValueError('姿态关键帧总数超过 10000 个')


def normalize_pose(pose=None):
    result = copy.deepcopy(empty_pose() if pose is None else pose)
    validate_pose(result)
    result.setdefault('defaults', {})
    result.setdefault('clips', [])
    return result


def smooth(u):
    u = max(0., min(1., u))
    return u ** 3 * (10 + u * (-15 + 6 * u))


class PoseSampler:
    def __init__(self, pose=None):
        self.pose = normalize_pose(pose)
        self.tracks = {(hand, target): [] for hand in HANDS for target in TARGETS}
        self.key_times = set()
        for clip in self.pose['clips']:
            if clip.get('enabled', True):
                self.tracks[clip['hand'], clip['target']].append(clip)
                self.key_times.update((clip['start'], clip['end'],
                    clip['start'] + clip.get('fade_in', 0.), clip['end'] - clip.get('fade_out', 0.)))
                self.key_times.update(key['time'] for key in clip['keys'])

    def values(self, hand, target, t):
        result = neutral_values(target)
        result.update(self.pose['defaults'].get(f'{hand}:{target}', {}))
        for clip in self.tracks[hand, target]:
            if not clip['start'] <= t < clip['end']:
                continue
            fade_in, fade_out = clip.get('fade_in', 0.), clip.get('fade_out', 0.)
            weight = min(smooth((t - clip['start']) / fade_in) if fade_in else 1.,
                         smooth((clip['end'] - t) / fade_out) if fade_out else 1.)
            keys = clip['keys']
            at = max(0, bisect_right(keys, t, key=lambda k: k['time']) - 1)
            a, b = keys[at], keys[min(at + 1, len(keys) - 1)]
            u = max(0., min(1., (t - a['time']) / (b['time'] - a['time']))) if b['time'] > a['time'] else 0.
            if clip.get('curve', 'smooth') == 'smooth':
                u = smooth(u)
            for name in a['values'].keys() | b['values'].keys():
                value = a['values'].get(name, result[name]) * (1 - u) + b['values'].get(name, result[name]) * u
                result[name] += (value - result[name]) * weight
        return result

    def frame(self, t):
        return {f'{hand}:{target}': self.values(hand, target, t) for hand in HANDS for target in TARGETS}


def frame_values(job, side, target='hand'):
    hand = 'left' if side in (-1, 'left') else 'right'
    return job.get('pose_frame', {}).get(f'{hand}:{target}', neutral_values(target))


def motion_style(job, side, finger=None):
    hand = frame_values(job, side)
    part = frame_values(job, side, finger) if finger else neutral_values('index')
    factor = hand['lift_scale'] * part['lift_scale']
    phase = hand['peak_shift'] + part['peak_shift']
    if factor == 1. and phase == 0.:
        return job
    return dict(job, pose_lift_scale=factor,
                stroke_peak_phase=max(.12, min(.90, job.get('stroke_peak_phase', .62) + phase)))


def idle_weight(events, t):
    at = bisect_right(events, t, key=lambda c: c['start']) - 1
    previous = events[at] if at >= 0 else None
    following = events[at + 1] if at + 1 < len(events) else None
    if previous and previous['start'] <= t <= previous['end']:
        return 0.
    release = smooth((t - previous['end']) / .08) if previous else 1.
    landing = smooth((following['start'] - t) / .08) if following else 1.
    return min(release, landing)


def finger_deltas(frame, side, finger, length):
    hand = frame_values(dict(pose_frame=frame), side)
    part = frame_values(dict(pose_frame=frame), side, finger)
    curl = hand['curl'] + part['curl']
    raise_angle = math.asin(max(-.6, min(.6, part['raise'] / max(.03, length * .8))))
    spread = part['spread'] + hand['spread'] * {'index': -1., 'middle': 0., 'ring': .6, 'little': 1., 'thumb': 1.}[finger]
    return ((raise_angle - curl * .3, 0., -side * spread),
            (-curl * .65 - part['middle_curl'], 0., 0.),
            (-curl * .45 - part['tip_curl'], 0., 0.))


def rotation_matrix(yaw=0., pitch=0., roll=0.):
    cz, sz = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(roll), math.sin(roll)
    return ((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx))


def rotate(matrix, point):
    return tuple(sum(a * b for a, b in zip(row, point)) for row in matrix)


def finger_planes(center, matrix, finger, limit=None, axes=None):
    if finger == 'thumb':
        return []
    limit = max(.015, (limit if limit is not None else
        {'index': .35, 'middle': .22, 'ring': .28, 'little': .35}[finger]) - .025)
    forward, lateral = axes or ((0., 1., 0.), (1., 0., 0.))
    forward, lateral = rotate(matrix, forward), rotate(matrix, lateral)
    result = []
    for sign in (-1, 1):
        normal = tuple(sign * a - math.tan(limit) * b for a, b in zip(lateral, forward))
        size = math.sqrt(sum(v * v for v in normal))
        normal = tuple(v / size for v in normal)
        result.append((normal, sum(a * b for a, b in zip(normal, center))))
    return result


def contact_reach(length, finger):
    return length * (.92 if finger == 'thumb' else .94)


def wrist_constraints(reference, values, constraints, planes):
    strength = smooth(max(math.sqrt(sum(values[n] ** 2 for n in ('x', 'y', 'z'))) / .012,
                          max(abs(values[n]) for n in ('yaw', 'pitch', 'roll')) / .15))
    spheres = [(center, radius + max(0., math.dist(reference, center) - radius) * (1 - strength))
               for center, radius in constraints]
    bounds = [(normal, offset - max(0., offset - sum(a * b for a, b in zip(reference, normal))) * (1 - strength))
              for normal, offset in planes]
    return spheres, bounds


def fit_wrist(requested, constraints, minimum=.018, planes=()):
    root = tuple(requested)
    for _ in range(48):
        before = root
        for center, radius in constraints:
            distance = math.dist(root, center)
            if distance > radius:
                root = tuple(c + (p - c) * radius / distance for p, c in zip(root, center))
        for normal, offset in planes:
            missing = offset - sum(a * b for a, b in zip(root, normal))
            if missing > 0:
                root = tuple(p + a * missing for p, a in zip(root, normal))
        root = (*root[:2], max(minimum, root[2]))
        if math.dist(before, root) < 1e-8:
            break
    error = max((math.dist(root, center) - radius for center, radius in constraints), default=0.)
    error = max(error, max((offset - sum(a * b for a, b in zip(root, normal)) for normal, offset in planes), default=0.))
    return root, max(0., error)


def mirrored_values(values, target):
    result = dict(values)
    if target == 'hand':
        for name in ('x', 'yaw', 'roll'):
            if name in result:
                result[name] *= -1
    return result
