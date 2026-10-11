from __future__ import annotations

from bisect import bisect_right
import copy
import math

from flowphi.motion import (MotionIndex, default_profile, finger_pose, finish_motion, palm_offsets,
                            point_at, smooth, world_xy, wrist_pose)
from flowphi.pose import (FINGERS, PoseSampler, contact_reach, finger_deltas, finger_planes, fit_wrist, frame_values,
                         idle_weight, rotate, rotation_matrix, wrist_constraints)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scaled(a, value):
    return tuple(x * value for x in a)


def unit(a):
    return scaled(a, 1 / max(1e-12, math.dist(a, (0., 0., 0.))))


def chain_to(points, target, lengths):
    points = list(points)
    base = points[0]
    if math.dist(base, target) >= sum(lengths):
        direction = unit(sub(target, base))
        return [base, *[add(base, scaled(direction, sum(lengths[:i]))) for i in (1, 2, 3)]]
    for _ in range(32):
        points[-1] = target
        for i in (2, 1, 0):
            points[i] = add(points[i + 1], scaled(unit(sub(points[i], points[i + 1])), lengths[i]))
        points[0] = base
        for i in range(3):
            points[i + 1] = add(points[i], scaled(unit(sub(points[i + 1], points[i])), lengths[i]))
        if math.dist(points[-1], target) < 1e-6:
            break
    return points


def idle_chain(base, tip, length, deltas, reference=None):
    sizes = [length * value for value in (.43, .34, .23)]
    if reference is None:
        delta = sub(tip, base)
        points = [base, *[add(add(base, scaled(delta, u)), (0., 0., length * .22 * math.sin(math.pi * u)))
                         for u in (.43, .77, 1.)]]
        points = chain_to(points, tip, sizes)
    else:
        points = reference
    if not any(abs(value) > 1e-9 for delta in deltas for value in delta):
        return points, sizes
    rotations = []
    result = [base]
    for i, angles in enumerate(deltas):
        rotations.append(rotation_matrix(angles[2], angles[0], angles[1]))
        segment = sub(points[i + 1], points[i])
        for matrix in reversed(rotations):
            segment = rotate(matrix, segment)
        result.append(add(result[-1], segment))
    return result, sizes


class PosePreview:
    def __init__(self):
        self.document = self.chart = None
        self.motion_revision = self.pose_revision = -1
        self.override = None
        self.sampler = PoseSampler()
        self.warning = ''

    def set_document(self, document, chart=None):
        if document is not self.document or chart is not self.chart:
            self.motion_revision = self.pose_revision = -1
        self.document, self.chart = document, chart
        self.override = None
        self.pose_revision = -1

    def set_pose(self, pose):
        self.override = pose
        self.sampler = PoseSampler(pose if pose is not None else self.document.pose if self.document else None)
        self.pose_revision = self.document.pose_revision if self.document else -1

    def refresh(self):
        doc = self.document
        if doc is None:
            return
        if self.override is None and self.pose_revision != doc.pose_revision:
            self.sampler = PoseSampler(doc.pose)
            self.pose_revision = doc.pose_revision
        if self.motion_revision == doc.motion_revision:
            return
        from flowphi.ui.contact_state import resolve_contacts
        self.warning = ''
        try:
            contacts, _, problems = resolve_contacts(doc, self.chart)
            self.contacts = copy.deepcopy(contacts)
            if problems:
                self.warning = '触控动作有冲突，接触预览以当前分配估算'
        except (ValueError, KeyError, IndexError):
            self.contacts = [copy.deepcopy(doc.contact(i)) for i in range(len(doc))]
            self.warning = '触控动作有冲突，接触预览以当前分配估算'
        self.contacts.sort(key=lambda c: (c['start'], c['hand'], c['finger']))
        self.profile = default_profile(doc.baseline['profile']['scale'])
        self.profile['fingers'].update(doc.baseline['profile'].get('fingers', {}))
        screen = doc.baseline['physical_screen']
        self.rests = finish_motion(self.contacts, self.profile, screen, .28 / .82)
        self.index = MotionIndex(self.contacts, self.rests)
        self.offsets = palm_offsets(self.profile)
        self.rest_chains = {(int(key.split(':')[0]), key.split(':')[1]):
            idle_chain(tuple(data['base']), tuple(data['offset']), data['length'], [(0., 0., 0.)] * 3)[0]
            for key, data in self.profile['fingers'].items()}
        self.tracks = {(side, finger): [c for c in self.contacts if c['hand'] == hand and c['finger'] == finger]
                       for side, hand in ((-1, 'left'), (1, 'right')) for finger in FINGERS}
        if any(a['end'] > b['start'] + 1e-9 for events in self.tracks.values() for a, b in zip(events, events[1:])):
            self.warning = '同一手指有重叠动作，请在“问题”页检查指法'
        self.job = dict(screen=screen, hand_scale=self.profile['scale'], contact_height=.0005,
            palm_motion=doc.baseline.get('palm_motion', 'v4'), finger_motion='whole_finger', palm_lift_mode='gentle',
            palm_lift_ratio=.70, palm_strike_speed=1.8, stroke_peak_phase=.62, palm_lift_low=.042,
            palm_lift_high=.035, stroke_gap_limit=1., transfer_sway=.004, lift_height=.034)
        self.job.update(doc.baseline.get('pose_style', {}))
        self.motion_revision = doc.motion_revision

    def sample(self, key, t, rest, job):
        events = self.tracks[key]
        i = bisect_right(events, t, key=lambda c: c['start']) - 1
        if i >= 0 and abs(t - events[i]['end']) < 1e-9:
            return (*world_xy(point_at(events[i]['points'], t), job['screen']), job['contact_height']), 1., True
        return finger_pose(events, t, rest, job, key)

    def frame(self, t):
        if self.document is None:
            return None
        self.refresh()
        frame = self.sampler.frame(t)
        job = dict(self.job, pose_frame=frame)
        result = dict(screen=job['screen'], hands={}, warning=self.warning)
        for side, hand in ((-1, 'left'), (1, 'right')):
            keys = [(side, finger) for finger in FINGERS]
            rest_root = (side * .09, -.16, .055 * job['hand_scale'] / .29)
            samples = {key: self.sample(key, t, add(rest_root, self.profile['fingers'][f'{side}:{key[1]}']['offset']), job)
                       for key in keys}
            aiming = [(pos, weight) for pos, weight, _ in samples.values() if weight > .001]
            yaw = 0.
            if aiming:
                total = sum(w for _, w in aiming)
                aim = tuple(sum(p[axis] * w for p, w in aiming) / total for axis in range(3))
                yaw = max(-.45, min(.45, -math.atan2(aim[0] - side * .12, max(.08, aim[1] + .24))))
            for guide in self.document.baseline.get('pose_guides', []):
                if guide['hand'] == hand and guide['prepare'] <= t <= guide['release']:
                    weight = (smooth((t - guide['prepare']) / max(1e-6, guide['start'] - guide['prepare'])) if t < guide['start']
                              else smooth((guide['release'] - t) / max(1e-6, guide['release'] - guide['end'])) if t > guide['end'] else 1.)
                    yaw += (guide['yaw'] - yaw) * weight
            values = frame_values(job, side)
            rotation = rotation_matrix(yaw + values['yaw'], values['pitch'], values['roll'])
            offsets = {key: rotate(rotation, self.offsets[key]) for key in self.offsets}
            bases = {key: rotate(rotation, self.profile['fingers'][f'{side}:{key[1]}']['base']) for key in keys}
            root = wrist_pose(self.contacts, self.rests, side, t, job, offsets, self.offsets, self.index)
            manual = any(abs(values[name]) > 1e-9 for name in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'))
            constraints = [(sub(pos, bases[key]), self.profile['fingers'][f'{side}:{key[1]}']['length'] * .94)
                           for key, (pos, _, down) in samples.items() if down]
            active = [sub(pos, offsets[key]) for key, (pos, _, down) in samples.items() if down]
            if active:
                anchor = tuple(sum(p[a] for p in active) / len(active) for a in range(3))
                delta = (root[0] - anchor[0], root[1] - anchor[1], 0.)
                radius = .015 * job['hand_scale'] / .29
                if math.dist(delta, (0., 0., 0.)) > radius:
                    xy = add(anchor, scaled(unit(delta), radius))
                    root = (xy[0], xy[1], root[2])
            for _ in range(16):
                for center, radius in constraints:
                    r = math.sqrt(max(.000001, radius ** 2 - (root[2] - center[2]) ** 2))
                    delta = (root[0] - center[0], root[1] - center[1], 0.)
                    if math.dist(delta, (0., 0., 0.)) > r:
                        xy = add(center, scaled(unit(delta), r))
                        root = (xy[0], xy[1], root[2])
            requested, residual = root, 0.
            if manual:
                requested = add(root, (values['x'], values['y'], values['z']))
                constraints = [(sub(pos, bases[key]), contact_reach(self.profile['fingers'][f'{side}:{key[1]}']['length'], key[1]))
                               for key, (pos, _, down) in samples.items() if down]
                planes = [plane for key, (pos, _, down) in samples.items() if down
                          for plane in finger_planes(sub(pos, bases[key]), rotation, key[1], job.get('finger_lateral_limit'))]
                constraints, planes = wrist_constraints(root, values, constraints, planes)
                root, residual = fit_wrist(requested, constraints, planes=planes)
            fingers = {}
            for key in keys:
                finger = key[1]
                data = self.profile['fingers'][f'{side}:{finger}']
                tip_rest = add(root, rotate(rotation, data['offset']))
                tip_rest = (*tip_rest[:2], max(.012, tip_rest[2]))
                pos, weight, down = self.sample(key, t, tip_rest, job)
                idle = idle_weight(self.tracks[key], t)
                deltas = finger_deltas(frame, side, finger, data['length'])
                adjusted = [scaled(delta, idle) for delta in deltas]
                local, sizes = idle_chain(tuple(data['base']), tuple(data['offset']), data['length'], adjusted, self.rest_chains[key])
                joints = [add(root, rotate(rotation, p)) for p in local]
                influence = weight * (1 - idle if any(abs(v) > 1e-9 for row in deltas for v in row) else 1.)
                if influence > 1e-5:
                    solved = chain_to(joints, pos, sizes)
                    joints = [add(scaled(a, 1 - influence), scaled(b, influence)) for a, b in zip(joints, solved)]
                fingers[finger] = dict(joints=joints, down=down, idle=idle, target=pos,
                    error=math.dist(joints[-1], pos) if down else 0.)
            result['hands'][hand] = dict(root=root, requested=requested, rotation=rotation, fingers=fingers,
                correction=math.dist(root, requested), reach_error=residual,
                palm=[add(root, rotate(rotation, p)) for p in ((-.035, .012, 0.), (.035, .012, 0.),
                    tuple(self.profile['fingers'][f'{side}:little']['base']), tuple(self.profile['fingers'][f'{side}:index']['base']))])
        return result
