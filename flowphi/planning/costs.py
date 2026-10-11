import math
import numpy as np
from flowphi.motion import FINGER_ORDER, palm_offsets, point_at, travel_time, world_xy

KEYS = [(hand, finger) for hand in ('left', 'right') for finger in ('index', 'middle', 'ring', 'little', 'thumb')]


def assignment_keys(plan):
    allowed = plan.get('settings', {}).get('fingers', [k[1] for k in KEYS])
    keys = [k for k in KEYS if k[1] in allowed]
    if not keys:
        raise ValueError('No enabled fingers for assignment')
    return keys

def pair_cost(a, b, screen, offsets, comfort=None, keys=KEYS, load_only=False):
    comfort = comfort or {}
    samehand = np.array([[x[0] == y[0] for y in keys] for x in keys])
    samefinger = np.eye(len(keys), dtype=bool)
    gap = b['start'] - a['end']
    cost = np.zeros((len(keys), len(keys)))
    if gap < -1e-8:
        low, high = b['start'], min(a['end'], b['end']) - 1e-7
        # Linear contact paths attain their extreme relative X at these knots.
        times = sorted({low, high, *(p[0] for c in (a, b) for p in c['points'] if low < p[0] < high)})
        pa = np.array([world_xy(point_at(a['points'], t), screen) for t in times])
        pb = np.array([world_xy(point_at(b['points'], t), screen) for t in times])
        span = np.zeros_like(cost)
        reverse = np.zeros_like(cost)
        natural_x = offsets[:, None, 0] - offsets[None, :, 0]
        for left, right in zip(pa, pb):
            anchors = (left - offsets)[:, None, :] - (right - offsets)[None, :, :]
            span = np.maximum(span, np.linalg.norm(anchors, axis=2))
            reverse = np.maximum(reverse, -(left[0] - right[0]) * np.sign(natural_x))
        ordered = np.array([[x[1] != 'thumb' and y[1] != 'thumb' and x != y for y in keys] for x in keys])
        if load_only:
            cost[samehand & ordered & (reverse > .002)] = 1e8
            cost[samefinger] = 1e12
            return cost
        cost += samehand * (180 * (span / .05)**2 + 2e6 * np.maximum(0, span - .04)**2)
        cost += samehand * ordered * (reverse > .002) * 1e8
        if comfort:
            # Compatible wrist anchors keep adjacent fingers relaxed instead of
            # forcing a splayed grip merely to preserve a finger preference.
            cost += samehand * 350 * (span / .04)**2
            min_distance = float(np.min(np.linalg.norm(pa - pb, axis=1)))
            cost += samehand * 3e6 * max(0, comfort.get('finger_spacing_m', .018) - min_distance)**2
        # Crossing complete hands is also expensive; a central relay remains allowed.
        for ka, (ha, _) in enumerate(keys):
            for kb, (hb, _) in enumerate(keys):
                if ha != hb:
                    crossed = np.max((pa[:, 0] - pb[:, 0]) * (1 if ha == 'left' else -1))
                    cost[ka, kb] += 2e6 * max(0, crossed - .025)**2
                    if comfort:
                        anchors_a = pa - offsets[ka]
                        anchors_b = pb - offsets[kb]
                        separation = np.min((anchors_b[:, 0] - anchors_a[:, 0]) * (1 if ha == 'left' else -1))
                        cost[ka, kb] += 2e5 * max(0, comfort.get('hand_spacing_m', .065) - separation)**2
        cost[samefinger] = 1e12
    else:
        if load_only:
            return cost
        pa = np.array(world_xy(a['points'][-1][1:], screen))
        pb = np.array(world_xy(b['points'][0][1:], screen))
        distance = float(np.linalg.norm(pa - pb))
        required = travel_time(distance, 1.2, 12.)
        cost += samefinger * (6 * (required / max(.008, gap))**2 + 600 * max(0, required - gap)**2 / .01)
        anchors = np.linalg.norm((pa - offsets)[:, None, :] - (pb - offsets)[None, :, :], axis=2)
        cost += samehand * 30 * (np.maximum(0, anchors - .015) / max(.06, gap))**2
        cost += samefinger * 6 * math.exp(-max(0, gap) / .65) * min(1., a['end'] - a['start'] + .05)
    return cost

def graph(contacts, plan):
    keys = assignment_keys(plan)
    measured = palm_offsets(plan['profile'])
    offsets = np.array([measured[-1 if h == 'left' else 1, f][:2] for h, f in keys])
    unary = np.zeros((len(contacts), len(keys)))
    neighbors = [[] for _ in contacts]
    comfort = plan.get('comfort', {})
    load_only = plan.get('settings', {}).get('fingering_objective') == 'load'
    priorities = comfort.get('finger_costs', {'index': 0., 'middle': .1, 'ring': 1., 'little': 12., 'thumb': 35.})
    for i, c in enumerate(contacts):
        x = sum(p[1] for p in c['points']) / len(c['points'])
        for k, (hand, finger) in enumerate(keys):
            side = -1 if hand == 'left' else 1
            unary[i, k] = 0. if load_only else 3000 * max(0, -side * (x - .5) - .06)**2 + priorities[finger]
            if not load_only and c.get('manual_hand') and hand != c['manual_hand']:
                unary[i, k] += 1e9
            if not load_only and c.get('fixed_finger') and (hand, finger) != tuple(c['fixed_finger']):
                unary[i, k] += 1e12
    for i, a in enumerate(contacts):
        for j in range(i + 1, len(contacts)):
            b = contacts[j]
            if b['start'] > a['end'] + .45:
                break
            if load_only and b['start'] >= a['end']:
                break
            matrix = pair_cost(a, b, plan['physical_screen'], offsets, comfort, keys, load_only)
            neighbors[i].append((j, matrix))
            neighbors[j].append((i, matrix.T))
    return unary, neighbors

def objective(labels, unary, neighbors):
    return float(sum(unary[i, k] for i, k in enumerate(labels)) +
                 sum(matrix[labels[i], labels[j]] for i, items in enumerate(neighbors) for j, matrix in items if j > i))
