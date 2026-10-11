from __future__ import annotations

import math


def simplify(points, tolerance, screen):
    if len(points) < 3:
        return points
    errors = [tolerance / (size * math.sqrt(2)) for size in screen]
    result = [points[0]]
    anchor = points[0]
    bounds = [(-math.inf, math.inf), (-math.inf, math.inf)]
    for j in range(1, len(points)):
        point = points[j]
        dt = point[0] - anchor[0]
        slopes = [(point[axis + 1] - anchor[axis + 1]) / dt for axis in range(2)]
        if any(not low <= slope <= high for slope, (low, high) in zip(slopes, bounds)):
            anchor = points[j - 1]
            result.append(anchor)
            dt = point[0] - anchor[0]
            slopes = [(point[axis + 1] - anchor[axis + 1]) / dt for axis in range(2)]
            bounds = [(-math.inf, math.inf), (-math.inf, math.inf)]
        bounds = [(max(low, slope - error / dt), min(high, slope + error / dt))
                  for slope, error, (low, high) in zip(slopes, errors, bounds)]
    result.append(points[-1])
    return result


def edit_points(points, *, operation, rows=None, x=None, y=None, tolerance=.0005, screen=(.28, .1575)):
    """Return a fresh trajectory; disjoint row selections remain separate runs."""
    if operation not in ('translate', 'set', 'freeze', 'straighten', 'simplify', 'delete'):
        raise ValueError('未知的批量轨迹操作')
    for value in (x, y):
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError('轨迹坐标必须是有限数值')
    if operation in ('translate', 'set') and x is None and y is None:
        raise ValueError('请至少选择 X 或 Y 一个方向')
    if operation == 'simplify' and (not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('减少控制点时，允许的位置变化必须大于零')
    selected = list(range(len(points))) if rows is None else sorted(set(rows))
    if any(not isinstance(j, int) or not 0 <= j < len(points) for j in selected):
        raise ValueError('控制点选择已失效，请重新选择')
    if not selected:
        return points
    if operation == 'delete':
        removed = set(selected) - {0}
        return [list(p) for j, p in enumerate(points) if j not in removed]
    if operation in ('translate', 'set'):
        result = [list(p) for p in points]
        for j in selected:
            for axis, value in enumerate((x, y), 1):
                if value is not None:
                    coordinate = result[j][axis] + value if operation == 'translate' else value
                    # A shared translation at the edge can round a few ulps
                    # outside the screen. Do not clip meaningful displacements.
                    result[j][axis] = min(1., max(0., coordinate)) if -1e-12 <= coordinate <= 1 + 1e-12 else coordinate
        return result
    # Preserve every unselected point and both endpoints of each selected run.
    runs = []
    low = previous = selected[0]
    for j in selected[1:]:
        if j != previous + 1:
            runs.append((low, previous + 1))
            low = j
        previous = j
    runs.append((low, previous + 1))
    result, cursor = [], 0
    for low, high in runs:
        result.extend(points[cursor:low])
        part = points[low:high]
        if operation == 'freeze':
            part = [list(part[0])] if len(part) == 1 else [list(part[0]), [part[-1][0], *part[0][1:]]]
        elif operation == 'straighten':
            part = part if len(part) < 3 else [part[0], part[-1]]
        else:
            part = simplify(part, tolerance, screen)
        result.extend(part)
        cursor = high
    result.extend(points[cursor:])
    return [list(p) for p in result]
