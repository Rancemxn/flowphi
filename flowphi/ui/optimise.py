from __future__ import annotations

import numpy as np


class AssignmentUnavailable(ValueError):
    def __init__(self, row):
        self.row = row
        super().__init__('暂未找到满足当前动作安排的指法')


def constrained_beam(unary, neighbors, allowed, width):
    n, k = unary.shape
    histories = np.zeros((1, n), dtype=np.int8)
    totals = np.zeros(1)
    for i in range(n):
        values = np.tile(unary[i], (len(histories), 1)) + totals[:, None]
        for j, matrix in neighbors[i]:
            if j < i:
                values += matrix[:, histories[:, j]].T
        values[:, ~allowed[i]] = np.inf
        ranked = np.argsort(values, axis=None)
        ranked = ranked[np.isfinite(values.ravel()[ranked])][:width]
        if not len(ranked):
            raise AssignmentUnavailable(i)
        parents, labels = np.divmod(ranked, k)
        totals = values[parents, labels]
        histories = histories[parents].copy()
        histories[:, i] = labels
    return histories[int(np.argmin(totals))]


def _reassign_contacts(document, selected, *, width=32, max_contacts=1200, hands=None, fingers=None):
    from flowphi.planning.costs import assignment_keys, graph, objective
    from flowphi.planning.load import assign_load, score_labels
    selected = set(selected)
    if not selected or any(not isinstance(i, int) or not 0 <= i < len(document) for i in selected):
        raise ValueError('请先选择需要重新分配手指的动作')
    mutable = {i for i in selected if not document.contact(i)['assignment_locked']}
    if not mutable:
        raise ValueError('所选动作的指法都已锁定；需要重新安排时，请先解锁这些动作')
    if not 1 <= width <= 128 or not 32 <= max_contacts:
        raise ValueError('重新分配的候选数或动作数上限无效，请检查性能设置')
    low = min(document.contact(i)['start'] for i in mutable)
    high = max(document.contact(i)['end'] for i in mutable)
    context = set(document.index.query(low - .5, high + .5)) | selected
    before, after = {}, {}
    for i in document.sorted_ids(range(len(document))):
        c = document.contact(i)
        key = c['hand'], c['finger']
        if c['end'] <= low:
            before[key] = i
        elif c['start'] >= high and key not in after:
            after[key] = i
    context.update(before.values())
    context.update(after.values())
    if len(context) > max_contacts:
        raise ValueError(f'本次需计算 {len(context)} 段动作，超过手动设置的 {max_contacts} 段上限；请在性能设置中提高上限，或分段安排')
    ids = document.sorted_ids(context)
    contacts = [dict(document.contact(i)) for i in ids]
    for c in contacts:
        # The editor domains below are authoritative in both balanced and load
        # modes; old soft-cost annotations cannot bypass or reintroduce a lock.
        c.pop('fixed_finger', None)
        c.pop('manual_hand', None)
    plan = document.baseline
    keys = assignment_keys(plan)
    if hands is not None and (not hands or any(h not in ('left', 'right') for h in hands)):
        raise ValueError('可用手无效')
    if fingers is not None and (not fingers or any(f not in document.allowed_fingers for f in fingers)):
        raise ValueError('可用手指无效')
    unary, neighbors = graph(contacts, plan)
    allowed = np.ones((len(ids), len(keys)), dtype=bool)
    current = np.array([keys.index((c['hand'], c['finger'])) for c in contacts])
    for row, i in enumerate(ids):
        if i in mutable:
            allowed[row] = [(hands is None or h in hands) and (fingers is None or f in fingers) for h, f in keys]
        else:
            allowed[row] = False
            allowed[row, current[row]] = True
    strict_neighbors = [[(j, np.where(matrix >= 1e8, np.inf, matrix)) for j, matrix in edges] for edges in neighbors]
    load_only = plan['settings'].get('fingering_objective') == 'load'
    if load_only:
        labels = assign_load(contacts, keys, plan['settings'], strict_neighbors, width, allowed=allowed)
        old_score = score_labels(contacts, keys, plan['settings'], current)[0]
        new_score = score_labels(contacts, keys, plan['settings'], labels)[0]
    else:
        try:
            labels = constrained_beam(unary, strict_neighbors, allowed, width)
        except AssignmentUnavailable as exc:
            c = contacts[exc.row]
            locked = sum(i not in mutable for i in ids)
            raise ValueError(f"暂未排出合适的指法：谱面时间 {c['start']:.3f} 秒附近，动作 #{ids[exc.row] + 1}。"
                f'当前有 {locked} 段边界或锁定动作保持原样。'
                '请检查这一段的触点复用、按下/松开时间和滑动轨迹；这里只分配现有动作，不会调整它们的时间和轨迹，不能据此判定谱面无解。') from None
        old_score = objective(current, unary, strict_neighbors)
        new_score = objective(labels, unary, strict_neighbors)
    old_valid = all(allowed[row, label] for row, label in enumerate(current)) and all(
        matrix[current[row], current[j]] < 1e8 for row, edges in enumerate(neighbors) for j, matrix in edges)
    if old_valid and old_score <= new_score:
        labels = current
        new_score = old_score
    for row, i in enumerate(ids):
        if i not in mutable and labels[row] != current[row]:
            raise RuntimeError('局部重新分配改变了边界或锁定内容')
    assignments = [[i, *keys[int(labels[row])]] for row, i in enumerate(ids)
                   if i in mutable and labels[row] != current[row]]
    return dict(assignments=assignments, context_contacts=len(ids), selected_contacts=len(selected),
                changed_contacts=len(assignments), score_before=float(old_score) if np.isfinite(old_score) else None,
                score_after=float(new_score), mode='load' if load_only else 'balanced')


def reassign(document, selected, *, width=32, max_contacts=1200, chart=None, hands=None, fingers=None):
    from flowphi.ui.contact_state import resolve_contacts
    from flowphi.ui.model import PlanDocument
    selected = set(selected)
    if not selected or any(not isinstance(i, int) or not 0 <= i < len(document) for i in selected):
        raise ValueError('请先选择需要重新分配手指的动作')
    contacts, groups, problems = resolve_contacts(document, chart)
    problem = next((p for p in problems if selected.intersection(p['contacts'])), None)
    if problem:
        raise ValueError(f"谱面时间 {problem['time']:.3f} 秒的动作 #{problem['contacts'][0] + 1} 没有接上前面的触点。"
                         '请把它和要接续的动作安排给同一手指；确需独立点击时再设为“重新按下”。')
    chosen = {groups[i] for i in selected}
    for i in chosen:
        if not set(contacts[i]['editor_contacts']) <= selected:
            ids = '、'.join(f'#{j + 1}' for j in contacts[i]['editor_contacts'])
            raise ValueError(f'{ids} 共用一次持续按住请一起选择这些动作，或先将后段设为“重新按下”')
    for c in contacts:
        c['assignment_locked'] = any(document.contact(i)['assignment_locked'] for i in c['editor_contacts'])
        c['fixed_finger'] = [c['hand'], c['finger']] if c['assignment_locked'] else None
    physical = PlanDocument(dict(document.baseline, contacts=contacts), chart=document.chart, copy_baseline=False)
    result = _reassign_contacts(physical, chosen, width=width, max_contacts=max_contacts, hands=hands, fingers=fingers)
    result['assignments'] = [[j, hand, finger] for i, hand, finger in result['assignments']
                             for j in contacts[i]['editor_contacts']]
    result['changed_contacts'] = len(result['assignments'])
    result['selected_contacts'] = len(selected)
    return result
