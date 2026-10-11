import math
import numpy as np

from flowphi.planning.planner import COMFORT, PEAK, Settings, load_at


def score_labels(contacts, keys, options, labels):
    settings = Settings(**options)
    history, decisions = [], []
    total = 0.
    for source, label in zip(contacts, labels):
        c = dict(source)
        hand, finger = keys[int(label)]
        c.update(hand=hand, finger=finger)
        t, end = c['start'], c['end']
        f, b = load_at(history, t, hand, finger, settings)
        hf, hb = load_at(history, t, hand, settings=settings)
        previous = next((old for old in reversed(history) if (old['hand'], old['finger']) == (hand, finger)), None)
        cps = 0. if previous is None else 1 / max(.001, t - previous['start'])
        comfort = COMFORT[finger] * (1 - .3 * f - .15 * hf)
        kind = c.get('load_kind', c['kind'])
        c['burst_cost'] = .1 * max(0., cps / comfort - 1) if kind == 'tap' else 0.
        c['effort'] = .018 if c['kind'] == 'drag' else .055
        af, ab = load_at(history, end, hand, finger, settings)
        ownf, ownb = load_at([c], end, hand, finger, settings)
        af, ab = min(1., af + ownf), max(0., ab - (1 - ownb))
        costs = dict(fatigue=.8 * f + 1.5 * max(0., af - f),
                     burst=2 * c['burst_cost'] + .2 * (1 - ab), hand=.6 * hf + .3 * (1 - hb),
                     preference=0., movement=0., posture=0., visual=0., side=0., habit=0., degraded=0.)
        limit = PEAK[finger] * (.65 + .35 * b) * (1 - .2 * hf)
        degraded = ['burst_capacity'] if kind in ('tap', 'hold') and cps > limit else []
        decisions.append(dict(costs=costs, fatigue_before=f, fatigue_after=af,
            burst_before=b, burst_after=ab, hand_fatigue=hf, finger_cps=cps,
            effort=c['effort'], burst_cost=c['burst_cost'], degraded=degraded))
        total += sum(costs.values())
        history.append(c)
    return total, decisions


def advance(values, start, end, contacts, active, histories, tau):
    result = values * math.exp(-(end - start) / tau)
    rows = np.arange(len(histories))
    for j in active:
        stop = min(end, contacts[j]['end'])
        if stop > start:
            contribution = .11 * tau * (1 - math.exp(-(stop - start) / tau)) * math.exp(-(end - stop) / tau)
            result[rows, histories[:, j]] += contribution
    return result


def assign_load(contacts, keys, options, neighbors, width=96, *, allowed=None):
    settings = Settings(**options)
    n, k = len(contacts), len(keys)
    if allowed is not None and np.shape(allowed) != (n, k):
        raise ValueError('Allowed finger domains do not match contacts and keys')
    histories = np.zeros((1, n), dtype=np.int8)
    fatigue = np.zeros((1, k)); debt = fatigue.copy()
    last_start = np.full((1, k), -np.inf)
    totals = np.zeros(1)
    active = []
    previous_time = contacts[0]['start'] if contacts else 0.
    comforts = np.array([COMFORT[f] for _, f in keys])
    hands = {h: np.array([hand == h for hand, _ in keys]) for h in ('left', 'right')}
    for i, c in enumerate(contacts):
        t, end = c['start'], c['end']
        fatigue = advance(fatigue, previous_time, t, contacts, active, histories, settings.fatigue_seconds)
        debt *= math.exp(-(t - previous_time) / settings.recovery_seconds)
        active = [j for j in active if contacts[j]['end'] > t]
        f = np.minimum(1., fatigue)
        hf, hb = np.zeros_like(f), np.zeros_like(f)
        for mask in hands.values():
            hf[:, mask] = np.minimum(1., .5 * fatigue[:, mask].sum(axis=1))[:, None]
            hb[:, mask] = np.maximum(0., 1 - .55 * debt[:, mask].sum(axis=1))[:, None]
        cps = 1 / np.maximum(.001, t - last_start)
        comfort = comforts[None, :] * (1 - .3 * f - .15 * hf)
        burst = .1 * np.maximum(0., cps / comfort - 1) if c.get('load_kind', c['kind']) == 'tap' else np.zeros_like(f)
        effort = .018 if c['kind'] == 'drag' else .055
        future = advance(fatigue, t, end, contacts, active, histories, settings.fatigue_seconds)
        # Match the existing sum of separately clamped prior/own loads.
        own = effort * math.exp(-(end - t) / settings.fatigue_seconds)
        if c['kind'] == 'hold':
            own += .11 * settings.fatigue_seconds * (1 - math.exp(-(end - t) / settings.fatigue_seconds))
        after = np.minimum(1., np.minimum(1., future) + min(1., own))
        future_debt = debt * math.exp(-(end - t) / settings.recovery_seconds)
        own_debt = burst * math.exp(-(end - t) / settings.recovery_seconds)
        after_b = np.maximum(0., np.maximum(0., 1 - future_debt) - np.minimum(1., own_debt))
        values = totals[:, None] + .8 * f + 1.5 * np.maximum(0., after - f) + 2 * burst + .2 * (1 - after_b) + .6 * hf + .3 * (1 - hb)
        for j, matrix in neighbors[i]:
            if j < i:
                forbidden = matrix[:, histories[:, j]].T >= 1e8
                values[forbidden] = np.inf
        if allowed is not None:
            values[:, ~np.asarray(allowed[i], dtype=bool)] = np.inf
        ranked = np.argsort(values, axis=None)
        ranked = ranked[np.isfinite(values.ravel()[ranked])][:width]
        if not len(ranked):
            raise ValueError(f"No non-overlapping load assignment at {t}: {c['note_ids']}")
        parents, labels = np.divmod(ranked, k)
        totals = values[parents, labels]
        histories = histories[parents].copy(); histories[:, i] = labels
        fatigue = fatigue[parents].copy(); debt = debt[parents].copy()
        last_start = last_start[parents].copy()
        rows = np.arange(len(parents))
        fatigue[rows, labels] += effort
        debt[rows, labels] += burst[parents, labels]
        last_start[rows, labels] = t
        if c['kind'] == 'hold': active.append(i)
        previous_time = t
        if i % 200 == 0:
            print(f'ASSIGN_LOAD {i}/{n} states={len(histories)} cost={totals.min():.6f}', flush=True)
    labels = histories[int(np.argmin(totals))].copy()
    exact, _ = score_labels(contacts, keys, options, labels)
    if not math.isclose(exact, float(totals.min()), rel_tol=1e-10, abs_tol=1e-8):
        raise AssertionError(f'Load accumulator mismatch: {exact} != {totals.min()}')
    return labels
