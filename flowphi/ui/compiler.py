from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

from flowphi.ui.model import check_points, write_json_atomic
from flowphi.ui.validation import movement_warnings, quick_check
from flowphi.ui.contact_state import clip_contact, resolve_contacts
from flowphi.motion import FINGER_ORDER, attach_plan, finish_motion


def compile_plan(document, chart, *, time_range=None):
    from flowphi.planning.touch import ScreenUtil, TouchAction, VirtualTouchEvent, dump_data
    from flowphi.rendering.pipeline import contacts_from_psap
    conflicts = quick_check(document, chart=chart, time_range=time_range)
    if conflicts:
        raise ValueError('请先解决手指占用或触控接续问题，再导出动作数据')
    plan = copy.deepcopy(document.baseline)
    plan['pose'] = copy.deepcopy(document.pose)
    contacts = []
    resolved, _, _ = resolve_contacts(document, chart)
    for original in resolved:
        c = clip_contact(original, time_range)
        if c is None:
            continue
        i = c['editor_contacts'][0]
        c['pointer'] = (0 if c['hand'] == 'left' else 5) + FINGER_ORDER.index(c['finger'])
        c['start'], c['end'] = round(c['start'] * 1000) / 1000, round(c['end'] * 1000) / 1000
        points = []
        for t, x, y in c['points']:
            point = [round(t * 1000) / 1000, x, y]
            if points and points[-1][0] == point[0]:
                if any(abs(a - b) > 1e-12 for a, b in zip(points[-1][1:], point[1:])):
                    raise ValueError(f'动作 {i + 1} 的两个控制点取整后落在同一毫秒，请拉开时间')
                continue
            points.append(point)
        check_points(c, points)
        c['points'] = points
        c['planned'] = True
        c['editor_contact'] = i
        if c['assignment_locked']:
            c['fixed_finger'] = [c['hand'], c['finger']]
            c['manual_hand'] = c['hand']
        else:
            c.pop('fixed_finger', None)
            c.pop('manual_hand', None)
        contacts.append(c)
    contacts.sort(key=lambda c: (c['start'], c['pointer']))
    plan['contacts'] = contacts
    plan['hand_rest'] = finish_motion(contacts, plan['profile'], plan['physical_screen'], .28 / .82)
    events = defaultdict(list)
    for c in contacts:
        for j, (t, x, y) in enumerate(c['points']):
            events[round(t * 1000)].append(VirtualTouchEvent(complex(x * chart.width, y * chart.height),
                TouchAction.DOWN if j == 0 else TouchAction.MOVE, c['pointer']))
        x, y = c['points'][-1][1:]
        events[round(c['end'] * 1000)].append(VirtualTouchEvent(complex(x * chart.width, y * chart.height), TouchAction.UP, c['pointer']))
    raw = dump_data(ScreenUtil(chart.width, chart.height), [(ms, sorted(row, key=lambda e: (e.action != TouchAction.UP, e.pointer_id)))
                                                          for ms, row in sorted(events.items())])
    _, decoded = contacts_from_psap(raw)
    if len(decoded) != len(contacts):
        raise ValueError('导出后的动作数量与方案不一致')
    for c, source in zip(contacts, decoded):
        # Use the exact decoded floating-point coordinates required by attach_plan.
        c.update(source)
    plan['psap_sha256'] = hashlib.sha256(raw).hexdigest()
    plan['authoring'] = dict(editor_version=1, chart_sha256=document.chart.get('sha256'),
                             modified_contacts=sorted(document.edits), frozen=True, time_range=time_range)
    if time_range is not None:
        plan['diagnostics'] = dict(plan.get('diagnostics', {}))
        plan['diagnostics']['degraded'] = [d for d in plan['diagnostics'].get('degraded', [])
                                         if time_range[0] <= d.get('time', -float('inf')) < time_range[1]]
    plan.setdefault('diagnostics', {})['finger_usage'] = dict(Counter(f"{c['hand']}:{c['finger']}" for c in contacts))
    plan['diagnostics']['source_estimates_before_edit'] = bool(document.edits)
    plan['diagnostics'].setdefault('degraded', [])
    if document.edits:
        plan['algorithm_diagnostics'] = copy.deepcopy(document.baseline.get('diagnostics', {}))
        plan['diagnostics']['degraded'] = [dict(
            note_ids=sorted({n for i in warning['contacts'] for n in document.contact(i)['note_ids']}),
            time=warning['time'], reasons=[warning['code']]) for warning in movement_warnings(document)
            if time_range is None or time_range[0] <= warning['time'] < time_range[1]]
        plan['diagnostics']['estimate_scope'] = 'finger transfers after manual editing'
    attach_plan(raw, decoded, plan)
    return raw, plan


def export_plan(document, source, output, report, *, time_range=None):
    if not report.get('valid') or not report.get('geometry_checked'):
        raise ValueError('导出前需要通过完整的手指占用与判定区域检查')
    if report.get('time_range') != time_range:
        raise ValueError('检查范围与导出范围不一致，请重新检查')
    raw, plan = compile_plan(document, source.chart, time_range=time_range)
    if report.get('validated_psap_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('方案已修改，请重新检查后再导出')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Callers choose a fresh revision directory, so cancelled exports cannot
    # replace a previously verified render input.
    if any((output / name).exists() for name in ('plan.psap', 'motion-plan.json')):
        raise ValueError('目标目录已有导出的动作数据，请选择新的目录')
    (output / 'plan.psap').write_bytes(raw)
    write_json_atomic(output / 'motion-plan.json', plan)
    write_json_atomic(output / 'validation.json', report)
    write_json_atomic(output / 'project.fingering.json', document.snapshot(destination=output / 'project.fingering.json'))
    manifest = dict(format='flowphi-authored-render', version=1, chart=source.reference,
                    psap_sha256=hashlib.sha256(raw).hexdigest(),
                    motion_sha256=hashlib.sha256((output / 'motion-plan.json').read_bytes()).hexdigest(),
                    geometry_checked=True, time_range=time_range)
    write_json_atomic(output / 'render-inputs.json', manifest)
    return manifest
