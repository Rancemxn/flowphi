from __future__ import annotations

import ast
import copy
import json
import math
from types import SimpleNamespace

import numpy as np


NUMPY = 'array asarray arange linspace zeros ones full concatenate stack column_stack diff gradient interp clip minimum maximum abs sqrt sin cos tan arcsin arccos arctan arctan2 exp log power sum mean min max std median percentile quantile sort argsort where isfinite all any round dot cross'.split()
METHODS = set('append extend insert pop remove clear copy get items keys values update setdefault count index sort reverse add discard union intersection difference tolist reshape flatten ravel astype sum mean min max std round clip join split strip replace startswith endswith lower upper'.split())


def calculate(code, source):
    if not isinstance(code, str) or not code.strip() or len(code) > 100_000:
        raise ValueError('Python 代码应为 1–100,000 字符')
    tree = ast.parse(code)
    # Expose only numeric/data operations; never give generated code an IO or reflection capability.
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.With, ast.AsyncWith, ast.AsyncFunctionDef,
                 ast.Global, ast.Nonlocal, ast.Delete)
    for node in ast.walk(tree):
        if isinstance(node, forbidden) or isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError('Python 仅支持数据计算，不能导入、反射或访问外部资源')
        if isinstance(node, ast.Attribute):
            namespace = node.value.id if isinstance(node.value, ast.Name) else ''
            allowed = (set(NUMPY) | {'pi'} if namespace == 'np' else
                       {name for name in dir(math) if not name.startswith('_')} if namespace == 'math' else METHODS)
            if node.attr not in allowed or not isinstance(node.ctx, ast.Load):
                raise ValueError('不支持的 Python 属性：' + node.attr)
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.name.startswith('_')):
            raise ValueError('不支持装饰器或私有函数')
    data = copy.deepcopy(source)
    output = []
    def emit(*values, sep=' ', end='\n'):
        text = sep.join(map(str, values)) + end
        if sum(map(len, output)) + len(text) > 16_000:
            raise ValueError('Python 输出过长，请只打印摘要')
        output.append(text)
    builtins = dict(zip('abs all any bool dict enumerate float int len list map max min range reversed round set sorted str sum tuple zip'.split(),
                        (abs, all, any, bool, dict, enumerate, float, int, len, list, map, max, min, range, reversed, round, set, sorted, str, sum, tuple, zip)))
    builtins['isinstance'] = isinstance
    numeric = SimpleNamespace(**{k: getattr(np, k) for k in NUMPY}, pi=np.pi)
    env = dict(__builtins__=builtins, data=data, operations=[], result=None, print=emit,
               math=SimpleNamespace(**{k: getattr(math, k) for k in dir(math) if not k.startswith('_')}), np=numeric)
    exec(compile(tree, '<模型计算>', 'exec'), env)
    data, operations = env['data'], env['operations']
    if not isinstance(data, dict) or set(data) != set(source) or not isinstance(operations, list):
        raise ValueError('保留 data 结构，operations 必须是数组')
    rows = data['contacts']
    originals = {c['id']: c for c in source['contacts']}
    if not isinstance(rows, list) or len(rows) != len(originals) or {c['id'] for c in rows} != set(originals):
        raise ValueError('Python 不能增删动作或改变动作编号；请修改字段')
    editable = {'hand', 'finger', 'start', 'end', 'points', 'assignment_locked', 'path_locked', 'touch_mode'}
    for row in rows:
        original = originals[row['id']]
        if set(row) != set(original) or any(row[k] != original[k] for k in original.keys() - editable):
            raise ValueError('音符关联、动作编号和类型只读')
        changed = {k: row[k] for k in editable if k in row and row[k] != original[k]}
        if changed:
            operations.insert(0, dict(op='contact', scope=dict(ids=[row['id']]), values=changed))
    if data['pose'] != source['pose'] or data['notes'] != source['notes']:
        raise ValueError('姿态请通过 operations 中的 pose_defaults/pose_clip 修改；notes 只读')
    if not isinstance(data['settings'], dict) or set(data['settings']) != set(source['settings']):
        raise ValueError('请保留设置字段，只修改字段值')
    settings = {k: v for k, v in data['settings'].items() if v != source['settings'].get(k)}
    result = dict(settings=settings, operations=operations, result=env['result'], stdout=''.join(output))
    if len(json.dumps(dict(result=result['result'], stdout=result['stdout']), ensure_ascii=False, allow_nan=False)) > 16_000:
        raise ValueError('计算结果过长，请只返回摘要')
    json.dumps(result, allow_nan=False)
    return result
