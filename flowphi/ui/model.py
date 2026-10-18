from __future__ import annotations

from bisect import bisect_right
import copy
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Iterable

from flowphi.motion import FINGER_ORDER
from flowphi.pose import normalize_pose

FORMAT = 'flowphi-project'
LEGACY_FORMAT = 'phisap-fingering-project'
FORMAT_VERSION = 1
MAX_PROJECT_BYTES = 128 * 1024 * 1024
MAX_CONTACTS = 100_000
MAX_POINTS = 2_000_000
HANDS = ('left', 'right')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class TimeIndex:
    def __init__(self, intervals: Iterable[tuple[float, float, int]]):
        self.rows = sorted(intervals, key=lambda r: (r[0], r[2]))
        self.starts = [r[0] for r in self.rows]
        self.size = 1 << max(0, (len(self.rows) - 1).bit_length())
        self.maximum = [-math.inf] * (self.size * 2)
        for i, (_, end, _) in enumerate(self.rows):
            self.maximum[self.size + i] = end
        for i in range(self.size - 1, 0, -1):
            self.maximum[i] = max(self.maximum[i * 2], self.maximum[i * 2 + 1])
        self.visited = 0

    def query(self, start: float, end: float) -> list[int]:
        if end < start:
            start, end = end, start
        stop = bisect_right(self.starts, end)
        result = []
        self.visited = 0
        stack = [(1, 0, self.size)]
        while stack:
            node, low, high = stack.pop()
            self.visited += 1
            if low >= stop or self.maximum[node] <= start:
                continue
            if high - low == 1:
                if low < len(self.rows):
                    result.append(self.rows[low][2])
                continue
            middle = (low + high) // 2
            stack.append((node * 2 + 1, middle, high))
            stack.append((node * 2, low, middle))
        return result


def check_points(contact, points):
    if not isinstance(points, list) or not points:
        raise ValueError('每段动作的轨迹至少需要一个控制点')
    previous = -math.inf
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 3 or not all(map(finite, point)):
            raise ValueError('轨迹控制点必须是有限的时间、X 和 Y')
        t, x, y = point
        if t <= previous or t < contact['start'] - 1e-9 or t >= contact['end']:
            raise ValueError('控制点必须按时间先后排列，不可重合；时间不能早于按下，也不能达到或晚于抬起')
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError('落点需要位于屏幕内')
        previous = t
    if abs(points[0][0] - contact['start']) > 1e-9:
        raise ValueError('第一个控制点的时间必须与按下时刻一致')


def check_baseline(plan, *, allow_empty=False):
    if not isinstance(plan, dict) or plan.get('version') != 1 or plan.get('algorithm') != 5:
        raise ValueError('请选择 flowphi 工程（.fingering.json）或兼容的 motion-plan.json')
    if plan.get('timebase') != 'chart_seconds' or plan.get('coordinates') != 'normalized_screen':
        raise ValueError('此规划的时间或坐标格式不受支持')
    contacts = plan.get('contacts')
    if not isinstance(contacts, list) or (not contacts and not allow_empty) or len(contacts) > MAX_CONTACTS:
        raise ValueError('工程需要包含 1–100,000 段动作')
    if not isinstance(plan.get('profile'), dict) or not isinstance(plan.get('settings'), dict):
        raise ValueError('规划缺少手型或算法配置')
    if not finite(plan['profile'].get('scale')) or plan['profile']['scale'] <= 0:
        raise ValueError('规划的手型尺度无效')
    fingers = plan['settings'].get('fingers', FINGER_ORDER)
    if not isinstance(fingers, (list, tuple)) or not fingers or any(f not in FINGER_ORDER for f in fingers):
        raise ValueError('规划的可用手指配置无效')
    screen = plan.get('physical_screen', [])
    if len(screen) != 2 or any(not finite(v) or v <= 0 for v in screen):
        raise ValueError('规划的屏幕尺寸无效')
    total = 0
    for c in contacts:
        if not isinstance(c, dict) or c.get('hand') not in HANDS or c.get('finger') not in FINGER_ORDER:
            raise ValueError('规划包含无效的手或手指')
        if not finite(c.get('start')) or not finite(c.get('end')) or c['end'] <= c['start']:
            raise ValueError('方案包含无效的按下或抬起时间')
        if c.get('kind') not in ('tap', 'hold', 'drag', 'flick'):
            raise ValueError('方案包含不受支持的动作类型')
        if c.get('touch_mode', 'auto') not in ('auto', 'press', 'continue'):
            raise ValueError('方案包含无效的触控接续方式')
        if 'down_time' in c and not finite(c['down_time']):
            raise ValueError('方案包含无效的触控按下时间')
        if (not isinstance(c.get('note_ids'), list) or not c['note_ids'] or
                any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in c['note_ids'])):
            raise ValueError('规划缺少有效的音符编号')
        check_points(c, c.get('points'))
        total += len(c['points'])
    if total > MAX_POINTS:
        raise ValueError('轨迹控制点过多，超过本版工程上限')


@dataclass
class Change:
    label: str
    before: dict
    after: dict
    before_state: int
    after_state: int
    size: int
    kind: str = 'contacts'
    created: str = field(default_factory=lambda: datetime.now().strftime('%H:%M:%S'))
    source: str = '你'
    preferences_before: dict = field(default_factory=dict)
    preferences_after: dict = field(default_factory=dict)


class PlanDocument:
    def __init__(self, baseline, *, chart=None, edits=None, pose=None, title='', copy_baseline=True, allow_empty=False):
        check_baseline(baseline, allow_empty=allow_empty)
        self.baseline = copy.deepcopy(baseline) if copy_baseline else baseline
        self.chart = copy.deepcopy(chart or {})
        self.title = title or '未命名指法'
        self.edits = {}
        self.pose = normalize_pose(baseline.get('pose') if pose is None else pose)
        self._cache = {}
        self.undo_stack: list[Change] = []
        self.redo_stack: list[Change] = []
        self.history_bytes = 0
        self.history_limit = 16 * 1024 * 1024
        self.revision = 0
        self.motion_revision = self.pose_revision = 0
        self._state = self._saved_state = self._serial = 0
        self._saved_chart = copy.deepcopy(self.chart)
        self._saved_title = self.title
        self.path: Path | None = None
        self.baseline_index = TimeIndex((c['start'], c['end'], i) for i, c in enumerate(self.baseline['contacts']))
        self.note_contacts = {}
        for i, c in enumerate(self.baseline['contacts']):
            for n in c['note_ids']:
                self.note_contacts.setdefault(n, []).append(i)
        for key, value in (edits or {}).items():
            index = int(key)
            self._validate_edit(index, value)
            if value:
                self.edits[index] = copy.deepcopy(value)
        self._reindex()

    def __len__(self):
        return len(self.baseline['contacts'])

    @property
    def dirty(self):
        return self._state != self._saved_state or self.chart != self._saved_chart or self.title != self._saved_title

    @property
    def allowed_fingers(self):
        return tuple(self.baseline['settings'].get('fingers', FINGER_ORDER))

    def contact(self, index):
        if index not in self._cache:
            source = self.baseline['contacts'][index]
            edit = self.edits.get(index, {})
            result = dict(source)
            result.update({k: v for k, v in edit.items() if k in ('hand', 'finger', 'points', 'start', 'end', 'touch_mode', 'note_ids')})
            if result['start'] != source['start']:
                result['down_time'] = result['start']
            if result['end'] != source['end']:
                # The planner's tail exemption applies only to its original UP.
                result.pop('terminal_hold_release_ms', None)
                result.pop('terminal_hold_release_reason', None)
            result['assignment_locked'] = edit.get('assignment_locked', bool(source.get('fixed_finger')))
            result['path_locked'] = edit.get('path_locked', bool(source.get('path_locked')))
            self._cache[index] = result
        return self._cache[index]

    def sorted_ids(self, indices):
        return sorted(set(indices), key=lambda i: (self.contact(i)['start'], i))

    def _reindex(self):
        self.index = TimeIndex((self.contact(i)['start'], self.contact(i)['end'], i) for i in range(len(self)))
        self.contact_duration = max((self.contact(i)['end'] for i in range(len(self))), default=0.)
        self._pose_duration()
        self.note_contacts = {}
        for i in range(len(self)):
            for nid in self.contact(i)['note_ids']:
                self.note_contacts.setdefault(nid, []).append(i)

    def _pose_duration(self):
        self.duration = max(self.contact_duration, max((c['end'] for c in self.pose['clips']), default=0.))

    def _commit_change(self, label, before, after, kind):
        if before == after:
            return []
        size = len(json.dumps([before, after], separators=(',', ':')))
        self._serial += 1
        change = Change(label, before, after, self._state, self._serial, size, kind)
        change.source = '模型' if label.startswith('模型') else '你'
        changed = self._apply_change(change, forward=True)
        self.undo_stack.append(change)
        self.history_bytes += size
        while len(self.undo_stack) > 1 and (len(self.undo_stack) > 200 or self.history_bytes > self.history_limit):
            self.history_bytes -= self.undo_stack.pop(0).size
        self.redo_stack.clear()
        return changed

    def commit_pose(self, label, pose):
        return self._commit_change(label, copy.deepcopy(self.pose), normalize_pose(pose), 'pose')

    def commit_chart(self, label, chart):
        return self._commit_change(label, copy.deepcopy(self.chart), copy.deepcopy(chart), 'chart')

    def _apply_pose(self, pose, state):
        self.pose = copy.deepcopy(pose)
        self._pose_duration()
        self._state = state
        self.revision += 1
        self.pose_revision += 1

    def _validate_edit(self, index, edit):
        if not isinstance(index, int) or not 0 <= index < len(self):
            raise ValueError('工程包含找不到对应动作的编号')
        if not isinstance(edit, dict) or set(edit) - {'hand', 'finger', 'points', 'start', 'end', 'assignment_locked', 'path_locked', 'touch_mode', 'note_ids'}:
            raise ValueError('工程包含不受支持的修改字段')
        if 'hand' in edit and edit['hand'] not in HANDS:
            raise ValueError('请选择左手或右手')
        if 'finger' in edit and edit['finger'] not in self.allowed_fingers:
            raise ValueError('此方案未启用这根手指')
        if edit.get('touch_mode', 'auto') not in ('auto', 'press', 'continue'):
            raise ValueError('触控接续方式无效')
        if 'note_ids' in edit and (not isinstance(edit['note_ids'], list) or not edit['note_ids'] or
                any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in edit['note_ids'])):
            raise ValueError('关联音符编号无效')
        contact = self.baseline['contacts'][index] | edit
        if not finite(contact['start']) or not finite(contact['end']) or contact['end'] <= contact['start']:
            raise ValueError('抬起时间必须晚于按下时间')
        if any(key in edit for key in ('points', 'start', 'end')):
            check_points(contact, contact['points'])
        for key in ('assignment_locked', 'path_locked'):
            if key in edit and not isinstance(edit[key], bool):
                raise ValueError('锁定状态无效')

    def commit(self, label, updates):
        after = {}
        for index, edit in updates.items():
            self._validate_edit(index, edit)
            if edit != self.edits.get(index, {}):
                after[index] = copy.deepcopy(edit)
        if not after:
            return []
        before = {i: copy.deepcopy(self.edits.get(i, {})) for i in after}
        return self._commit_change(label, before, after, 'contacts')

    def _apply(self, values, state):
        timing_changed = any(
            (self.edits.get(i, {}).get(key, self.baseline['contacts'][i][key]) !=
             edit.get(key, self.baseline['contacts'][i][key]))
            for i, edit in values.items() for key in ('start', 'end', 'note_ids'))
        for index, value in values.items():
            if value:
                self.edits[index] = value
            else:
                self.edits.pop(index, None)
            self._cache.pop(index, None)
        if timing_changed:
            self._reindex()
        self._state = state
        self.revision += 1
        self.motion_revision += 1

    def assign(self, indices, hand, finger):
        return self.commit('指定手指', {i: self.edits.get(i, {}) | dict(hand=hand, finger=finger, assignment_locked=True)
                                         for i in indices})

    def alternate(self, indices, finger='index', first='left'):
        hands = (first, 'right' if first == 'left' else 'left')
        return self.commit('左右交替', {i: self.edits.get(i, {}) | dict(hand=hands[n % 2], finger=finger, assignment_locked=True)
                                       for n, i in enumerate(self.sorted_ids(indices))})

    def swap_hands(self, indices):
        return self.commit('互换左右手', {i: self.edits.get(i, {}) | dict(
            hand='right' if self.contact(i)['hand'] == 'left' else 'left', assignment_locked=True) for i in indices})

    def set_lock(self, indices, locked, *, path=False):
        key = 'path_locked' if path else 'assignment_locked'
        return self.commit('锁定' if locked else '解锁', {i: self.edits.get(i, {}) | {key: bool(locked)} for i in indices})

    def set_points(self, index, points):
        return self.commit('修改轨迹', {index: self.edits.get(index, {}) | dict(points=points, path_locked=True)})

    def edit_paths(self, indices, *, operation, rows=None, only_hold=False, **options):
        from flowphi.ui.path_edit import edit_points
        ids = self.sorted_ids(indices)
        if rows is not None and len(ids) != 1:
            raise ValueError('仅修改所选控制点时，请只选择一段动作')
        labels = dict(translate='批量平移轨迹', set='批量设置坐标', freeze='固定轨迹落点',
                      straighten='拉直轨迹', simplify='简化轨迹', delete='批量移除控制点')
        updates = {}
        for i in ids:
            c = self.contact(i)
            if only_hold and c['kind'] != 'hold':
                continue
            points = edit_points(c['points'], operation=operation, rows=rows,
                                 screen=self.baseline['physical_screen'], **options)
            if points != c['points']:
                updates[i] = self.edits.get(i, {}) | dict(points=points, path_locked=True)
        # All trajectories are validated before a single undoable edit is applied.
        return self.commit(labels.get(operation, '批量修改轨迹'), updates)

    def set_release(self, indices, *, mode, value, reshape='trim'):
        from flowphi.ui.gestures import release_points
        if mode not in ('absolute', 'offset', 'duration') or not finite(value):
            raise ValueError('抬起时间设置无效')
        ids = self.sorted_ids(indices)
        if mode == 'absolute' and len(ids) != 1:
            raise ValueError('指定音乐播放到第几秒时抬起，需要只选择一段动作')
        updates = {}
        for i in ids:
            c = self.contact(i)
            end = round((value if mode == 'absolute' else c['end'] + value if mode == 'offset'
                         else c['start'] + value) * 1000) / 1000
            if end == c['end']:
                continue
            points = release_points(c, end, reshape)
            updates[i] = self.edits.get(i, {}) | dict(end=end, points=points, path_locked=True)
        return self.commit('修改抬起时间', updates)

    def set_touch_mode(self, indices, mode):
        return self.commit('修改触控接续', {i: self.edits.get(i, {}) | dict(touch_mode=mode) for i in indices})

    def split_heads(self, indices, chart):
        from flowphi.ui.contact_state import chart_notes, heads
        from flowphi.motion import point_at
        notes = chart_notes(chart)
        baseline = dict(self.baseline, contacts=list(self.baseline['contacts']))
        edits = copy.deepcopy(self.edits)
        current = [copy.deepcopy(self.contact(i)) for i in range(len(self))]
        changed = []
        for i in self.sorted_ids(indices):
            c = current[i]
            required = sorted(heads(c, notes), key=lambda n: (notes[n].seconds, n))
            if len(required) < 2:
                continue
            retained = [n for n in c['note_ids'] if n not in required[1:]]
            baseline['contacts'][i] = dict(baseline['contacts'][i], note_ids=retained)
            edits[i] = edits.get(i, {}) | dict(note_ids=retained)
            current[i] = dict(c, note_ids=retained)
            changed.append(i)
            for nid in required[1:]:
                note = notes[nid]
                beat = c.get('judgement_times', {}).get(str(nid), note.seconds)
                start = round(beat * 1000) / 1000
                end = max(start + .001, min(c['end'], round((note.seconds + max(note.hold, .024) + .001) * 1000) / 1000))
                segment = copy.deepcopy(c)
                segment.update(start=start, end=end, beat=beat, down_time=start,
                    points=[[start, *point_at(c['points'], start)], *[list(p) for p in c['points'] if start < p[0] < end]],
                    note_ids=[nid], kind=note.type.name.lower(), touch_mode='press',
                    assignment_locked=False, path_locked=True)
                for name in ('fixed_finger', 'manual_hand', 'shared_contact', 'joint_note_coverage',
                             'joint_note_windows', 'collective_hold', 'editor_contact', 'editor_contacts',
                             'judgement_context', 'terminal_hold_release_ms', 'terminal_hold_release_reason'):
                    segment.pop(name, None)
                choices = [(hand, finger) for hand in HANDS for finger in self.allowed_fingers]
                def score(key):
                    overlaps = sum(max(0., min(end, other['end']) - max(start, other['start']))
                        for other in current if (other['hand'], other['finger']) == key)
                    head_occupied = any(other['start'] < start + .012 and other['end'] > start
                        for other in current if (other['hand'], other['finger']) == key)
                    return overlaps > 1e-9, head_occupied, overlaps, key[0] != c['hand'], abs(FINGER_ORDER.index(key[1]) - FINGER_ORDER.index(c['finger']))
                hand, finger = min(choices, key=score)
                segment.update(hand=hand, finger=finger, pointer=(0 if hand == 'left' else 5) + FINGER_ORDER.index(finger))
                check_points(segment, segment['points'])
                changed.append(len(current))
                current.append(segment)
                baseline['contacts'].append(segment)
        if not changed:
            return []
        check_baseline(baseline)
        before = dict(baseline=self.baseline, edits=copy.deepcopy(self.edits), pose=self.pose)
        after = dict(baseline=baseline, edits=edits, pose=self.pose)
        self._commit_change('拆开合并的音符头', before, after, 'structure')
        return changed

    def replace_plan(self, document, label='重新生成指法初稿'):
        if (self.chart.get('sha256') and self.chart['sha256'] != document.chart.get('sha256') or
                set(self.note_contacts) != set(document.note_contacts)):
            raise ValueError('新初稿与当前工程的谱面不匹配')
        before = dict(baseline=self.baseline, edits=copy.deepcopy(self.edits), pose=self.pose)
        after = dict(baseline=document.baseline, edits=copy.deepcopy(document.edits), pose=document.pose)
        return self._commit_change(label, before, after, 'structure')

    def _apply_structure(self, values, state):
        self.baseline = values['baseline']
        self.edits = copy.deepcopy(values['edits'])
        self.pose = copy.deepcopy(values['pose'])
        self._cache.clear()
        self.baseline_index = TimeIndex((c['start'], c['end'], i) for i, c in enumerate(self.baseline['contacts']))
        self._reindex()
        self._state = state
        self.revision += 1
        self.motion_revision += 1
        self.pose_revision += 1

    def set_flicks(self, indices, chart, **options):
        from flowphi.ui.gestures import flick_contacts, flick_path
        updates = {}
        for i, c, flicks in flick_contacts(self, chart, indices):
            change = flick_path(c, flicks, chart, **options)
            if change is not None:
                updates[i] = self.edits.get(i, {}) | change | dict(path_locked=True)
        return self.commit('修改 Flick 滑动设置', updates)

    def reset(self, indices):
        # Explicitly unlock legacy fixed_finger annotations as well.
        return self.commit('恢复所选初稿', {i: dict(assignment_locked=False, path_locked=False) for i in indices})

    def accept_assignments(self, assignments):
        updates = {}
        for i, hand, finger in assignments:
            if self.contact(i)['assignment_locked']:
                if (hand, finger) != (self.contact(i)['hand'], self.contact(i)['finger']):
                    raise ValueError('重新分配试图修改已锁定的指法')
                continue
            updates[i] = self.edits.get(i, {}) | dict(hand=hand, finger=finger)
        return self.commit('局部重新分配', updates)

    def _apply_change(self, change, *, forward):
        assert self._state == (change.before_state if forward else change.after_state), 'History state mismatch'
        values = change.after if forward else change.before
        state = change.after_state if forward else change.before_state
        if change.kind == 'pose':
            self._apply_pose(values, state)
            return None
        if change.kind == 'chart':
            self.chart = copy.deepcopy(values)
            self._state = state
            self.revision += 1
            return None
        if change.kind == 'preferences':
            self._state = state
            self.revision += 1
            return None
        if change.kind == 'structure':
            self._apply_structure(values, state)
            return list(range(len(self)))
        self._apply(values, state)
        return list(values)

    def undo(self):
        if not self.undo_stack:
            return []
        change = self.undo_stack[-1]
        changed = self._apply_change(change, forward=False)
        self.undo_stack.pop()
        self.history_bytes -= change.size
        self.redo_stack.append(change)
        return changed

    def redo(self):
        if not self.redo_stack:
            return []
        change = self.redo_stack[-1]
        changed = self._apply_change(change, forward=True)
        self.redo_stack.pop()
        self.undo_stack.append(change)
        self.history_bytes += change.size
        return changed

    def snapshot(self, *, destination=None):
        chart = copy.deepcopy(self.chart)
        if destination:
            for key in ('path', 'audio', 'background'):
                if chart.get(key):
                    try:
                        chart[key] = os.path.relpath(chart[key], Path(destination).resolve().parent)
                    except ValueError:
                        pass
        # The baseline never changes; copying only overrides keeps save preparation
        # cheap even for a dense song. Serialize the snapshot in a worker thread.
        return dict(format=FORMAT, version=FORMAT_VERSION, title=self.title,
                    chart=chart, baseline=self.baseline, edits=copy.deepcopy(self.edits), pose=copy.deepcopy(self.pose))

    def mark_saved(self, path=None, *, state=None, chart=None, title=None):
        self._saved_state = self._state if state is None else state
        self._saved_chart = copy.deepcopy(self.chart if chart is None else chart)
        self._saved_title = self.title if title is None else title
        if path:
            self.path = Path(path).resolve()

    @classmethod
    def from_dict(cls, data, *, path=None):
        if data.get('format') not in (FORMAT, LEGACY_FORMAT):
            return cls(data, copy_baseline=False)
        if data.get('version') != FORMAT_VERSION:
            raise ValueError('不支持此指法工程版本')
        chart = copy.deepcopy(data.get('chart', {}))
        if path:
            for key in ('path', 'audio', 'background'):
                if chart.get(key) and not Path(chart[key]).is_absolute():
                    chart[key] = str((Path(path).resolve().parent / chart[key]).resolve())
        doc = cls(data['baseline'], chart=chart, edits=data.get('edits'), pose=data.get('pose'),
                  title=data.get('title', ''), copy_baseline=False)
        doc.path = Path(path).resolve() if path else None
        return doc


def read_document(path):
    path = Path(path)
    if path.stat().st_size > MAX_PROJECT_BYTES:
        raise ValueError('工程文件超过 128 MiB')
    return PlanDocument.from_dict(json.loads(path.read_text(encoding='utf-8-sig')), path=path)


def write_json_atomic(path, data):
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(data, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
