from __future__ import annotations

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from flowphi.ui.chat_view import ChatView


FIELDS = {'hand': '手', 'finger': '手指', 'start': '按下时间', 'end': '松开时间', 'points': '轨迹',
          'assignment_locked': '指法锁定', 'path_locked': '轨迹锁定', 'touch_mode': '触点接法',
          'pose': '姿态', 'settings': '设置', 'left': '左手', 'right': '右手', 'index': '食指',
          'middle': '中指', 'ring': '无名指', 'little': '小指', 'thumb': '拇指',
          'auto': '自动接续', 'continue': '继续按住', 'press': '重新按下'}


def contact_changes(change):
    if change.kind == 'contacts':
        return [(i, change.before.get(i, {}), change.after.get(i, {}))
                for i in sorted(set(change.before) | set(change.after))]
    if change.kind != 'structure':
        return []
    before, after = change.before, change.after
    rows = []
    for i in range(max(len(before['baseline']['contacts']), len(after['baseline']['contacts']))):
        def contact(state):
            contacts = state['baseline']['contacts']
            return (contacts[i] | state['edits'].get(i, {})) if i < len(contacts) else {}
        a, b = contact(before), contact(after)
        if a != b:
            rows.append((i, a, b))
    return rows


class EditHistory(QWidget):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.signature = None
        self.changes = []
        box = QVBoxLayout(self)
        note = QLabel('本次打开工程后的修改。选中记录查看详情；可撤销、重做，或选中相关动作继续编辑。长期备份见“版本”。')
        note.setWordWrap(True)
        box.addWidget(note)
        splitter = QSplitter(Qt.Vertical)
        self.list = QTreeWidget()
        self.list.setHeaderLabels(['状态', '时间 / 来源', '修改'])
        self.list.setRootIsDecorated(False)
        self.list.currentItemChanged.connect(self.show_change)
        splitter.addWidget(self.list)
        self.detail = ChatView()
        self.detail.seek.connect(editor.seek)
        splitter.addWidget(self.detail)
        box.addWidget(splitter, 1)
        row = QHBoxLayout()
        for title, callback in [('撤销一步', editor.undo), ('重做一步', editor.redo), ('选中动作继续编辑', self.select_contacts)]:
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        box.addLayout(row)

    def refresh(self):
        doc = self.editor.document
        signature = (id(doc), doc.revision if doc is not None else None)
        if signature == self.signature:
            return
        self.signature = signature
        current = self.list.currentItem()
        selected = current.data(0, Qt.UserRole) if current else None
        self.list.clear()
        self.changes = [*doc.undo_stack, *reversed(doc.redo_stack)] if doc is not None else []
        applied_states = {c.after_state for c in doc.undo_stack} if doc is not None else set()
        for change in reversed(self.changes):
            applied = change.after_state in applied_states
            item = QTreeWidgetItem(['已应用' if applied else '已撤销', change.created + ' · ' + change.source, change.label])
            item.setData(0, Qt.UserRole, change.after_state)
            self.list.addTopLevelItem(item)
            if selected == change.after_state:
                self.list.setCurrentItem(item)
        if self.list.currentItem() is None and self.list.topLevelItemCount():
            self.list.setCurrentItem(self.list.topLevelItem(0))
        if not self.changes:
            self.detail.reset()

    def current_change(self):
        item = self.list.currentItem()
        return next((c for c in self.changes if item and c.after_state == item.data(0, Qt.UserRole)), None)

    def show_change(self, *_):
        change = self.current_change()
        self.detail.reset()
        if change is None:
            return
        rows = contact_changes(change)
        lines = [f'**{change.label}** · {change.source} · {change.created}', f'涉及 {len(rows)} 段动作']
        def value(key, v):
            if key == 'points':
                return f'{len(v)} 个控制点' if isinstance(v, list) else '初稿轨迹'
            if key in ('start', 'end') and isinstance(v, (int, float)):
                return f'{v + self.editor.offset:.3f} 秒'
            if v is None:
                return '沿用初稿'
            if isinstance(v, bool):
                return '是' if v else '否'
            return FIELDS.get(str(v), str(v))
        for i, before, after in rows:
            fields = []
            for key in FIELDS:
                if before.get(key) != after.get(key):
                    fields.append(f'{FIELDS[key]}：{value(key, before.get(key))} → {value(key, after.get(key))}')
            lines.append(f'- 动作 #{i + 1}：' + '；'.join(fields))
        if change.preferences_after:
            lines.append('- 编辑器设置已修改（完整参数见下方详情）')
        if change.kind == 'pose' or change.kind == 'structure' and change.before['pose'] != change.after['pose']:
            lines.append('- 手部姿态已修改')
        self.detail.add('assistant', '\n\n'.join(lines))
        changes = dict(contacts=[dict(id=i + 1, before=a, after=b) for i, a, b in rows])
        if change.kind == 'pose':
            changes['pose'] = dict(before=change.before, after=change.after)
        elif change.kind == 'structure' and change.before['pose'] != change.after['pose']:
            changes['pose'] = dict(before=change.before['pose'], after=change.after['pose'])
        elif change.kind == 'chart':
            changes['media'] = dict(before=change.before, after=change.after)
        self.detail.add('tool', '完整修改内容', detail='```json\n' + json.dumps(dict(changes,
            settings_before=change.preferences_before, settings_after=change.preferences_after),
            ensure_ascii=False, indent=2) + '\n```')

    def select_contacts(self):
        change = self.current_change()
        if change is None or self.editor.document is None:
            return
        ids = {i for i, _, _ in contact_changes(change) if i < len(self.editor.document)}
        if ids:
            self.editor._set_selection(ids)
            self.editor.seek(min(self.editor.document.contact(i)['start'] for i in ids) + self.editor.offset)
