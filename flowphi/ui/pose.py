from __future__ import annotations

import copy
import math
import uuid

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
    QFormLayout, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from flowphi.pose import (HANDS, TARGETS, PoseSampler, channels, empty_pose, mirrored_values,
                         neutral_values, normalize_pose)
from flowphi.motion import stroke_shape
from flowphi.ui.pose_preview import PosePreview

NAMES = dict(hand='手腕与整手', index='食指', middle='中指', ring='无名指', little='小指', thumb='拇指')
HAND_NAMES = dict(left='左手', right='右手')
COLORS = dict(left='#70b9fa', right='#ffa58c')
FIELDS = {
    'x': ('左右偏移', 1000., ' mm', 1.), 'y': ('前后偏移', 1000., ' mm', 1.),
    'z': ('高度偏移', 1000., ' mm', 1.), 'yaw': ('平面转向', 180 / math.pi, '°', 1.),
    'pitch': ('手掌俯仰', 180 / math.pi, '°', 1.), 'roll': ('手掌侧倾', 180 / math.pi, '°', 1.),
    'curl': ('整体弯曲', 180 / math.pi, '°', 1.), 'middle_curl': ('中间关节弯曲', 180 / math.pi, '°', 1.),
    'tip_curl': ('指尖关节弯曲', 180 / math.pi, '°', 1.), 'spread': ('张合', 180 / math.pi, '°', 1.),
    'raise': ('空闲抬指量', 1000., ' mm', 1.), 'lift_scale': ('抬落高度倍率', 1., ' ×', .05),
    'peak_shift': ('最高点时机偏移', 100., ' %', 1.),
}
HAND_FIELD_NAMES = dict(curl='空闲手指弯曲', spread='空闲手指张合')
FIELD_TIPS = {
    'x': '相对自动姿态左右移动手腕正值向屏幕右侧，负值向左侧；超出接触手指可达范围时会修正手腕位置',
    'y': '相对自动姿态前后移动手腕正值向屏幕上方，负值向下方；方向以屏幕为准，不随观察角度变化',
    'z': '相对自动姿态调整手腕高度正值抬高，负值降低；超出接触手指可达范围时会修正手腕位置',
    'yaw': '相对自动姿态，在屏幕平面内转动手腕接触中的指尖继续受触控轨迹约束',
    'pitch': '相对自动姿态调整手掌的前后倾斜角度接触中的指尖继续受触控轨迹约束',
    'roll': '相对自动姿态调整手掌向两侧的倾斜角度接触中的指尖继续受触控轨迹约束',
    'curl': '增加或减少这根空闲手指的整体弯曲接触前后会平滑过渡到触控动作',
    'middle_curl': '单独调整这根空闲手指中间关节的弯曲；与整体弯曲叠加',
    'tip_curl': '单独调整这根空闲手指末端关节的弯曲；与整体弯曲叠加',
    'spread': '调整这根空闲手指向两侧张开或收拢的角度接触前后会平滑过渡到触控动作',
    'raise': '调整空闲时的抬指幅度正值抬高，负值降低；接触前后会平滑过渡到触控动作',
    'lift_scale': '1× 沿用自动抬落幅度，数值越大抬得越高整手与单根手指的倍率相乘；按下和抬起时间由指法决定',
    'peak_shift': '正值让抬起最高点更晚，负值更早；百分比按一次抬落过程计算按下和抬起时间由指法决定',
}


def time_field():
    field = QDoubleSpinBox()
    field.setRange(-3600., 86400.)
    field.setDecimals(3)
    field.setSingleStep(.05)
    field.setSuffix(' s')
    field.setKeyboardTracking(False)
    return field


class StrokeCurve(QWidget):
    drag_changed = Signal(object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = dict(lift_scale=1., peak_shift=0.)
        self._origin = None
        self.setMinimumHeight(122)
        self.setMaximumHeight(150)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip('上下拖动圆点调整抬落高度，左右拖动调整最高点时机修改范围与右侧参数一致；Esc 取消本次拖动')

    def set_values(self, values):
        self._origin = None
        self.values = {name: values[name] for name in ('lift_scale', 'peak_shift')}
        self.update()

    def plot_rect(self):
        return QRectF(22, 28, max(1, self.width() - 44), max(1, self.height() - 53))

    def peak(self):
        rect = self.plot_rect()
        phase = max(.12, min(.9, .62 + self.values['peak_shift']))
        return QPointF(rect.left() + phase * rect.width(), rect.bottom() - self.values['lift_scale'] / 3 * rect.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#142234'))
        rect = self.plot_rect()
        painter.setPen(QColor('#aebed3'))
        painter.drawText(QPointF(8, 16), '抬落曲线 · 拖动圆点调整')
        painter.setPen(QPen(QColor('#30475e'), 1))
        painter.drawRect(rect)
        for level in (1, 2):
            y = rect.bottom() - level / 3 * rect.height()
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        path = QPainterPath()
        job = dict(palm_lift_mode='gentle', stroke_peak_phase=max(.12, min(.9, .62 + self.values['peak_shift'])))
        for i in range(101):
            u = i / 100
            point = QPointF(rect.left() + u * rect.width(),
                rect.bottom() - stroke_shape(u, job) * self.values['lift_scale'] / 3 * rect.height())
            path.moveTo(point) if i == 0 else path.lineTo(point)
        painter.setPen(QPen(QColor('#80bbef'), 2))
        painter.drawPath(path)
        painter.setBrush(QColor('#e7c777'))
        painter.drawEllipse(self.peak(), 5., 5.)
        painter.setPen(QColor('#9eb2ca'))
        painter.drawText(QPointF(rect.left(), self.height() - 7), '抬起')
        painter.drawText(QRectF(rect.left(), self.height() - 22, rect.width(), 19), Qt.AlignRight, '下次按下')

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and (event.position() - self.peak()).manhattanLength() < 22:
            self.setFocus()
            self._origin = dict(self.values)
            self._press = event.position()

    def _move(self, point, finished):
        rect = self.plot_rect()
        self.values = dict(peak_shift=max(-.4, min(.3, (point.x() - rect.left()) / rect.width() - .62)),
                           lift_scale=max(0., min(3., (rect.bottom() - point.y()) / rect.height() * 3)))
        self.drag_changed.emit({name: value - self._origin[name] for name, value in self.values.items()}, finished)
        self.update()

    def mouseMoveEvent(self, event):
        if self._origin is not None:
            self._move(event.position(), False)
        else:
            self.setCursor(Qt.SizeAllCursor if (event.position() - self.peak()).manhattanLength() < 22 else Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if self._origin is not None:
            if (event.position() - self._press).manhattanLength() >= 1:
                self._move(event.position(), True)
            else:
                self.values = self._origin
                self.drag_changed.emit(None, False)
                self.update()
            self._origin = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._origin is not None:
            self.values, self._origin = self._origin, None
            self.drag_changed.emit(None, False)
            self.update()
            return
        super().keyPressEvent(event)


class PosePanel(QWidget):
    change = Signal(dict, str)
    preview = Signal(object)
    seek = Signal(float)
    target_selected = Signal(str, str)
    clip_selected = Signal(str)
    error = Signal(str)

    def __init__(self, range_provider, parent=None):
        super().__init__(parent)
        self.range_provider = range_provider
        self.document = None
        self.pose = empty_pose()
        self.time = self.offset = 0.
        self.clip_id = None
        self.key_index = 0
        self._refreshing = False
        self._drag_origin = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        target_row = QHBoxLayout()
        self.hand = QComboBox()
        self.target = QComboBox()
        self.hand.setToolTip('选择要调整的左手或右手')
        self.target.setToolTip('选择手腕与整手动作，或单独调整一根手指；空闲手指也可以直接选择')
        for hand in HANDS:
            self.hand.addItem(HAND_NAMES[hand], hand)
        for target in TARGETS:
            self.target.addItem(NAMES[target], target)
        target_row.addWidget(self.hand)
        target_row.addWidget(self.target, 1)
        layout.addLayout(target_row)
        self.hand.currentIndexChanged.connect(self._target_changed)
        self.target.currentIndexChanged.connect(self._target_changed)
        self.scope = QComboBox()
        self.scope.setToolTip('整曲默认值是基础姿态片段只在起止范围内生效，并按淡入淡出过渡；同一目标的片段重叠时，后创建的优先未单独设置的参数沿用基础或较早片段的姿态')
        self.scope.currentIndexChanged.connect(self._scope_changed)
        layout.addWidget(self.scope)
        buttons = QGridLayout()
        self.new_button = QPushButton('新建姿态片段')
        self.delete_button = QPushButton('删除片段')
        self.mirror_button = QPushButton('镜像到另一手')
        self.reset_button = QPushButton('清除整曲设置')
        for i, button in enumerate((self.new_button, self.delete_button, self.mirror_button, self.reset_button)):
            buttons.addWidget(button, i // 2, i % 2)
        self.new_button.setToolTip('使用时间轴下方的“片段起止”范围，为所选手腕或手指新建姿态片段')
        self.delete_button.setToolTip('删除所选姿态片段及其中的全部关键帧可用 Ctrl+Z 撤销')
        self.new_button.clicked.connect(self.new_clip)
        self.delete_button.clicked.connect(self.delete_clip)
        self.mirror_button.clicked.connect(self.mirror)
        self.reset_button.clicked.connect(self.reset_values)
        layout.addLayout(buttons)
        self.cursor = QLabel()
        self.cursor.setWordWrap(True)
        layout.addWidget(self.cursor)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(7, 7, 7, 7)
        self.clip_controls = QWidget()
        clip_layout = QVBoxLayout(self.clip_controls)
        clip_layout.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox('启用此片段')
        self.enabled.setToolTip('取消勾选后暂停应用此片段，保留其中的关键帧和设置')
        self.enabled.toggled.connect(lambda value: self._clip_option('enabled', value))
        clip_layout.addWidget(self.enabled)
        form = QFormLayout()
        self.start, self.end, self.fade_in, self.fade_out = (time_field() for _ in range(4))
        for name, field in (('开始（音乐时间）', self.start), ('结束（音乐时间）', self.end),
                            ('淡入时长', self.fade_in), ('淡出时长', self.fade_out)):
            form.addRow(name, field)
        self.start.setToolTip('修改起止时间会按比例移动片段内的关键帧')
        self.end.setToolTip(self.start.toolTip())
        self.fade_in.setToolTip('片段开头逐渐应用姿态的时间；0 s 表示立即生效')
        self.fade_out.setToolTip('片段结束前逐渐减弱姿态的时间；0 s 表示直接结束')
        self.start.valueChanged.connect(self._resize_clip)
        self.end.valueChanged.connect(self._resize_clip)
        self.fade_in.valueChanged.connect(lambda value: self._clip_option('fade_in', value))
        self.fade_out.valueChanged.connect(lambda value: self._clip_option('fade_out', value))
        self.curve = QComboBox()
        self.curve.addItem('平滑过渡', 'smooth')
        self.curve.addItem('匀速过渡', 'linear')
        self.curve.setToolTip('控制关键帧之间的参数变化：平滑过渡会渐快渐慢，匀速过渡按相同速度变化片段的淡入淡出始终平滑')
        self.curve.currentIndexChanged.connect(lambda: self._clip_option('curve', self.curve.currentData()))
        form.addRow('关键帧之间', self.curve)
        clip_layout.addLayout(form)
        self.keys = QTableWidget(0, 2)
        self.keys.setHorizontalHeaderLabels(['音乐时间 / s', '已设参数'])
        self.keys.setToolTip('选择一行编辑该关键帧，并将播放位置移到该帧右列显示本帧单独设置的参数数量')
        self.keys.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.keys.verticalHeader().hide()
        self.keys.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.keys.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.keys.setSelectionMode(QAbstractItemView.SingleSelection)
        self.keys.setMinimumHeight(110)
        self.keys.setMaximumHeight(170)
        self.keys.itemSelectionChanged.connect(self._key_selected)
        clip_layout.addWidget(self.keys)
        key_buttons = QHBoxLayout()
        self.add_key = QPushButton('播放位置加帧')
        self.remove_key = QPushButton('删除关键帧')
        self.add_key.setToolTip('在播放位置添加关键帧；超出片段时取最近的片段边界该时刻已有关键帧时，直接选中它')
        self.remove_key.setToolTip('删除所选关键帧；每个片段至少保留一帧可用 Ctrl+Z 撤销')
        self.add_key.clicked.connect(self.add_keyframe)
        self.remove_key.clicked.connect(self.remove_keyframe)
        key_buttons.addWidget(self.add_key)
        key_buttons.addWidget(self.remove_key)
        clip_layout.addLayout(key_buttons)
        key_form = QFormLayout()
        self.key_time = time_field()
        self.key_time.setToolTip('移动所选关键帧的时间，须位于片段内，并与其他关键帧错开')
        self.key_time.valueChanged.connect(self._move_key)
        key_form.addRow('所选关键帧时间', self.key_time)
        clip_layout.addLayout(key_form)
        body.addWidget(self.clip_controls)
        self.value_title = QLabel()
        self.value_title.setWordWrap(True)
        body.addWidget(self.value_title)
        self.fields = {}
        self.groups = {}
        for target_type in ('hand', 'index'):
            group = QWidget()
            fields = QFormLayout(group)
            fields.setContentsMargins(0, 0, 0, 0)
            self.groups[target_type] = group
            for name, (low, high, _) in channels(target_type).items():
                label, scale, suffix, step = FIELDS[name]
                if target_type == 'hand':
                    label = HAND_FIELD_NAMES.get(name, label)
                field = QDoubleSpinBox()
                field.setDecimals(2 if name == 'lift_scale' else 1)
                field.setRange(low * scale, high * scale)
                field.setSuffix(suffix)
                field.setSingleStep(step)
                field.setKeyboardTracking(False)
                field.valueChanged.connect(lambda value, n=name: self._channel_changed(n, value))
                tip = FIELD_TIPS[name]
                if target_type == 'hand' and name in ('curl', 'spread'):
                    tip = ('统一调整这只手所有空闲手指的' + ('弯曲' if name == 'curl' else '张合') +
                           '，并与各手指的单独设置叠加接触前后会平滑过渡到触控动作')
                field.setToolTip(tip)
                fields.addRow(label, field)
                self.fields[target_type, name] = field
            body.addWidget(group)
        self.stroke_curve = StrokeCurve()
        self.stroke_curve.drag_changed.connect(self.drag)
        body.addWidget(self.stroke_curve)
        body.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        self.feedback = QLabel('先选手腕或手指，再选择整曲默认值或姿态片段')
        self.feedback.setWordWrap(True)
        self.feedback.setStyleSheet('color: #bacbe0; padding: 4px;')
        layout.addWidget(self.feedback)
        hint = QLabel('这里显示简化的手部关节生成片段视频可查看真实手模表面和避碰后的动作')
        hint.setWordWrap(True)
        hint.setStyleSheet('color: #92a6c0;')
        layout.addWidget(hint)
        self._refresh()

    @property
    def selection(self):
        return self.hand.currentData(), self.target.currentData()

    def current_clip(self, pose=None):
        return next((c for c in (self.pose if pose is None else pose)['clips'] if c['id'] == self.clip_id), None)

    def set_document(self, document, offset=0.):
        changed = self.document is not document
        self.document, self.offset = document, offset
        self.pose = copy.deepcopy(document.pose) if document else empty_pose()
        if changed:
            self.clip_id, self.key_index = None, 0
        self._drag_origin = None
        self._refresh()

    def set_time(self, t):
        self.time = t
        clip = self.current_clip()
        suffix = ''
        if clip:
            if not clip.get('enabled', True):
                suffix = ' · 片段已停用'
            elif not clip['start'] <= t < clip['end']:
                suffix = ' · 播放位置在所选片段外'
            elif t < clip['start'] + clip.get('fade_in', 0.):
                suffix = ' · 淡入中'
            elif t > clip['end'] - clip.get('fade_out', 0.):
                suffix = ' · 淡出中'
            else:
                suffix = ' · 所选片段内'
        elif any((c['hand'], c['target']) == self.selection and c.get('enabled', True)
                 and c['start'] <= t < c['end'] for c in self.pose['clips']):
            suffix = ' · 此时有姿态片段生效'
        self.cursor.setText(f'播放位置 {t + self.offset:.3f} s{suffix}')

    def select_target(self, hand, target):
        if (hand, target) == self.selection:
            return
        self._refreshing = True
        self.hand.setCurrentIndex(self.hand.findData(hand))
        self.target.setCurrentIndex(self.target.findData(target))
        self._refreshing = False
        self._target_changed()

    def select_clip(self, identity, key_time=None):
        clip = next((c for c in self.pose['clips'] if c['id'] == identity), None)
        if clip is None:
            return
        self.select_target(clip['hand'], clip['target'])
        self.clip_id = identity
        self.key_index = min(range(len(clip['keys'])), key=lambda i: abs(clip['keys'][i]['time'] -
                            (self.time if key_time is None else key_time)))
        self._refresh()
        self.clip_selected.emit(identity)
        self.seek.emit(clip['keys'][self.key_index]['time'])

    def _target_changed(self, *_):
        if self._refreshing:
            return
        self.clip_id, self.key_index = None, 0
        self._refresh()
        self.target_selected.emit(*self.selection)
        self.clip_selected.emit('')

    def _scope_changed(self, *_):
        if self._refreshing:
            return
        self.clip_id, self.key_index = self.scope.currentData(), 0
        self._refresh()
        self.clip_selected.emit(self.clip_id or '')
        clip = self.current_clip()
        if clip:
            self.seek.emit(clip['keys'][0]['time'])

    def _values(self):
        hand, target = self.selection
        values = neutral_values(target)
        values.update(self.pose['defaults'].get(f'{hand}:{target}', {}))
        clip = self.current_clip()
        if clip:
            values.update(clip['keys'][self.key_index]['values'])
        return values

    def _refresh(self):
        self._refreshing = True
        try:
            hand, target = self.selection
            self.scope.clear()
            self.scope.addItem('整曲默认值', None)
            for clip in self.pose['clips']:
                if (clip['hand'], clip['target']) == (hand, target):
                    self.scope.addItem(f"{clip['start'] + self.offset:.3f}–{clip['end'] + self.offset:.3f} s"
                                       + ('' if clip.get('enabled', True) else '（停用）'), clip['id'])
            index = self.scope.findData(self.clip_id)
            self.scope.setCurrentIndex(max(0, index))
            self.clip_id = self.scope.currentData()
            clip = self.current_clip()
            self.clip_controls.setVisible(clip is not None)
            self.delete_button.setEnabled(clip is not None)
            if clip:
                self.key_index = min(self.key_index, len(clip['keys']) - 1)
                self.enabled.setChecked(clip.get('enabled', True))
                self.start.setRange(-3600 + self.offset, 86400 + self.offset)
                self.end.setRange(-3600 + self.offset, 86400 + self.offset)
                self.start.setValue(clip['start'] + self.offset)
                self.end.setValue(clip['end'] + self.offset)
                for field, name in ((self.fade_in, 'fade_in'), (self.fade_out, 'fade_out')):
                    field.setRange(0., clip['end'] - clip['start'])
                    field.setValue(clip.get(name, 0.))
                self.curve.setCurrentIndex(self.curve.findData(clip.get('curve', 'smooth')))
                self.keys.setRowCount(len(clip['keys']))
                for i, key in enumerate(clip['keys']):
                    self.keys.setItem(i, 0, QTableWidgetItem(f"{key['time'] + self.offset:.3f}"))
                    parameter = QTableWidgetItem(f"{len(key['values'])} 项")
                    names = [HAND_FIELD_NAMES.get(name, FIELDS[name][0]) if target == 'hand' else FIELDS[name][0]
                             for name in key['values']]
                    parameter.setToolTip('、'.join(names) if names else '本帧未单独设置参数，沿用整曲默认值或较早的重叠片段')
                    self.keys.setItem(i, 1, parameter)
                self.keys.selectRow(self.key_index)
                self.key_time.setRange(clip['start'] + self.offset, clip['end'] + self.offset)
                self.key_time.setValue(clip['keys'][self.key_index]['time'] + self.offset)
                self.remove_key.setEnabled(len(clip['keys']) > 1)
            values = self._values()
            target_type = 'hand' if target == 'hand' else 'index'
            for kind, group in self.groups.items():
                group.setVisible(kind == target_type)
            for name, value in values.items():
                self.fields[target_type, name].setValue(value * FIELDS[name][1])
            self.stroke_curve.set_values(values)
            part = f'{HAND_NAMES[hand]} · {NAMES[target]}'
            scope = '整曲默认值' if clip is None else f'关键帧 {self.key_index + 1} · {self.key_time.value():.3f} s'
            self.value_title.setText(f'正在编辑：{part}\n{scope}')
            self.value_title.setToolTip('下方参数和抬落曲线修改这里标明的对象；预览画面显示播放位置的姿态')
            self.stroke_curve.setToolTip(f'修改{part}的{scope}上下拖动圆点调整高度，左右拖动调整最高点时机；Esc 取消本次拖动')
            self.reset_button.setText('清除整曲设置' if clip is None else '清除本帧设置')
            self.reset_button.setToolTip(
                f'清除{part}的整曲默认值，恢复此项的自动值；姿态片段保留可用 Ctrl+Z 撤销' if clip is None else
                '只清除所选关键帧的参数，沿用整曲默认值或较早的重叠片段；其他关键帧保留可用 Ctrl+Z 撤销')
            self.mirror_button.setToolTip(
                '镜像当前部位的整曲默认值，替换另一手对应部位的整曲设置可用 Ctrl+Z 撤销' if clip is None else
                '镜像整个片段及其中的关键帧，在另一手的对应部位新增片段；重叠时新片段优先可用 Ctrl+Z 撤销')
            self.set_time(self.time)
        finally:
            self._refreshing = False

    def _submit(self, pose, label):
        try:
            pose = normalize_pose(pose)
        except ValueError as exc:
            self.error.emit(str(exc))
            self._refresh()
            return
        if pose != self.pose:
            self.pose = pose
            self.change.emit(copy.deepcopy(pose), label)
        self._refresh()
        self.clip_selected.emit(self.clip_id or '')

    def _value_container(self, pose):
        clip = self.current_clip(pose)
        if clip:
            return clip['keys'][self.key_index]['values']
        return pose['defaults'].setdefault(':'.join(self.selection), {})

    def _channel_changed(self, name, value):
        if self._refreshing or self.document is None:
            return
        pose = copy.deepcopy(self.pose)
        low, high, _ = channels(self.selection[1])[name]
        self._value_container(pose)[name] = max(low, min(high, value / FIELDS[name][1]))
        self._submit(pose, '调整' + FIELDS[name][0])

    def _clip_option(self, name, value):
        if self._refreshing or not self.current_clip():
            return
        pose = copy.deepcopy(self.pose)
        self.current_clip(pose)[name] = value
        self._submit(pose, '修改姿态片段')

    def new_clip(self):
        if self.document is None:
            return
        start, end = self.range_provider()
        if end <= start:
            self.error.emit('请先在时间轴下方设置“片段起止”，结束时间须晚于开始时间')
            return
        fade = min(.12, (end - start) / 4)
        t = min(end - fade, max(start + fade, self.time)) if start <= self.time < end else (start + end) / 2
        hand, target = self.selection
        pose = copy.deepcopy(self.pose)
        self.clip_id, self.key_index = uuid.uuid4().hex, 0
        pose['clips'].append(dict(id=self.clip_id, hand=hand, target=target, start=start, end=end,
            fade_in=fade, fade_out=fade, curve='smooth',
            enabled=True, keys=[dict(time=t, values=PoseSampler(self.pose).values(hand, target, t))]))
        self._submit(pose, '新建姿态片段')
        self.seek.emit(t)

    def delete_clip(self):
        if not self.current_clip():
            return
        pose = copy.deepcopy(self.pose)
        pose['clips'] = [c for c in pose['clips'] if c['id'] != self.clip_id]
        self.clip_id = None
        self._submit(pose, '删除姿态片段')

    def mirror(self):
        if self.document is None:
            return
        hand, target = self.selection
        opposite = 'right' if hand == 'left' else 'left'
        pose = copy.deepcopy(self.pose)
        clip = self.current_clip()
        if clip:
            mirrored = copy.deepcopy(clip)
            mirrored.update(id=uuid.uuid4().hex, hand=opposite)
            for key in mirrored['keys']:
                key['values'] = mirrored_values(key['values'], target)
            pose['clips'].append(mirrored)
        else:
            pose['defaults'][f'{opposite}:{target}'] = mirrored_values(pose['defaults'].get(f'{hand}:{target}', {}), target)
        self._submit(pose, '镜像姿态到另一手')

    def reset_values(self):
        if self.document is None:
            return
        pose = copy.deepcopy(self.pose)
        if self.current_clip():
            self._value_container(pose).clear()
        else:
            pose['defaults'].pop(':'.join(self.selection), None)
        self._submit(pose, '还原姿态参数')

    def _resize_clip(self, *_):
        if self._refreshing or not self.current_clip():
            return
        start, end = self.start.value() - self.offset, self.end.value() - self.offset
        if end - start < .001 - 1e-9:
            self.error.emit('片段至少需要 0.001 秒，请将结束时间设在开始时间之后')
            self._refresh()
            return
        pose = copy.deepcopy(self.pose)
        clip = self.current_clip(pose)
        ratio = (end - start) / (clip['end'] - clip['start'])
        for key in clip['keys']:
            key['time'] = min(end, max(start, start + (key['time'] - clip['start']) * ratio))
        clip.update(start=start, end=end)
        for name in ('fade_in', 'fade_out'):
            clip[name] = min(end - start, clip.get(name, 0.))
        self._submit(pose, '调整姿态片段时长')

    def _key_selected(self):
        if self._refreshing or not self.current_clip() or self.keys.currentRow() < 0:
            return
        self.key_index = self.keys.currentRow()
        self._refresh()
        self.seek.emit(self.current_clip()['keys'][self.key_index]['time'])

    def add_keyframe(self):
        clip = self.current_clip()
        if clip is None:
            return
        t = round(min(clip['end'], max(clip['start'], self.time)), 3)
        t = min(clip['end'], max(clip['start'], t))
        if any(abs(key['time'] - t) < .0005 for key in clip['keys']):
            self.select_clip(clip['id'], t)
            return
        single = copy.deepcopy(clip)
        single.update(fade_in=0., fade_out=0., enabled=True)
        sampler = PoseSampler(dict(version=1, defaults=self.pose['defaults'], clips=[single]))
        pose = copy.deepcopy(self.pose)
        changed = self.current_clip(pose)
        changed['keys'].append(dict(time=t, values=sampler.values(clip['hand'], clip['target'], min(t, math.nextafter(clip['end'], -math.inf)))))
        changed['keys'].sort(key=lambda key: key['time'])
        self.key_index = next(i for i, key in enumerate(changed['keys']) if key['time'] == t)
        self._submit(pose, '添加姿态关键帧')
        self.seek.emit(t)

    def remove_keyframe(self):
        if not self.current_clip() or len(self.current_clip()['keys']) <= 1:
            return
        pose = copy.deepcopy(self.pose)
        self.current_clip(pose)['keys'].pop(self.key_index)
        self.key_index = max(0, self.key_index - 1)
        self._submit(pose, '删除姿态关键帧')

    def _move_key(self, value):
        if self._refreshing or not self.current_clip():
            return
        pose = copy.deepcopy(self.pose)
        clip = self.current_clip(pose)
        key = clip['keys'][self.key_index]
        t = min(clip['end'], max(clip['start'], value - self.offset))
        if any(other is not key and abs(other['time'] - t) < .0005 for other in clip['keys']):
            self.error.emit('该时刻已有关键帧请选择其他时间，或在列表中选中已有帧进行编辑')
            self._refresh()
            return
        key['time'] = t
        clip['keys'].sort(key=lambda key: key['time'])
        self.key_index = clip['keys'].index(key)
        self._submit(pose, '移动姿态关键帧')
        self.seek.emit(t)

    def drag(self, delta, finished=False):
        if self.document is None:
            return
        if delta is None:
            self._drag_origin = None
            self.preview.emit(None)
            return
        if not any(abs(value) > 1e-10 for value in delta.values()):
            self.preview.emit(None)
            if finished:
                self._drag_origin = None
            return
        if self._drag_origin is None:
            self._drag_origin = copy.deepcopy(self.pose)
        pose = copy.deepcopy(self._drag_origin)
        values = self._value_container(pose)
        current = self._values()
        for name, value in delta.items():
            if name in channels(self.selection[1]):
                low, high, _ = channels(self.selection[1])[name]
                values[name] = max(low, min(high, current[name] + value))
        if finished:
            self._drag_origin = None
            self.preview.emit(None)
            self._submit(pose, '拖动手部姿态')
        else:
            self.preview.emit(pose)


class PoseCanvas(QWidget):
    target_selected = Signal(str, str)
    drag_changed = Signal(object, bool)
    pause_requested = Signal()
    feedback = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = PosePreview()
        self.time = 0.
        self.selection = ('left', 'hand')
        self.mode = 'select'
        self.azimuth, self.elevation, self.zoom = -.15, .90, 1.
        self.editable = True
        self.hits, self._drag = [], None
        self._frame = None
        self._feedback = ''
        self.setMinimumSize(280, 180)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip('点击手掌或手指进行选择调整手腕时，先选择上方的移动、转向或高度工具，再拖动手腕圆点右键转动视角，滚轮缩放；Esc 取消拖动')

    def set_document(self, document, chart=None):
        self._drag = None
        self.engine.set_document(document, chart)
        self._frame = None
        self.update()

    def set_pose(self, pose=None):
        self.engine.set_pose(pose)
        self._frame = None
        self.update()

    def set_time(self, t):
        self.time = t
        self._frame = None
        if self.isVisible():
            self.update()

    def select_target(self, hand, target):
        self.selection = hand, target
        self.update()

    def set_view(self, view):
        self.azimuth, self.elevation = {'angle': (-.15, .90), 'top': (0., math.pi / 2),
                                       'side': (.15, .22)}[view]
        self.update()

    def project(self, point):
        x, y, z = point[0], point[1] + .04, point[2] - .02
        ca, sa = math.cos(self.azimuth), math.sin(self.azimuth)
        scale = self.scale()
        return QPointF(self.width() / 2 + (x * ca - y * sa) * scale,
                       self.height() / 2 + (-x * sa - y * ca) * math.sin(self.elevation) * scale - z * math.cos(self.elevation) * scale)

    def scale(self):
        return min(self.width() / .48, max(1, self.height() - 44) / .36) * self.zoom

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#101a28'))
        if self._frame is None:
            self._frame = self.engine.frame(self.time)
        frame = self._frame
        self.hits = []
        if frame is None:
            painter.setPen(QColor('#adbed5'))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             '先打开工程，或打开谱面并生成指法初稿\n即可调整手腕和空闲手指的姿态')
            return
        width, height = frame['screen']
        corners = [self.project((x * width / 2, y * height / 2, 0.)) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        painter.setPen(QPen(QColor('#496681'), 1.5))
        painter.setBrush(QColor('#1c3142'))
        painter.drawPolygon(QPolygonF(corners))
        painter.setPen(QPen(QColor('#2c475d'), 1))
        for i in range(1, 8):
            x = (i / 8 - .5) * width
            painter.drawLine(self.project((x, -height / 2, 0.)), self.project((x, height / 2, 0.)))
        for i in range(1, 5):
            y = (i / 5 - .5) * height
            painter.drawLine(self.project((-width / 2, y, 0.)), self.project((width / 2, y, 0.)))
        for hand, data in frame['hands'].items():
            color = QColor(COLORS[hand])
            selected = self.selection == (hand, 'hand')
            palm = [self.project(p) for p in data['palm']]
            center = QPointF(sum(p.x() for p in palm) / len(palm), sum(p.y() for p in palm) / len(palm))
            palm.sort(key=lambda p: math.atan2(p.y() - center.y(), p.x() - center.x()))
            painter.setPen(QPen(color.lighter(140) if selected else color, 2.5 if selected else 1.5))
            fill = QColor(color)
            fill.setAlpha(78 if selected else 45)
            painter.setBrush(fill)
            painter.drawPolygon(QPolygonF(palm))
            root = self.project(data['root'])
            painter.drawLine(root, center)
            painter.drawEllipse(root, 9. if selected else 6., 9. if selected else 6.)
            self.hits.append((hand, 'hand', root, center, 15.))
            for finger, finger_data in data['fingers'].items():
                points = [self.project(p) for p in finger_data['joints']]
                chosen = self.selection == (hand, finger)
                width_px = max(4., min(14., .008 * self.scale())) + (2 if chosen else 0)
                pen = QPen(color.lighter(150) if chosen else color, width_px, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                for a, b in zip(points, points[1:]):
                    painter.drawLine(a, b)
                    self.hits.append((hand, finger, a, b, max(9., width_px)))
                painter.setPen(QPen(QColor('#d8e7f6'), 1))
                painter.setBrush(QColor('#1b3044'))
                for point in points[:-1]:
                    painter.drawEllipse(point, 2.8, 2.8)
                painter.setBrush(QColor('#86e4ae') if finger_data['down'] else color)
                painter.drawEllipse(points[-1], 5., 5.)
                if finger_data['down']:
                    target = self.project(finger_data['target'])
                    painter.setPen(QPen(QColor('#86e4ae'), 1))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(target, 9., 9.)
            if data['correction'] > .0001:
                requested = self.project(data['requested'])
                painter.setPen(QPen(QColor('#f8c56d'), 1.5, Qt.DashLine))
                painter.drawLine(requested, root)
                painter.drawLine(requested + QPointF(-5, -5), requested + QPointF(5, 5))
                painter.drawLine(requested + QPointF(-5, 5), requested + QPointF(5, -5))
            if selected and self.mode != 'select':
                for delta, color_axis, label in (((.04, 0., 0.), '#fa8b86', 'X'), ((0., .04, 0.), '#8bddb4', 'Y'),
                                                  ((0., 0., .04), '#89aafa', 'Z')):
                    end = self.project(tuple(a + b for a, b in zip(data['root'], delta)))
                    painter.setPen(QPen(QColor(color_axis), 2))
                    painter.drawLine(root, end)
                    painter.drawText(end + QPointF(3, -3), label)
            painter.setPen(color)
            painter.drawText(root + QPointF(12, 17), HAND_NAMES[hand])
        hand, target = self.selection
        data = frame['hands'][hand]
        if target == 'hand':
            count = sum(f['down'] for f in data['fingers'].values())
            feedback = f'{HAND_NAMES[hand]} · {count} 根手指接触中'
            if data['correction'] > .0001:
                feedback += f" · 为保持接触，手腕已修正 {data['correction'] * 1000:.1f} mm（黄叉为设定位置）"
            if data['reach_error'] > .001:
                feedback += f" · 仍超出可达范围 {data['reach_error'] * 1000:.1f} mm，请减小手腕偏移或倾斜角度"
        else:
            finger_data = data['fingers'][target]
            feedback = f'{HAND_NAMES[hand]}{NAMES[target]} · ' + ('接触中，指尖跟随触控轨迹' if finger_data['down'] else
                ('空闲，可调整弯曲、张合和抬起' if finger_data['idle'] >= .999 else '正在衔接触控动作，空闲姿态逐渐过渡'))
        if frame['warning']:
            feedback += ' · ' + frame['warning']
        if feedback != self._feedback:
            self._feedback = feedback
            self.feedback.emit(feedback)
        painter.setPen(QColor('#9eb3ce'))
        painter.drawText(QRectF(12, 8, self.width() - 24, 22), Qt.AlignLeft, '简化关节预览 · 绿环表示接触')
        painter.drawText(QRectF(12, self.height() - 27, self.width() - 24, 22), Qt.AlignLeft,
                         '右键转动视角 · 滚轮缩放 · 点击选择手掌或手指')

    def _hit(self, point):
        found = []
        for hand, target, a, b, radius in self.hits:
            dx, dy = b.x() - a.x(), b.y() - a.y()
            u = max(0., min(1., ((point.x() - a.x()) * dx + (point.y() - a.y()) * dy) / max(1., dx * dx + dy * dy)))
            distance = math.hypot(point.x() - a.x() - dx * u, point.y() - a.y() - dy * u)
            if distance <= radius:
                found.append((distance, hand, target))
        return min(found)[1:] if found else None

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.RightButton:
            self._drag = dict(kind='view', point=event.position(), azimuth=self.azimuth, elevation=self.elevation)
            return
        if event.button() != Qt.LeftButton:
            return
        hit = self._hit(event.position())
        if hit:
            self.selection = hit
            self.target_selected.emit(*hit)
            self.pause_requested.emit()
            if self.editable and self.mode in ('move', 'yaw', 'height') and hit[1] == 'hand':
                self._drag = dict(kind=self.mode, point=event.position())
            self.update()

    def _delta(self, point):
        dx, dy = point.x() - self._drag['point'].x(), point.y() - self._drag['point'].y()
        kind = self._drag['kind']
        if kind == 'yaw':
            return dict(yaw=dx * .008)
        if kind == 'height':
            return dict(z=-dy / (self.scale() * max(.25, math.cos(self.elevation))))
        ca, sa = math.cos(self.azimuth), math.sin(self.azimuth)
        x, y = dx / self.scale(), -dy / (self.scale() * max(.15, math.sin(self.elevation)))
        return dict(x=x * ca + y * sa, y=-x * sa + y * ca)

    def mouseMoveEvent(self, event):
        if self._drag:
            if self._drag['kind'] == 'view':
                delta = event.position() - self._drag['point']
                self.azimuth = self._drag['azimuth'] + delta.x() * .006
                self.elevation = max(.15, min(math.pi / 2, self._drag['elevation'] + delta.y() * .006))
                self.update()
            else:
                self.drag_changed.emit(self._delta(event.position()), False)
        else:
            self.setCursor(Qt.PointingHandCursor if self._hit(event.position()) else Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if self._drag:
            if self._drag['kind'] != 'view':
                self.drag_changed.emit(self._delta(event.position()), True)
            self._drag = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._drag:
            self._drag = None
            self.drag_changed.emit(None, False)
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        self.zoom = max(.45, min(3., self.zoom * 1.12 ** (event.angleDelta().y() / 120)))
        self.update()
        event.accept()


class PoseTimeline(QWidget):
    target_selected = Signal(str, str)
    clip_selected = Signal(str, float)
    seek = Signal(float)
    range_selected = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document = None
        self.time = self.offset = self.start = 0.
        self.span = 12.
        self.selection = ('left', 'hand')
        self.clip_id = ''
        self.rows = [(hand, target) for hand in HANDS for target in TARGETS]
        self.label_width, self.row_height, self.ruler = 94, 23, 27
        self._drag = None
        self.setMinimumHeight(self.ruler + len(self.rows) * self.row_height + 4)
        self.setToolTip('点击片段或菱形关键帧进行编辑；拖动空白处设置下方的“片段起止”，再点“新建姿态片段”左侧名称旁的圆点表示已设整曲默认值Ctrl＋滚轮缩放，滚轮平移')

    def set_document(self, document, offset=0.):
        self.document, self.offset = document, offset
        self.update()

    def set_time(self, t, follow=False):
        self.time = t
        if follow and not self.start <= t <= self.start + self.span * .9:
            self.start = t - self.span * .15
        if self.isVisible():
            self.update()

    def select_target(self, hand, target):
        self.selection = hand, target
        self.update()

    def select_clip(self, identity):
        self.clip_id = identity
        self.update()

    def x(self, t):
        return self.label_width + (t - self.start) / self.span * max(1, self.width() - self.label_width)

    def at(self, x):
        return self.start + (x - self.label_width) / max(1, self.width() - self.label_width) * self.span

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#132031'))
        step = max(.1, 10 ** math.floor(math.log10(self.span / 6)))
        while self.span / step > 12:
            step *= 2
        first = math.floor((self.start + self.offset) / step) * step - self.offset
        for i in range(math.ceil(self.span / step) + 2):
            t = first + i * step
            x = self.x(t)
            if x < self.label_width:
                continue
            painter.setPen(QPen(QColor('#2a3b50'), 1))
            painter.drawLine(QPointF(x, self.ruler), QPointF(x, self.height()))
            painter.setPen(QColor('#aebed2'))
            painter.drawText(QPointF(x + 3, 17), f'{t + self.offset:g}')
        for row, (hand, target) in enumerate(self.rows):
            y = self.ruler + row * self.row_height
            if self.selection == (hand, target):
                painter.fillRect(QRectF(0, y, self.width(), self.row_height), QColor('#243951'))
            painter.setPen(QPen(QColor('#25364b'), 1))
            painter.drawLine(0, y, self.width(), y)
            painter.setPen(QColor(COLORS[hand]))
            label = ('左' if hand == 'left' else '右') + ('手腕 / 整手' if target == 'hand' else NAMES[target])
            if self.document and self.document.pose['defaults'].get(f'{hand}:{target}'):
                label += ' ●'
            painter.drawText(QRectF(6, y, self.label_width - 6, self.row_height), Qt.AlignVCenter, label)
        painter.save()
        painter.setClipRect(QRectF(self.label_width, self.ruler, self.width() - self.label_width, self.height()))
        if self.document:
            for clip in self.document.pose['clips']:
                if clip['end'] < self.start or clip['start'] > self.start + self.span:
                    continue
                row = self.rows.index((clip['hand'], clip['target']))
                y = self.ruler + row * self.row_height + 4
                rect = QRectF(self.x(clip['start']), y, max(2., self.x(clip['end']) - self.x(clip['start'])), self.row_height - 8)
                color = QColor(COLORS[clip['hand']])
                color.setAlpha(140 if clip.get('enabled', True) else 40)
                painter.setBrush(color)
                painter.setPen(QPen(QColor('#f1f6ff') if clip['id'] == self.clip_id else color, 2 if clip['id'] == self.clip_id else 1))
                painter.drawRoundedRect(rect, 3, 3)
                painter.setPen(QPen(QColor('#d1deee'), 1, Qt.DotLine))
                for t in (clip['start'] + clip.get('fade_in', 0.), clip['end'] - clip.get('fade_out', 0.)):
                    painter.drawLine(QPointF(self.x(t), y), QPointF(self.x(t), y + rect.height()))
                painter.setBrush(QColor('#eff5ff'))
                painter.setPen(Qt.NoPen)
                for key in clip['keys']:
                    x, middle = self.x(key['time']), y + rect.height() / 2
                    painter.drawPolygon(QPolygonF([QPointF(x, middle - 4), QPointF(x + 4, middle),
                                                   QPointF(x, middle + 4), QPointF(x - 4, middle)]))
        if self._drag:
            a, b = sorted(self._drag)
            painter.fillRect(QRectF(self.x(a), self.ruler, self.x(b) - self.x(a), self.height()), QColor(120, 179, 236, 55))
        painter.setPen(QPen(QColor('#f7d47f'), 1.5))
        painter.drawLine(QPointF(self.x(self.time), self.ruler), QPointF(self.x(self.time), self.height()))
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        row = int((event.position().y() - self.ruler) // self.row_height)
        if 0 <= row < len(self.rows):
            self.selection = self.rows[row]
            self.target_selected.emit(*self.selection)
        if event.position().x() < self.label_width:
            self.update()
            return
        t = self.at(event.position().x())
        if self.document and 0 <= row < len(self.rows):
            for clip in reversed(self.document.pose['clips']):
                if (clip['hand'], clip['target']) == self.rows[row] and clip['start'] <= t <= clip['end']:
                    closest = min(clip['keys'], key=lambda key: abs(key['time'] - t))
                    self.clip_selected.emit(clip['id'], closest['time'])
                    return
        self._drag = (t, t)
        self.seek.emit(t)
        self.update()

    def mouseMoveEvent(self, event):
        if self._drag:
            self._drag = self._drag[0], self.at(event.position().x())
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drag:
            a, b = sorted(self._drag)
            if b - a > .01:
                self.range_selected.emit(a, b)
            self._drag = None
            self.update()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120
        if event.modifiers() & Qt.ControlModifier:
            anchor = self.at(event.position().x())
            fraction = (anchor - self.start) / self.span
            self.span = max(.15, min(3600., self.span / 1.2 ** steps))
            self.start = anchor - fraction * self.span
        else:
            self.start -= steps * self.span * .12
        self.update()
        event.accept()
