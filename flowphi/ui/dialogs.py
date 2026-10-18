from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from flowphi.ui.sources import ROOT
from flowphi.config import OUTPUTS


def path_row(value, parent, caption, file_filter, directory=False):
    widget = QWidget(parent)
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    field = QLineEdit(value)
    layout.addWidget(field)
    button = QPushButton('选择…')
    layout.addWidget(button)

    def browse():
        if directory:
            path = QFileDialog.getExistingDirectory(parent, caption, field.text() or str(ROOT))
        else:
            path, _ = QFileDialog.getOpenFileName(parent, caption, field.text() or str(ROOT), file_filter)
        if path:
            field.setText(path)

    button.clicked.connect(browse)
    return widget, field


def spin(value, low, high, decimals=None):
    widget = QSpinBox() if decimals is None else QDoubleSpinBox()
    if decimals is not None:
        widget.setDecimals(decimals)
    widget.setRange(low, high)
    widget.setValue(value)
    return widget


def buttons(dialog, layout, accept='确定'):
    box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    box.button(QDialogButtonBox.Ok).setText(accept)
    box.button(QDialogButtonBox.Cancel).setText('取消')
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    layout.addWidget(box)


class InterfaceScaleDialog(QDialog):
    def __init__(self, selected, current, parent=None):
        super().__init__(parent)
        self.setWindowTitle('界面缩放')
        self.setMinimumWidth(430)
        layout = QVBoxLayout(self)
        intro = QLabel('同时缩放字体、按钮、面板和时间轴100% 遵循系统显示缩放')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.scale = spin(selected, 50, 200)
        self.scale.setSingleStep(5)
        self.scale.setSuffix(' %')
        self.scale.setToolTip('小于 100% 可在同一窗口中显示更多内容；大于 100% 可放大整个界面')
        form.addRow('界面比例', self.scale)
        layout.addLayout(form)
        reset = QPushButton('恢复 100%')
        reset.clicked.connect(lambda: self.scale.setValue(100))
        layout.addWidget(reset)
        note = QLabel(f'当前生效：{current}%保存后重启 FlowPhi 生效')
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons(self, layout, '保存设置')


class PerformanceDialog(QDialog):
    def __init__(self, values, parent=None):
        super().__init__(parent)
        self.setWindowTitle('性能设置')
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        label = QLabel('编辑卡顿时，可降低预览帧率或画面宽度生成视频时会单独设置视频画质')
        label.setWordWrap(True)
        layout.addWidget(label)
        form = QFormLayout()
        layout.addLayout(form)
        self.fields = {}
        definitions = [('preview_fps', '编辑预览帧率 / FPS', 30, 10, 60),
                       ('preview_width', '编辑画面宽度 / 像素', 960, 480, 1920),
                       ('lookahead', '显示未来音符 / 秒', 8, 2, 30),
                       ('timeline_span', '时间轴每屏时长 / 秒', 12, 2, 120),
                       ('replan_beam', '分配候选方案数', 32, 1, 128),
                       ('replan_contacts', '分配动作数上限', 1200, 32, 5000),
                       ('history_mib', '撤销记录内存上限 / MiB', 16, 4, 128)]
        tips = dict(preview_fps='数值越高，播放越流畅，计算量也越大',
                    preview_width='降低后预览更模糊，但绘制更快', lookahead='预览中最多显示未来多少秒的音符',
                    timeline_span='时间轴一屏默认显示多少秒；也可用 Ctrl＋滚轮调整',
                    replan_beam='重新分配手指时，每一步保留的候选方案数越多越慢，不保证结果更好',
                    replan_contacts='包含所选动作及参与计算的前后动作超过时可缩小选择范围或提高上限',
                    history_mib='超过上限会丢弃较早的撤销记录，保留最近一次完整操作')
        for key, name, default, low, high in definitions:
            self.fields[key] = spin(values.get(key, default), low, high)
            self.fields[key].setToolTip(tips[key])
            form.addRow(name, self.fields[key])
        note = QLabel('鼠标停在设置项上可查看说明重新分配手指的候选数越多，等待时间通常越长')
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons(self, layout)

    def values(self):
        return {key: field.value() for key, field in self.fields.items()}


class BatchPathDialog(QDialog):
    def __init__(self, contacts, selected_points, reference, parent=None):
        super().__init__(parent)
        self.setWindowTitle('批量修改轨迹')
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        count = sum(len(c['points']) for c in contacts)
        summary = QLabel(f'已选 {len(contacts)} 段动作，共 {count:,} 个控制点先选修改范围，再选操作；Ctrl+Z 可撤销整次修改')
        summary.setWordWrap(True)
        layout.addWidget(summary)
        form = QFormLayout()
        layout.addLayout(form)
        self.scope = QComboBox()
        self.scope.addItem('所选动作的全部控制点', 'all')
        if len(contacts) == 1 and selected_points:
            self.scope.addItem(f'仅表格中选中的 {len(selected_points)} 个控制点', 'rows')
            if len(selected_points) > 1:
                self.scope.setCurrentIndex(1)
        form.addRow('修改范围', self.scope)
        self.only_hold = QCheckBox('只修改类型为 Hold 的动作')
        self.only_hold.setToolTip('按动作列表中的类型筛选；标为 Flick 的共享动作不会处理')
        self.only_hold.setEnabled(len(contacts) > 1)
        form.addRow('', self.only_hold)
        self.operation = QComboBox()
        for text, value in [('平移所选点', 'translate'), ('统一设定坐标', 'set'),
                            ('停在各段起点', 'freeze'), ('首尾直线连接', 'straighten'),
                            ('减少控制点', 'simplify')]:
            self.operation.addItem(text, value)
        form.addRow('操作', self.operation)
        self.x_enabled, self.y_enabled = QCheckBox('X'), QCheckBox('Y')
        self.x_enabled.setChecked(True)
        self.y_enabled.setChecked(True)
        self.x, self.y = spin(0., -100., 100., 3), spin(0., -100., 100., 3)
        for label, check, field in [('横向', self.x_enabled, self.x), ('纵向', self.y_enabled, self.y)]:
            row = QWidget()
            box = QHBoxLayout(row)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(check)
            box.addWidget(field, 1)
            form.addRow(label, row)
            check.toggled.connect(lambda checked: self._update_fields())
        self.tolerance = spin(.5, .01, 10., 2)
        self.tolerance.setSuffix(' mm')
        form.addRow('允许的位置变化', self.tolerance)
        self.tolerance.setToolTip('减少控制点后，同一时刻的位置最多改变多少毫米按工程中的虚拟屏幕尺寸计算；数值越大，通常能删去更多点')
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        note = QLabel('按下和抬起时间保持不变只选部分点时，分别处理每段连续选区；修改后可用“检查方案”检查轨迹')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.reference = reference
        self.operation.currentIndexChanged.connect(self._operation_changed)
        self._operation_changed()
        buttons(self, layout, '应用到所选轨迹')

    def _operation_changed(self, *_):
        mode = self.operation.currentData()
        setting = mode == 'set'
        for field, value in ((self.x, self.reference[0]), (self.y, self.reference[1])):
            field.setRange(0. if setting else -100., 100.)
            field.setValue(value * 100 if setting else 0.)
            field.setSuffix(' %')
        self.x.setToolTip('X＝0% 在屏幕最左侧，100% 在最右侧' if setting else '例如 +5 表示向右移动屏幕宽度的 5%，−5 表示向左移动')
        self.y.setToolTip('Y＝0% 在屏幕最上方，100% 在最下方' if setting else '例如 +5 表示向下移动屏幕高度的 5%，−5 表示向上移动')
        self.hint.setText(dict(
            translate='勾选要移动的方向，输入位移：X 正数向右，Y 正数向下；负数反向所有目标点移动相同距离',
            set='把所选点放到同一个 X 或 Y 坐标两个方向都勾选时，所有目标点会落在同一位置',
            freeze='每段都停在自己的第一个点，直到这一段结束只保留首尾两个时间点，删除中间点',
            straighten='每段只保留首尾点，让手指在两点之间匀速直线移动',
            simplify='删去冗余点，保留首尾和必要转折位置变化控制在下方设定范围内，之后只需编辑留下的点'
        )[mode])
        self._update_fields()

    def _update_fields(self):
        coordinates = self.operation.currentData() in ('translate', 'set')
        self.x_enabled.setEnabled(coordinates)
        self.y_enabled.setEnabled(coordinates)
        self.x.setEnabled(coordinates and self.x_enabled.isChecked())
        self.y.setEnabled(coordinates and self.y_enabled.isChecked())
        self.tolerance.setEnabled(self.operation.currentData() == 'simplify')

    def values(self):
        return dict(operation=self.operation.currentData(), only_hold=self.only_hold.isChecked(),
                    x=self.x.value() / 100 if self.x_enabled.isChecked() else None,
                    y=self.y.value() / 100 if self.y_enabled.isChecked() else None,
                    tolerance=self.tolerance.value() / 1000)


class FlickDialog(QDialog):
    def __init__(self, entries, chart, offset=0., parent=None):
        super().__init__(parent)
        from flowphi.ui.gestures import flick_defaults
        self.setWindowTitle('Flick 滑动设置')
        self.setMinimumWidth(570)
        self.entries, self.chart = entries, chart
        layout = QVBoxLayout(self)
        count = sum(len(flicks) for _, _, flicks in entries)
        intro = QLabel(f'所选动作包含 {count} 个 Flick滑动会经过当前轨迹在判定时刻的位置；选中多个 Flick 时，统一使用下方设置')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        layout.addLayout(form)
        self.target = QComboBox()
        self.target.addItem(f'全部 {count} 个 Flick', None)
        self.target.setToolTip('初始数值根据第一个目标推算选择“全部”时，这组设置会应用到所有目标')
        if count > 1:
            for _, _, flicks in entries:
                for nid, _, beat in flicks:
                    self.target.addItem(f'音符 {nid} · {beat + offset:.3f} s', nid)
        form.addRow('修改哪些 Flick', self.target)
        self.direction = QComboBox()
        for label, value in [('垂直判定线 · 向上', 'up'), ('垂直判定线 · 向下', 'down'),
                             ('平行判定线 · 向右', 'right'), ('平行判定线 · 向左', 'left'),
                             ('自定义角度', 'custom')]:
            self.direction.addItem(label, value)
        form.addRow('滑动方向', self.direction)
        self.direction.setToolTip('上、下、左、右均相对判定时刻的判定线，会随判定线一起旋转需要固定屏幕方向时，选“自定义角度”和“相对屏幕”')
        self.angle_space = QComboBox()
        self.angle_space.addItem('相对判定线', 'line')
        self.angle_space.addItem('相对屏幕', 'screen')
        form.addRow('角度基准', self.angle_space)
        self.angle = spin(0., -360., 360., 1)
        self.angle.setSuffix(' °')
        form.addRow('自定义角度', self.angle)
        self.angle_space.setToolTip('相对判定线：方向随判定线旋转相对屏幕：方向固定，不受判定线旋转影响')
        self.angle.setToolTip('0° 向右，90° 向上，180° 向左，−90° 向下；方向相对上方选择的基准')
        self.length = spin(16., .01, 200., 2)
        self.length.setSuffix(' % 屏幕宽度')
        form.addRow('滑动总长度', self.length)
        self.length.setToolTip('从开始滑动到滑完的总距离所有方向都按屏幕宽度计算，例如 10% 表示屏幕宽度的十分之一')
        self.duration = spin(70, 1, 10000)
        self.duration.setSuffix(' ms')
        form.addRow('滑动总时长', self.duration)
        self.duration.setToolTip('只计算移动的时间，不含滑完后的停留例如总时长 70 ms、提前 25 ms 开始，会在判定后 45 ms 滑完')
        self.lead = spin(25, 0, self.duration.value())
        self.lead.setSuffix(' ms')
        form.addRow('提前多久开始', self.lead)
        self.lead.setToolTip('相对该音符的判定时刻提前多少毫秒开始滑动，不能超过滑动总时长')
        self.release = spin(1, 1, 60000)
        self.release.setSuffix(' ms')
        form.addRow('滑完后停留时间', self.release)
        self.release.setToolTip('独立 Flick 停在终点，经过这段时间后抬起若同一动作还关联其他音符，只在必要时延后原有抬起时间')
        hint = QLabel('四个预设方向都相对判定线自定义角度可选“相对屏幕”：0° 向右，90° 向上，180° 向左，−90° 向下')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.shared_hint = QLabel()
        self.shared_hint.setWordWrap(True)
        self.shared_hint.setToolTip('共用动作会保留滑动时段外的控制点，但这些点与新滑动首尾点之间的连线也会随之改变滑完后仍可能继续原有轨迹')
        layout.addWidget(self.shared_hint)
        self.timing_hint = QLabel()
        self.timing_hint.setWordWrap(True)
        layout.addWidget(self.timing_hint)

        def target_changed(*_):
            nid = self.target.currentData()
            entry = next((row for row in entries if nid is None or any(f[0] == nid for f in row[2])))
            _, contact, flicks = entry
            chosen = [f for f in flicks if nid is None or f[0] == nid]
            values = flick_defaults(contact, chosen, chart)
            self.direction.setCurrentIndex(self.direction.findData(values['direction']))
            self.angle_space.setCurrentIndex(0)
            self.angle.setValue(values['angle'])
            self.length.setValue(values['length'])
            self.duration.setValue(values['duration_ms'])
            self.lead.setValue(values['lead_ms'])
            self.release.setValue(values['release_ms'])
            shared = any(len(c['note_ids']) > 1 for _, c, fs in entries if nid is None or any(f[0] == nid for f in fs))
            self.shared_hint.setText('部分 Flick 与其他音符共用一段动作：替换滑动时段内的控制点，保留区间外的点；'
                                     '仅在需要时提前按下或延后抬起需要提前抬起时，用“修改抬起时间”' if shared else
                                     '应用后会重设这些 Flick 的按下和抬起时间Ctrl+Z 可撤销整次修改')
            self._update_flick_fields()

        self.direction.currentIndexChanged.connect(self._update_flick_fields)
        self.duration.valueChanged.connect(lambda value: self.lead.setMaximum(value))
        for field in (self.duration, self.lead, self.release):
            field.valueChanged.connect(self._update_flick_fields)
        self.target.currentIndexChanged.connect(target_changed)
        target_changed()
        buttons(self, layout, '应用 Flick 设置')

    def _update_flick_fields(self, *_):
        custom = self.direction.currentData() == 'custom'
        self.angle.setEnabled(custom)
        self.angle_space.setEnabled(custom)
        self.timing_hint.setText(f'判定前 {self.lead.value()} ms 开始，判定后 '
                                f'{self.duration.value() - self.lead.value()} ms 滑完独立 Flick 再停留 '
                                f'{self.release.value()} ms 抬起；共用动作按上方说明处理修改后可用“检查方案”检查结果')

    def values(self):
        return dict(note_id=self.target.currentData(), direction=self.direction.currentData(),
                    angle=self.angle.value(), angle_space=self.angle_space.currentData(),
                    length=self.length.value() / 100, duration_ms=self.duration.value(),
                    lead_ms=self.lead.value(), release_ms=self.release.value())


class ReleaseDialog(QDialog):
    def __init__(self, contacts, offset=0., parent=None):
        super().__init__(parent)
        self.setWindowTitle('修改手指抬起时间')
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        summary = (f'按下 {contacts[0]["start"] + offset:.3f} s → 抬起 {contacts[0]["end"] + offset:.3f} s'
                   if len(contacts) == 1 else f'已选 {len(contacts)} 段动作，可统一提前、延后抬起，或设置每段的按住时长')
        label = QLabel(summary)
        label.setWordWrap(True)
        layout.addWidget(label)
        form = QFormLayout()
        layout.addLayout(form)
        self.mode = QComboBox()
        if len(contacts) == 1:
            self.mode.addItem('指定抬起时刻（音乐时间）', 'absolute')
        self.mode.addItem('提前或延后各自的抬起时间', 'offset')
        self.mode.addItem('设定每段的按住时长', 'duration')
        form.addRow('修改方式', self.mode)
        self.value = spin(0., -86400., 86400., 3)
        self.value_label = QLabel()
        form.addRow(self.value_label, self.value)
        self.reshape = QComboBox()
        self.reshape.addItem('提前时截断轨迹；延后时停在末点', 'trim')
        self.reshape.addItem('路径不变，整体加快或放慢移动', 'scale')
        form.addRow('轨迹处理', self.reshape)
        self.reshape.setToolTip('截断：提前抬起会删掉后半段路径，延后抬起则在原终点等待整体加快或放慢：按下时刻不变，所有控制点按同一比例提前或延后')
        self.mode_hint = QLabel()
        self.mode_hint.setWordWrap(True)
        layout.addWidget(self.mode_hint)
        note = QLabel('此处修改所选动作的结束时间若后段设为“继续按住”或符合自动接续条件，手指会接着完成后段，直到整段接续结束才抬起修改后用“检查方案”确认 Hold 是否按到结束、其他音符是否仍有对应动作Ctrl+Z 可撤销')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.offset = offset

        def update(*_):
            mode = self.mode.currentData()
            self.value_label.setText(dict(absolute='抬起时刻', offset='提前 / 延后量', duration='按住时长')[mode])
            self.mode_hint.setText(dict(
                absolute='填写音乐播放到第几秒时抬起例如 12.500 表示在 12 秒 500 毫秒处抬起',
                offset='负数提前，正数延后例如 −50 让每段动作比原来早 50 ms 抬起；按下时刻保持',
                duration='从每段动作各自的按下时刻起计算例如 200 表示每段都按住 200 ms，再抬起')[mode])
            self.value.setToolTip(self.mode_hint.text())
            self.value.setDecimals(3 if mode == 'absolute' else 0)
            self.value.setRange(-86400. if mode == 'absolute' else -60000. if mode == 'offset' else 1.,
                                86400. if mode == 'absolute' else 86400000. if mode == 'duration' else 60000.)
            self.value.setSuffix(' s' if mode == 'absolute' else ' ms')
            self.value.setValue(contacts[0]['end'] + offset if mode == 'absolute' else
                                (contacts[0]['end'] - contacts[0]['start']) * 1000 if mode == 'duration' else 0.)

        self.mode.currentIndexChanged.connect(update)
        update()
        buttons(self, layout, '应用抬起时间')

    def values(self):
        mode = self.mode.currentData()
        return dict(mode=mode, value=self.value.value() - self.offset if mode == 'absolute' else self.value.value() / 1000,
                    reshape=self.reshape.currentData())


class GenerateDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle('生成指法初稿')
        self.setMinimumWidth(600)
        layout = QVBoxLayout(self)
        text = '为当前谱面生成可编辑的指法初稿，之后可分别调整指法、轨迹和手部姿态选择手部模型会测量手型；留空使用通用手型'
        if getattr(parent, 'document', None) is not None:
            text += '\n\n重新生成会替换当前指法和轨迹修改；本曲的手部姿态设置会保留'
        intro = QLabel(text)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        layout.addLayout(form)
        row, self.model = path_row(settings.get('model', ''), self, '选择手部模型', 'Blender 模型 (*.blend)')
        self.model.setPlaceholderText('留空使用通用手型')
        self.model.setToolTip('选择 Blender 手部模型（.blend）后，按测量的手型生成指法测量结果会保存；生成视频时请使用同一模型')
        form.addRow('手部模型（可选）', row)
        row, self.blender = path_row(settings.get('blender', ''), self, '选择 Blender', 'Blender (blender.exe blender);;所有文件 (*)')
        self.blender.setPlaceholderText('自动查找已安装的 Blender')
        self.blender.setToolTip('选择手部模型时，用 Blender 测量手型留空会自动查找已安装的 Blender')
        form.addRow('Blender', row)
        self.fingers = QComboBox()
        self.fingers.addItem('左右手各用食指', ['index'])
        self.fingers.addItem('双手食指和中指', ['index', 'middle'])
        self.fingers.addItem('全部十指', ['index', 'middle', 'ring', 'thumb', 'little'])
        self.fingers.setCurrentIndex(['index', 'index_middle', 'all'].index(settings.get('generator_fingers', 'all')))
        form.addRow('使用手指', self.fingers)
        self.mode = QComboBox()
        self.mode.addItem('兼顾移动与手指负担', 'balanced')
        self.mode.addItem('优先分散手指负担', 'load')
        self.mode.setCurrentIndex(self.mode.findData(settings.get('fingering_objective', 'balanced')))
        self.mode.setToolTip('兼顾模式同时考虑移动和手指负担；分散负担模式优先考虑各手指的使用次数及连续击打压力')
        form.addRow('分配偏好', self.mode)
        self.speed = QCheckBox('考虑移动速度和加速度')
        self.speed.setToolTip('分配时考虑手指能否及时移动到下一个位置关闭后更容易生成初稿，但可能出现很快的移动')
        self.speed.setChecked(settings.get('speed_limits', True))
        form.addRow('', self.speed)
        self.windows = QCheckBox('允许在判定窗口内微调击打时刻')
        self.windows.setChecked(settings.get('judgement_windows', False))
        self.windows.setToolTip('让手指在方案允许的判定时间范围内略微提前或延后击打；原谱面的音符时间保持')
        form.addRow('', self.windows)
        self.beam = spin(settings.get('solver_beam', 8), 1, 32)
        self.beam.setToolTip('算法每一步保留多少种分配方案继续尝试数值越大，通常越慢；不保证一定能找到结果')
        form.addRow('每步保留方案数', self.beam)
        self.candidates = spin(settings.get('solver_candidates', 3), 1, 8)
        self.candidates.setToolTip('每根手指保留多少种落点或轨迹候选数值越大，通常越慢；不保证一定能找到结果')
        form.addRow('每指保留轨迹数', self.candidates)
        self.keep_unplanned = QCheckBox('未找到分配时保留临时指法')
        self.keep_unplanned.setChecked(settings.get('keep_unplanned', True))
        self.keep_unplanned.setToolTip('当前设置下未找到可行分配时，先用临时指法保留音符并继续生成，在“问题”页标出待检查项关闭后会停止生成未找到分配不代表谱面一定无解')
        form.addRow('', self.keep_unplanned)
        buttons(self, layout, '生成初稿')

    def values(self):
        return dict(model=self.model.text().strip(), blender=self.blender.text().strip(),
                    keep_unplanned=self.keep_unplanned.isChecked(),
                    settings=dict(fingers=self.fingers.currentData(), fingering_objective=self.mode.currentData(),
                                  speed_limits=self.speed.isChecked(), judgement_windows=self.windows.isChecked(),
                                  beam_width=self.beam.value(), candidates_per_finger=self.candidates.value()))


class RenderDialog(QDialog):
    def __init__(self, values, start, duration, parent=None, full=False):
        super().__init__(parent)
        self.setWindowTitle('生成整曲视频' if full else '生成片段视频')
        self.setMinimumWidth(620)
        self.full = full
        layout = QVBoxLayout(self)
        intro = QLabel('使用点击“开始生成视频”时的指法、轨迹和手部姿态生成视频生成期间可以继续编辑，新修改会用于下一次生成')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        layout.addLayout(form)
        row, self.model = path_row(values.get('model', ''), self, '选择手部模型', 'Blender 模型 (*.blend)')
        self.model.setToolTip('用于生成真实手部动画的 Blender 模型（.blend）若生成指法时测量过手型，请使用同一模型')
        form.addRow('手部模型', row)
        row, self.background = path_row(values.get('background') or '', self, '选择曲绘', '曲绘 (*.png *.jpg *.jpeg *.webp);;所有文件 (*)')
        self.background.setPlaceholderText('自动使用谱面曲绘，也可在此选择文件')
        self.background.setToolTip('用于谱面内背景和画面外的模糊背景留空时按谱面信息自动查找曲绘')
        form.addRow('曲绘', row)
        row, self.resources = path_row(values.get('resources', ''), self, '选择资源包', '资源包 (*.zip);;所有文件 (*)')
        self.resources.setPlaceholderText('资源包 ZIP，也可以直接填写资源目录')
        form.addRow('资源包', row)
        row, self.blender = path_row(values.get('blender', ''), self, '选择 Blender', 'Blender (blender.exe blender);;所有文件 (*)')
        self.blender.setPlaceholderText('自动查找已安装的 Blender')
        self.blender.setToolTip('用于生成手部动画的 Blender 程序留空会自动查找已安装的 Blender')
        form.addRow('Blender', row)
        self.start = spin(start, 0, 86400, 3)
        self.duration = spin(duration, .1, 86400, 3)
        self.start.setEnabled(not full)
        self.duration.setEnabled(not full)
        self.start.setToolTip('从音乐播放到第几秒开始生成视频生成整曲时自动确定范围')
        self.duration.setToolTip('从开始时刻起生成多少秒的视频生成整曲时自动确定范围')
        form.addRow('开始时刻 / 秒', self.start)
        form.addRow('视频时长 / 秒', self.duration)
        self.resolution = QComboBox()
        for label, dimensions in [('360p · 快速预览', (640, 360)), ('720p', (1280, 720)), ('1080p', (1920, 1080))]:
            self.resolution.addItem(label, dimensions)
        prefix = 'render' if full else 'clip'
        dimensions = tuple(map(int, values.get(prefix + '_resolution', '1280x720' if full else '640x360').split('x')))
        self.resolution.setCurrentIndex(next(i for i in range(self.resolution.count())
                                             if tuple(self.resolution.itemData(i)) == dimensions))
        form.addRow('分辨率', self.resolution)
        self.fps = QComboBox()
        for n in (15, 30, 60):
            self.fps.addItem(str(n), n)
        self.fps.setCurrentIndex(self.fps.findData(values.get(prefix + '_fps', 30)))
        form.addRow('帧率 / FPS', self.fps)
        self.auto_check = QCheckBox('生成前自动检查方案')
        self.auto_check.setChecked(values.get('auto_check', False))
        form.addRow('', self.auto_check)
        folder = OUTPUTS / (('video-' if full else 'preview-') + datetime.now().strftime('%Y%m%d-%H%M%S'))
        row, self.output = path_row(str(folder), self, '选择输出目录', '', directory=True)
        self.output.setToolTip('视频及本次使用的动作数据会写入这个目录每次选择新的目录，便于保留之前的结果')
        form.addRow('输出目录', row)
        buttons(self, layout, '开始生成视频')

    def values(self):
        width, height = self.resolution.currentData()
        return dict(model=self.model.text().strip(), background=self.background.text().strip(), resources=self.resources.text().strip(),
                    blender=self.blender.text().strip(), width=width, height=height, fps=self.fps.currentData(),
                    start=self.start.value(), duration=self.duration.value(), full=self.full,
                    auto_check=self.auto_check.isChecked())
