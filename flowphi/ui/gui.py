from __future__ import annotations

import copy
from datetime import datetime
import json
import logging
from pathlib import Path
import time
import uuid

from PySide6.QtCore import (QItemSelection, QItemSelectionModel, Qt, QThread,
                           QTimer, QUrl, Slot)
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QPalette
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDockWidget,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSplitter, QTabWidget, QTableView, QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from flowphi.ui.dialogs import (BatchPathDialog, FlickDialog, GenerateDialog, InterfaceScaleDialog,
                               PerformanceDialog, ReleaseDialog, RenderDialog)
from flowphi.ui.appearance import app_settings, interface_scale
from flowphi.ui.jobs import AsyncCall, JobRunner
from flowphi.ui.model import PlanDocument, read_document, write_json_atomic
from flowphi.ui.pose import PoseCanvas, PosePanel, PoseTimeline
from flowphi.ui.task_status import TASK_NAMES, TaskStatus
from flowphi.ui.sources import ChartSelectionRequired, DEFAULT_CACHE, ROOT, create_demo, load_source
from flowphi.config import APP_LOG, ASSETS, OUTPUTS, PROJECTS
from flowphi.ui.validation import draft_warnings, edit_check
from flowphi.ui.widgets import (ChartCanvas, ContactTableModel, FINGER_NAMES, FingeringTimeline,
                            PathTableModel, contact_label, finger_name)
from flowphi.motion import point_at

STYLE = '''
QMainWindow, QDialog, QWidget#workspace { background: #111824; color: #e2eaf5; }
QLabel { color: #d6e1f0; }
QToolBar { background: #1a2636; border: 0; spacing: 5px; padding: 7px; }
QToolButton, QPushButton { color: #e2eaf5; background: #27374c; border: 1px solid #3a4b63; border-radius: 5px; padding: 6px 9px; }
QToolButton:hover, QPushButton:hover { background: #354d6a; }
QToolButton:checked, QPushButton:checked { background: #35587e; }
QToolButton:disabled, QPushButton:disabled { color: #65748a; background: #1c2838; border-color: #293649; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { color: #e2eaf5; background: #1a2738; border: 1px solid #3a4b63; border-radius: 4px; padding: 4px; }
QComboBox QAbstractItemView { color: #e2eaf5; background: #23344b; selection-background-color: #41658f; }
QTableView, QTreeWidget, QPlainTextEdit { color: #d6e1f0; background: #142030; alternate-background-color: #1a293c; border: 1px solid #304158; selection-background-color: #354e70; selection-color: white; gridline-color: #25364c; }
QHeaderView { background: #142030; color: #b5c5dc; }
QHeaderView::section { background: #213146; color: #b5c5dc; border: 0; padding: 6px; }
QTableCornerButton::section { background: #213146; border: 1px solid #304158; }
QScrollArea { background: #111824; border: 1px solid #304158; }
QTabBar::tab { color: #9eb1cb; background: #182536; padding: 8px 16px; }
QTabBar::tab:selected { color: #ecf4ff; background: #2b4260; }
QTabWidget::pane { border: 1px solid #304158; }
QSplitter::handle { background: #26364b; }
QMenuBar, QMenu { color: #e2eaf5; background: #1b293b; }
QMenu::item:selected { background: #355476; }
QStatusBar { color: #acbdd4; background: #1a2636; }
QCheckBox { color: #c7d4e7; spacing: 6px; }
QToolTip { color: #edf4ff; background: #263a54; border: 1px solid #526b8a; }
'''


def editor_palette():
    """Cover native/auto-filled surfaces that are not painted by the stylesheet."""
    palette = QPalette()
    for role, color in (
        (QPalette.Window, '#111824'), (QPalette.WindowText, '#d6e1f0'),
        (QPalette.Base, '#142030'), (QPalette.AlternateBase, '#1a293c'),
        (QPalette.Text, '#d6e1f0'), (QPalette.Button, '#27374c'),
        (QPalette.ButtonText, '#e2eaf5'), (QPalette.Highlight, '#354e70'),
        (QPalette.HighlightedText, '#ffffff'), (QPalette.PlaceholderText, '#9eb1cb'),
        (QPalette.Light, '#526b8a'), (QPalette.Midlight, '#3a4b63'),
        (QPalette.Mid, '#304158'), (QPalette.Dark, '#111824'),
        (QPalette.Shadow, '#0a101a'), (QPalette.ToolTipBase, '#263a54'),
        (QPalette.ToolTipText, '#edf4ff'), (QPalette.Link, '#64afff'),
    ):
        palette.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor('#65748a'))
    return palette


class EditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('flowphi')
        # Qt stylesheets do not propagate an explicitly set widget palette to
        # plain scroll-area children. Set the application palette as their base.
        QApplication.setPalette(editor_palette())
        self.setStyleSheet(STYLE)
        available = self.screen().availableGeometry()
        self.setMinimumSize(960, 600)
        self.resize(min(1360, available.width() - 32), min(880, available.height() - 64))
        self.settings = app_settings()
        self.performance = {key: self.settings.value(key, default, type=int) for key, default in
            [('preview_fps', 30), ('preview_width', 960), ('lookahead', 8), ('timeline_span', 12),
             ('replan_beam', 32), ('replan_contacts', 1200), ('history_mib', 16)]}
        self.source = self.document = None
        self.selected_ids = set()
        self.playing = False
        self.music_time = 0.
        self.duration = 1.
        self.anchor_time = self.anchor_clock = 0.
        self._play_rate = 1.
        self._sync_selection = False
        self._seeking = False
        self._pending_close = False
        self._io = {}
        self._io_handlers = {}
        self._problems = []
        self._last_output = None
        self._job_log = None
        self.jobs = JobRunner(self)
        self.jobs.progress.connect(self._progress)
        self.jobs.progress_details.connect(self._job_progress)
        self.jobs.log_line.connect(self._log)
        self.jobs.warning.connect(lambda message: self.task_status.note_warning())
        self.jobs.busy_changed.connect(self._busy_changed)
        self.jobs.completed.connect(self._job_completed)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(.6)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio)
        self.player.positionChanged.connect(self._audio_position)
        self.player.durationChanged.connect(self._audio_duration)
        self.player.mediaStatusChanged.connect(self._media_status)
        self.player.errorOccurred.connect(lambda error, message: self.statusBar().showMessage('音频播放：' + message, 8000))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._build_actions()
        self._build_workspace()
        self._apply_performance()
        self._refresh_actions()

    def _action(self, text, callback, shortcut=None, checkable=False):
        action = QAction(text, self)
        action.setCheckable(checkable)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        return action

    def _build_actions(self):
        self.open_chart_action = self._action('打开谱面', self.open_chart, 'Ctrl+O')
        self.open_project_action = self._action('打开工程', self.open_project, 'Ctrl+Shift+O')
        self.generate_action = self._action('生成指法初稿', self.generate)
        self.save_action = self._action('保存工程', self.save_project, 'Ctrl+S')
        self.save_as_action = self._action('另存为…', lambda: self.save_project(save_as=True), 'Ctrl+Shift+S')
        self.demo_action = self._action('打开示例', self.open_demo)
        self.undo_action = self._action('撤销', self.undo, QKeySequence.Undo)
        self.redo_action = self._action('重做', self.redo, QKeySequence.Redo)
        self.redo_action.setShortcuts([QKeySequence('Ctrl+Y'), QKeySequence('Ctrl+Shift+Z')])
        for action in (self.open_chart_action, self.open_project_action, self.save_action,
                       self.save_as_action, self.undo_action, self.redo_action):
            shortcuts = ' / '.join(key.toString(QKeySequence.NativeText) for key in action.shortcuts())
            action.setIconText(f'{action.text()} ({shortcuts})')
            action.setText(f'{action.text()}\t{shortcuts}')
        self.validate_action = self._action('检查方案', self.validate_plan)
        self.export_action = self._action('导出动作数据', self.export)
        self.preview_action = self._action('生成片段视频', lambda: self.render(False))
        self.render_action = self._action('生成整曲视频', lambda: self.render(True))
        self.replan_action = self._action('重新分配手指', self.replan)
        self.batch_path_action = self._action('批量修改轨迹…', self.batch_paths)
        self.flick_action = self._action('Flick 滑动设置…', self.edit_flicks)
        self.release_action = self._action('修改抬起时间…', self.edit_release)
        self.split_heads_action = self._action('拆开合并的音符头', self.split_heads)
        self.touch_actions = [self._action(label, lambda checked=False, m=mode: self.edit_touch_mode(m)) for mode, label in
            (('auto', '自动接续'), ('continue', '继续按住'), ('press', '重新按下'))]
        self.performance_action = self._action('性能设置', self.configure_performance)
        self.scale_action = self._action('界面缩放…', self.configure_interface_scale)
        self.scale_action.setToolTip('设置整个界面的显示比例，支持 50%–200%，重启后生效')
        self.compare_action = self._action('显示初稿轨迹', self.compare_baseline, checkable=True)
        self.audio_action = self._action('关联音乐…', self.choose_audio)
        self.background_action = self._action('关联曲绘…', self.choose_background)
        self.background_action.setToolTip('为当前谱面选择曲绘，随工程保存，并用于谱面视频和模糊背景')
        self.attach_chart_action = self._action('关联原谱面…', self.attach_chart)
        for action, tip in (
            (self.open_chart_action, '打开谱面文件，开始设计这首曲子的指法'),
            (self.open_project_action, '继续编辑已保存的 .fingering.json 工程，也可导入 motion-plan.json 动作数据'),
            (self.generate_action, '为当前谱面生成可编辑的指法初稿，之后可分别调整指法、轨迹和手部姿态重新生成会替换指法及轨迹修改，可撤销；本曲的姿态设置会保留'),
            (self.save_action, '保存指法、轨迹和手部姿态设置，下次打开工程可继续编辑'),
            (self.validate_action, '检查音符覆盖、手指占用、触控轨迹、遮挡和 Hold 连续性手部姿态与皮肤接触效果请通过片段视频查看'),
            (self.export_action, '检查后导出触控序列、含姿态设置的动作数据和可继续编辑的工程生成视频请使用“生成片段视频”或“生成整曲视频”'),
            (self.preview_action, '仅检查指定片段并生成视频；跨越片段边界的 Hold 会保留片段内的持续按住状态'),
            (self.render_action, '使用当前指法、轨迹和手部姿态，生成整首曲目的手部动画视频'),
            (self.replan_action, '自动重新分配所选动作的手与手指，跳过已锁定的指法；轨迹和时间保持'),
            (self.batch_path_action, '同时修改多个动作的轨迹，或一条轨迹中的多个控制点'),
            (self.flick_action, '设置 Flick 的滑动方向、总距离、滑动时长及滑完后的停留时间'),
            (self.release_action, '设置手指何时离开屏幕；可批量提前、延后或修改按住时长'),
            (self.split_heads_action, '把同一动作中合并的多个 Tap / Hold 头拆开，保留现有轨迹并优先分配空闲手指；若没有全程空闲的手指，会在“问题”页标出重叠，供继续调整可撤销'),
            (self.compare_action, '开启后，画布显示工程最初的轨迹；关闭后显示当前修改切换显示不会撤销修改'),
            (self.performance_action, '调整编辑预览画质、自动分配的计算量和撤销记录占用'),
        ):
            action.setToolTip(tip)
        file_menu = self.menuBar().addMenu('文件')
        for action in (self.open_chart_action, self.open_project_action, self.generate_action, self.demo_action,
                       self.save_action, self.save_as_action, self.export_action, self.attach_chart_action, self.audio_action,
                       self.background_action):
            file_menu.addAction(action)
        edit_menu = self.menuBar().addMenu('编辑')
        for action in (self.undo_action, self.redo_action, self.batch_path_action, self.flick_action,
                       self.release_action, *self.touch_actions, self.split_heads_action, self.replan_action, self.validate_action):
            edit_menu.addAction(action)
        view_menu = self.menuBar().addMenu('视图与渲染')
        for action in (self.compare_action, self.preview_action, self.render_action, self.scale_action, self.performance_action):
            view_menu.addAction(action)
        toolbar = QToolBar('常用操作', self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for action in (self.open_chart_action, self.generate_action, self.open_project_action, self.save_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        for action in (self.undo_action, self.redo_action, self.validate_action, self.export_action, self.preview_action,
                       self.scale_action, self.performance_action):
            toolbar.addAction(action)

    def _build_workspace(self):
        workspace = QWidget()
        workspace.setObjectName('workspace')
        self.setCentralWidget(workspace)
        outer = QVBoxLayout(workspace)
        outer.setContentsMargins(12, 8, 12, 8)
        self.summary = QLabel('打开谱面 → 生成指法初稿 → 调整指法与姿态 → 保存工程或生成视频')
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet('font-size: 14px; padding: 6px 2px;')
        top = QHBoxLayout()
        top.addWidget(self.summary, 1)
        demo = QPushButton('打开示例')
        demo.setToolTip('打开自带的简单谱面并生成指法，熟悉编辑操作')
        demo.clicked.connect(self.open_demo)
        top.addWidget(demo)
        outer.addLayout(top)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split, 1)
        left = QSplitter(Qt.Vertical)
        left.setChildrenCollapsible(False)
        left.setHandleWidth(7)
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 7, 0)
        self.canvas = ChartCanvas()
        self.canvas.setMinimumHeight(180)
        self.canvas.selected.connect(self._set_selection)
        self.canvas.point_selected.connect(self._point_selected)
        self.canvas.point_edited.connect(self._point_edited)
        self.canvas.add_point.connect(self._add_point_at)
        self.canvas.pause_requested.connect(self.pause)
        self.view_tabs = QTabWidget()
        self.view_tabs.addTab(self.canvas, '谱面与触控轨迹')
        self.view_tabs.setTabToolTip(0, '查看音符与触点，修改使用的手指、触控轨迹和按下 / 抬起时间')
        pose_view = QWidget()
        pose_layout = QVBoxLayout(pose_view)
        pose_layout.setContentsMargins(0, 0, 0, 0)
        pose_tools = QHBoxLayout()
        self.pose_tool = QComboBox()
        for label, mode in (('选择手腕或手指', 'select'), ('移动手腕', 'move'),
                            ('转动手腕', 'yaw'), ('调整手腕高度', 'height')):
            self.pose_tool.addItem(label, mode)
        self.pose_tool.setToolTip('选择工具后，拖动预览中的手腕圆点前后倾斜和侧倾可在右侧参数中调整')
        pose_tools.addWidget(self.pose_tool)
        pose_tools.addStretch(1)
        pose_tools.addWidget(QLabel('观察角度'))
        self.pose_view = QComboBox()
        for label, view in (('斜视', 'angle'), ('俯视', 'top'), ('侧视', 'side')):
            self.pose_view.addItem(label, view)
        pose_tools.addWidget(self.pose_view)
        pose_layout.addLayout(pose_tools)
        self.pose_canvas = PoseCanvas()
        self.pose_canvas.pause_requested.connect(self.pause)
        self.pose_canvas.target_selected.connect(self._select_pose_target)
        self.pose_tool.currentIndexChanged.connect(lambda: self._pose_tool_changed())
        self.pose_view.currentIndexChanged.connect(lambda: self.pose_canvas.set_view(self.pose_view.currentData()))
        pose_layout.addWidget(self.pose_canvas, 1)
        self.view_tabs.addTab(pose_view, '手部姿态')
        self.view_tabs.setTabToolTip(1, '调整手腕、空闲手指和抬落动作此处显示简化关节，真实手模效果可生成片段视频查看')
        preview_layout.addWidget(self.view_tabs, 1)
        transport = QHBoxLayout()
        self.play_button = QPushButton('播放 (Space)')
        self.play_button.setToolTip('播放或暂停预览，也可按空格键')
        self.play_button.clicked.connect(self.toggle_play)
        transport.addWidget(self.play_button)
        back = QPushButton('−0.1 s')
        back.setToolTip('将播放位置向前移动 0.1 秒')
        back.clicked.connect(lambda: self.seek(self.music_time - .1))
        transport.addWidget(back)
        forward = QPushButton('+0.1 s')
        forward.setToolTip('将播放位置向后移动 0.1 秒')
        forward.clicked.connect(lambda: self.seek(self.music_time + .1))
        transport.addWidget(forward)
        self.speed = QComboBox()
        for number in (.1, .25, .5, .75, 1., 1.5, 2.):
            self.speed.addItem(f'{number:g}×', number)
        self.speed.setCurrentIndex(4)
        self.speed.setToolTip('调整编辑器的预览播放速度')
        self.speed.currentIndexChanged.connect(self._speed_changed)
        transport.addWidget(self.speed)
        self.clock_label = QLabel('00:00.000')
        self.clock_label.setToolTip('当前音乐播放位置，格式为分:秒.毫秒')
        self.clock_label.setMinimumWidth(91)
        transport.addWidget(self.clock_label)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderPressed.connect(lambda: self.pause())
        self.seek_slider.valueChanged.connect(lambda value: self.seek(value / 1000))
        transport.addWidget(self.seek_slider, 1)
        compare = QPushButton('显示初稿轨迹')
        compare.setToolTip(self.compare_action.toolTip())
        compare.setCheckable(True)
        compare.clicked.connect(self.compare_action.setChecked)
        compare.clicked.connect(self.compare_baseline)
        self.compare_action.toggled.connect(compare.setChecked)
        transport.addWidget(compare)
        preview_layout.addLayout(transport)
        left.addWidget(preview_panel)
        track_panel = QWidget()
        track_layout = QVBoxLayout(track_panel)
        track_layout.setContentsMargins(0, 0, 7, 0)
        self.timeline = FingeringTimeline()
        self.timeline.setToolTip('同一手指的重叠动作上下分层指向动作可查看编号和起止时间；在左侧手指名称上滚动可上下查看轨道')
        self.timeline.selected.connect(self._set_selection)
        self.timeline.seek.connect(lambda t: self.seek(t + self.offset))
        self.timeline.range_selected.connect(self._timeline_range)
        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.timeline_scroll.setMinimumHeight(150)
        self.timeline_scroll.setWidget(self.timeline)
        self.track_tabs = QTabWidget()
        self.track_tabs.addTab(self.timeline_scroll, '指法轨道')
        self.pose_timeline = PoseTimeline()
        self.pose_timeline.seek.connect(lambda t: self.seek(t + self.offset))
        self.pose_timeline.target_selected.connect(self._select_pose_target)
        self.pose_timeline.range_selected.connect(self._timeline_range)
        pose_scroll = QScrollArea()
        pose_scroll.setWidgetResizable(True)
        pose_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        pose_scroll.setWidget(self.pose_timeline)
        self.track_tabs.addTab(pose_scroll, '姿态轨道')
        self.track_tabs.setMinimumHeight(174)
        track_layout.addWidget(self.track_tabs, 1)
        range_bar = QHBoxLayout()
        range_bar.addWidget(QLabel('片段起止'))
        self.range_start, self.range_end = QDoubleSpinBox(), QDoubleSpinBox()
        for field, value in ((self.range_start, 0.), (self.range_end, 5.)):
            field.setDecimals(3)
            field.setRange(0, 86400)
            field.setValue(value)
            field.setSuffix(' s')
            range_bar.addWidget(field)
        self.range_start.setToolTip('音乐播放到第几秒时开始此范围用于循环播放、选择动作、新建姿态片段，以及片段视频的默认范围')
        self.range_end.setToolTip('音乐播放到第几秒时结束，须晚于开始时间此范围也可在时间轴空白处拖动选择')
        self.loop = QCheckBox('循环播放')
        self.loop.setToolTip('在“片段起止”设定的范围内循环播放')
        range_bar.addWidget(self.loop)
        pick_range = QPushButton('选中片段动作')
        pick_range.setToolTip('选中与上述时间段重叠的动作，方便批量改手指或轨迹')
        pick_range.clicked.connect(self.select_range)
        range_bar.addWidget(pick_range)
        range_bar.addStretch(1)
        range_bar.addWidget(QLabel('拖动框选 · Ctrl＋滚轮缩放'))
        track_layout.addLayout(range_bar)
        left.addWidget(track_panel)
        left.setStretchFactor(0, 1)
        left.setStretchFactor(1, 1)
        left.setSizes([440, 240])
        split.addWidget(left)
        right = QWidget()
        right.setMinimumWidth(320)
        right.setMaximumWidth(490)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(7, 0, 0, 0)
        self.inspector_split = QSplitter(Qt.Vertical)
        self.inspector_split.setChildrenCollapsible(False)
        self.inspector_split.setHandleWidth(7)
        right_layout.addWidget(self.inspector_split)
        assignment_panel = QWidget()
        assignment_layout = QVBoxLayout(assignment_panel)
        assignment_layout.setContentsMargins(4, 4, 4, 4)
        self.selection_label = QLabel('在动作列表或时间轴中选择动作，开始编辑指法')
        self.selection_label.setWordWrap(True)
        self.selection_label.setToolTip('每行是一段可编辑动作，可能关联多个音符接续的多段动作会合成一次完整的按下到抬起；指法决定手指，轨迹决定贴屏位置')
        assignment_layout.addWidget(self.selection_label)
        form = QFormLayout()
        self.hand = QComboBox()
        self.hand.addItem('左手', 'left')
        self.hand.addItem('右手', 'right')
        self.finger = QComboBox()
        for key, name in FINGER_NAMES.items():
            self.finger.addItem(name, key)
        form.addRow('手', self.hand)
        form.addRow('手指', self.finger)
        assignment_layout.addLayout(form)
        self.edit_buttons = []
        grid = QGridLayout()
        for n, (text, callback, tip) in enumerate([
            ('应用手指并锁定', self.assign, '把所选动作交给上方选定的手与手指，并锁定这份分配'),
            ('左右手交替', self.alternate, '按动作开始时间左右交替，使用上方选定的手指；从上方选定的手开始应用后锁定指法'),
            ('互换左右手', self.swap, '所选动作的左手改右手、右手改左手，手指名称保持；应用后锁定指法'),
            ('锁定指法', lambda: self.set_lock(True), '自动重新分配时跳过这些动作；仍可手动修改'),
            ('解锁指法', lambda: self.set_lock(False), '允许“重新分配手指”调整这些动作'),
            ('恢复所选初稿', self.reset_selected, '恢复所选动作最初的手指、轨迹及按下和抬起时间，并解除锁定可用 Ctrl+Z 撤销')]):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(callback)
            grid.addWidget(button, n // 2, n % 2)
            self.edit_buttons.append(button)
        assignment_layout.addLayout(grid)
        self.replan_button = QPushButton('重新分配手指')
        self.replan_button.setToolTip(self.replan_action.toolTip())
        self.replan_button.clicked.connect(self.replan)
        assignment_layout.addWidget(self.replan_button)
        self.assignment_scroll = QScrollArea()
        self.assignment_scroll.setWidgetResizable(True)
        self.assignment_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.assignment_scroll.setMinimumHeight(85)
        self.assignment_scroll.setWidget(assignment_panel)
        self.inspector_split.addWidget(self.assignment_scroll)
        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(330)
        self.inspector_split.addWidget(self.tabs)
        self.inspector_split.setStretchFactor(0, 0)
        self.inspector_split.setStretchFactor(1, 1)
        self.inspector_split.setSizes([180, 550])
        self.table_model = ContactTableModel(self)
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.verticalHeader().setVisible(True)
        self.table.verticalHeader().setMinimumWidth(42)
        self.table.horizontalHeader().setStretchLastSection(True)
        for col, width in enumerate((72, 72, 45, 65, 55)):
            self.table.setColumnWidth(col, width)
        self.table.selectionModel().selectionChanged.connect(self._table_selection)
        self.table.doubleClicked.connect(lambda index: self.seek(self.document.contact(index.row())['start'] + self.offset))
        self.tabs.addTab(self.table, '动作')
        self.tabs.setTabToolTip(0, '每行是一段可编辑动作；接续的多段动作共用一次按下与抬起Ctrl / Shift 多选，双击跳转到动作开始')
        paths = QWidget()
        paths_layout = QVBoxLayout(paths)
        paths_layout.setContentsMargins(0, 0, 0, 0)
        self.path_split = QSplitter(Qt.Vertical)
        self.path_split.setChildrenCollapsible(False)
        self.path_split.setHandleWidth(7)
        paths_layout.addWidget(self.path_split)
        path_controls = QWidget()
        path_layout = QVBoxLayout(path_controls)
        path_layout.setContentsMargins(6, 6, 6, 6)
        description = QLabel('控制点指定手指在某个时刻的位置选一段动作可编辑下表；Ctrl / Shift 多选点可一起修改')
        description.setWordWrap(True)
        path_layout.addWidget(description)
        self.path_summary = QLabel('选择一个动作可编辑控制点；选择多个动作可批量修改轨迹')
        self.path_summary.setWordWrap(True)
        self.path_summary.setWordWrap(True)
        path_layout.addWidget(self.path_summary)
        self.batch_path_button = QPushButton('批量修改轨迹…')
        self.batch_path_button.setToolTip('先在“动作”页或时间轴中选择一个或多个动作，再统一平移、固定落点或减少控制点')
        self.batch_path_button.clicked.connect(self.batch_paths)
        path_layout.addWidget(self.batch_path_button)
        gesture_tools = QHBoxLayout()
        self.flick_button = QPushButton('Flick 滑动设置…')
        self.flick_button.setToolTip(self.flick_action.toolTip())
        self.flick_button.clicked.connect(self.edit_flicks)
        self.release_button = QPushButton('修改抬起时间…')
        self.release_button.setToolTip(self.release_action.toolTip())
        self.release_button.clicked.connect(self.edit_release)
        gesture_tools.addWidget(self.flick_button)
        gesture_tools.addWidget(self.release_button)
        path_layout.addLayout(gesture_tools)
        touch_tools = QHBoxLayout()
        self.touch_mode = QComboBox()
        for name, mode in (('自动接续', 'auto'), ('继续按住', 'continue'), ('重新按下', 'press')):
            self.touch_mode.addItem(name, mode)
        self.touch_mode.setToolTip('自动接续：同一手指前后相接的 Flick / Drag 自动连成一次按住\n继续按住：连接前一段动作，中间不抬起\n重新按下：在本段开始时单独按下\n每个新 Tap 或 Hold 起点都需要单独按下')
        self.touch_button = QPushButton('应用接续方式')
        self.touch_button.clicked.connect(lambda: self.edit_touch_mode(self.touch_mode.currentData()))
        touch_tools.addWidget(self.touch_mode)
        touch_tools.addWidget(self.touch_button)
        path_layout.addLayout(touch_tools)
        self.split_heads_button = QPushButton('拆开合并的音符头')
        self.split_heads_button.setToolTip(self.split_heads_action.toolTip())
        self.split_heads_button.clicked.connect(self.split_heads)
        path_layout.addWidget(self.split_heads_button)
        self.drag_path = QCheckBox('拖动一点，移动整条轨迹')
        self.drag_path.setToolTip('开启：拖动画布中的一点，整条轨迹一起移动关闭：只移动该点；若它属于多选范围，则一起移动选中的点')
        self.drag_path.toggled.connect(self._drag_path_mode)
        path_layout.addWidget(self.drag_path)
        self.path_tools_scroll = QScrollArea()
        self.path_tools_scroll.setWidgetResizable(True)
        self.path_tools_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.path_tools_scroll.setMinimumHeight(90)
        self.path_tools_scroll.setWidget(path_controls)
        self.path_split.addWidget(self.path_tools_scroll)
        point_panel = QWidget()
        point_panel.setMinimumHeight(200)
        point_layout = QVBoxLayout(point_panel)
        point_layout.setContentsMargins(6, 0, 6, 6)
        self.path_model = PathTableModel(self)
        self.path_model.edited.connect(self._path_edited)
        self.path_model.invalid.connect(self._error)
        self.path_table = QTableView()
        self.path_table.setModel(self.path_model)
        self.path_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.path_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.path_table.selectionModel().selectionChanged.connect(self._path_selection_changed)
        self.path_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.path_table.clicked.connect(lambda index: self._path_row_selected(index.row()))
        self.path_table.setMinimumHeight(115)
        point_layout.addWidget(self.path_table, 1)
        tools = QHBoxLayout()
        self.add_path_button = QPushButton('播放位置加点')
        self.add_path_button.setToolTip('在当前播放时刻、轨迹经过的位置添加控制点；先将播放位置移到所选动作内部')
        self.add_path_button.clicked.connect(self.add_path_point)
        self.remove_path_button = QPushButton('删除所选点')
        self.remove_path_button.setToolTip('删除表格中选中的控制点，自动保留第一个点；删除后相邻点直接相连')
        self.remove_path_button.clicked.connect(self.remove_path_point)
        tools.addWidget(self.add_path_button)
        tools.addWidget(self.remove_path_button)
        point_layout.addLayout(tools)
        self.zone_button = QPushButton('显示当前点的可用区域')
        self.zone_button.setToolTip('绿色区域表示当前控制点时刻可放置手指的判定区域，已扣除不可触碰的遮挡区域；其他时刻的区域可能不同')
        self.zone_button.clicked.connect(self.show_zone)
        point_layout.addWidget(self.zone_button)
        self.path_split.addWidget(point_panel)
        self.path_split.setStretchFactor(0, 0)
        self.path_split.setStretchFactor(1, 1)
        self.path_split.setSizes([175, 320])
        self.tabs.addTab(paths, '轨迹')
        self.problem_list = QTreeWidget()
        self.problem_list.setHeaderHidden(True)
        self.problem_list.setColumnCount(1)
        self.problem_list.header().setSectionResizeMode(QHeaderView.Stretch)
        self.problem_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.problem_list.setWordWrap(True)
        self.problem_list.setIndentation(16)
        self.problem_list.itemClicked.connect(self._problem_selected)
        self.tabs.addTab(self.problem_list, '问题')
        self.tabs.setTabToolTip(2, '点击问题选中关联动作；展开后点击子项可单独选择其中一个需修复项会阻止导出')
        self.problem_list.setToolTip('点击问题定位并放大时间轴，点击下方动作可分别编辑')
        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.setMinimumWidth(340)
        self.inspector_tabs.setMaximumWidth(510)
        self.inspector_tabs.addTab(right, '指法与轨迹')
        self.pose_panel = PosePanel(lambda: (self.range_start.value() - self.offset, self.range_end.value() - self.offset))
        self.pose_panel.change.connect(self._edit_pose)
        self.pose_panel.preview.connect(self.pose_canvas.set_pose)
        self.pose_panel.seek.connect(self._seek_pose)
        self.pose_panel.error.connect(self._error)
        self.pose_panel.target_selected.connect(self._pose_target_changed)
        self.pose_panel.clip_selected.connect(self.pose_timeline.select_clip)
        self.pose_canvas.drag_changed.connect(self.pose_panel.drag)
        self.pose_canvas.feedback.connect(self.pose_panel.feedback.setText)
        self.pose_timeline.clip_selected.connect(self.pose_panel.select_clip)
        self.inspector_tabs.addTab(self.pose_panel, '手部姿态')
        self.inspector_tabs.setTabToolTip(0, '选择动作后，修改使用的手指、触控轨迹和接触时间')
        self.inspector_tabs.setTabToolTip(1, '选择手腕或手指后，调整整曲默认姿态，或用片段关键帧编排动作')
        self.view_tabs.currentChanged.connect(self._pose_tab_changed)
        self.inspector_tabs.currentChanged.connect(self._pose_tab_changed)
        split.addWidget(self.inspector_tabs)
        split.setSizes([920, 380])
        self.task_status = TaskStatus()
        outer.addWidget(self.task_status)
        self.progress_label = QLabel('就绪')
        self.progress_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.progress_label.setFixedHeight(self.progress_label.fontMetrics().lineSpacing())
        self.statusBar().addWidget(self.progress_label, 1)
        self.cancel_button = self.task_status.cancel_button
        self.cancel_button.clicked.connect(self.jobs.cancel)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(20000)
        log_body = QWidget()
        log_layout = QVBoxLayout(log_body)
        log_layout.setContentsMargins(6, 4, 6, 4)
        controls = QHBoxLayout()
        controls.addWidget(QLabel('实时输出 · 完整记录同时写入日志文件'), 1)
        self.open_log_button = QPushButton('打开完整日志')
        self.open_log_button.setEnabled(False)
        self.open_log_button.clicked.connect(self.open_job_log)
        controls.addWidget(self.open_log_button)
        copy_log = QPushButton('复制显示内容')
        copy_log.clicked.connect(lambda: QApplication.clipboard().setText(self.log_view.toPlainText()))
        controls.addWidget(copy_log)
        log_layout.addLayout(controls)
        log_layout.addWidget(self.log_view)
        self.log_dock = QDockWidget('后台任务日志', self)
        self.log_dock.setWidget(log_body)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()
        self.task_status.log_button.clicked.connect(self.show_job_log)
        window_menu = self.menuBar().addMenu('窗口')
        window_menu.addAction(self.log_dock.toggleViewAction())
        window_menu.addAction(self._action('显示任务进度', self.task_status.show))
        app_log_action = self._action('打开程序日志', self.open_app_log)
        app_log_action.setToolTip(str(APP_LOG))
        window_menu.addAction(app_log_action)
        self.play_shortcut = QAction(self)
        self.play_shortcut.setShortcut('Space')
        self.play_shortcut.triggered.connect(self.toggle_play)
        self.addAction(self.play_shortcut)

    @property
    def offset(self):
        return self.source.chart.offset if self.source else (self.document.chart.get('offset', 0.) if self.document else 0.)

    def _asset_defaults(self):
        result = {key: self.settings.value(key, '', type=str) for key in ('model', 'resources', 'blender')}
        result['background'] = self.source.reference.get('background', '') if self.source else ''
        for key, path in (('model', ASSETS / 'Hands + armature.blend'), ('resources', ASSETS / 'resources.zip')):
            if not result[key] and path.exists():
                result[key] = str(path)
        return result

    def _async(self, function, success):
        key = uuid.uuid4().hex
        thread = AsyncCall(key, function, self)
        self._io[key] = thread
        self._io_handlers[key] = success
        thread.succeeded.connect(self._io_succeeded)
        thread.failed.connect(self._io_failed)
        thread.finished.connect(lambda: self._io_finished(key))
        thread.start()
        self._refresh_actions()

    def _load_source(self, path, success, **options):
        def read():
            try:
                return load_source(path, **options)
            except ChartSelectionRequired as choice:
                return choice

        def opened(result):
            if self._pending_close:
                return
            if isinstance(result, ChartSelectionRequired):
                entry, accepted = QInputDialog.getItem(self, '选择谱面',
                    f'{Path(path).name}\n选择要打开的谱面', result.entries, 0, False)
                if accepted:
                    self._load_source(path, success, **(options | dict(chart_entry=entry)))
                else:
                    self._progress('已取消选择谱面')
            else:
                success(result)

        self._async(read, opened)

    @Slot(object)
    def _io_succeeded(self, message):
        handler = self._io_handlers.pop(message['key'], None)
        if handler:
            try:
                handler(message['value'])
            except Exception as exc:
                logging.getLogger(__name__).exception('处理后台操作结果失败')
                self._error(str(exc))

    @Slot(object)
    def _io_failed(self, message):
        self._io_handlers.pop(message['key'], None)
        self._error(message['error'])

    def _io_finished(self, key):
        thread = self._io.pop(key, None)
        if thread:
            thread.deleteLater()
        self._refresh_actions()
        if self._pending_close and not self._io and not self.jobs.running:
            QTimer.singleShot(0, self.close)

    def _guard_replace(self, callback):
        if self.document and self.document.dirty:
            answer = QMessageBox.warning(self, '保存当前工程', '当前工程有未保存的修改',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.Save:
                self.save_project(on_saved=lambda: self._guard_replace(callback))
                return
        callback()

    def open_chart(self, path=None):
        if isinstance(path, bool):
            path = None
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, '打开谱面', self.settings.value('last_directory', str(ROOT)),
                                                 '谱面 (*.zip *.pez *.json *.pec);;所有文件 (*)')
        if not path:
            return
        def start():
            self.pause()
            self._progress('正在读取谱面…')
            self._load_source(path, self._source_opened)
        self._guard_replace(start)

    def _source_opened(self, source):
        self.document = None
        self._set_source(source)
        self._set_document(None)
        self.settings.setValue('last_directory', str(Path(source.reference['path']).parent))
        self._progress('谱面已打开点击“生成指法初稿”开始，或通过“打开工程”继续已有方案')

    def _set_source(self, source):
        self.pause()
        self.source = source
        self.canvas.set_source(source)
        audio = source.reference.get('audio') if source else None
        self.player.setSource(QUrl.fromLocalFile(audio) if audio else QUrl())
        self._update_duration()
        self.seek(0.)
        self._refresh_actions()

    def _update_duration(self):
        self.duration = max(1., self.source.duration if self.source else 0.,
                            self.document.duration + self.offset + 1. if self.document else 0.,
                            self.player.duration() / 1000 if self.source else 0.)
        self.seek_slider.setMaximum(round(self.duration * 1000))

    def open_project(self, path=None):
        if isinstance(path, bool):
            path = None
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, '打开工程或导入动作数据', self.settings.value('last_directory', str(ROOT)),
                                                 '指法工程 / 动作数据 (*.fingering.json *.json)')
        if not path:
            return
        def start():
            self.pause()
            self._progress('正在读取工程或动作数据…')
            self._async(lambda: read_document(path), self._project_opened)
        self._guard_replace(start)

    def _project_opened(self, document):
        logging.getLogger(__name__).info('打开工程：%s', document.path)
        if document.chart.get('path') and Path(document.chart['path']).is_file():
            self.document = document
            self._set_source(None)
            self._set_document(document)
            self._load_source(document.chart['path'], lambda source: self._attach_project(document, source),
                              expected=document.chart, audio=document.chart.get('audio'),
                              background=document.chart.get('background'))
        elif not document.chart.get('sha256') and self.source:
            self._attach_project(document, self.source)
        else:
            self.document = document
            self._set_source(None)
            self._set_document(document)
            path, _ = QFileDialog.getOpenFileName(self, '选择此指法对应的原谱面', str(ROOT), '谱面 (*.zip *.pez *.json *.pec)')
            if path:
                self._load_source(path, lambda source: self._attach_project(document, source), expected=document.chart)
            else:
                self._progress('方案已打开通过“文件 → 关联原谱面”添加谱面后，即可检查方案、播放预览和生成视频')

    def attach_chart(self):
        if not self.document:
            return
        document = self.document
        path, _ = QFileDialog.getOpenFileName(self, '关联此工程的原谱面', str(ROOT), '谱面 (*.zip *.pez *.json *.pec)')
        if path:
            self._load_source(path, lambda source: self._attach_project(document, source, undoable=True),
                              expected=document.chart, audio=document.chart.get('audio'),
                              background=document.chart.get('background'))

    def _attach_project(self, document, source, *, undoable=False):
        expected = set(range(source.reference['note_count']))
        if set(document.note_contacts) != expected:
            raise ValueError('此方案的音符编号与当前谱面不匹配，请选择正确的谱面')
        if undoable:
            document.commit_chart('关联原谱面', source.reference)
        else:
            document.chart = copy.deepcopy(source.reference)
        self.document = document
        self._set_source(source)
        self._set_document(document)
        self._progress('方案已打开双击动作跳转到按下时刻；手动分配手指后会锁定，自动分配时跳过')

    def _set_document(self, document):
        self.document = document
        self.selected_ids = set()
        if document:
            document.history_limit = self.performance['history_mib'] * 1024 * 1024
        self.canvas.set_document(document)
        self.timeline.set_document(document, self.offset)
        self._refresh_pose_editor()
        self.table_model.set_document(document, self.offset)
        self._table_contact_count = len(document) if document else 0
        self.path_model.set_points([], self.offset)
        self._path_selection_changed()
        self.finger.clear()
        for finger in document.allowed_fingers if document else FINGER_NAMES:
            self.finger.addItem(FINGER_NAMES[finger], finger)
        self._show_problems(edit_check(document, self.source.chart if self.source else None) + draft_warnings(document) if document else [])
        self._update_duration()
        self._refresh_actions()
        self._update_title()

    def open_demo(self):
        def start():
            self._progress('准备示例谱面…')
            self._async(lambda: load_source(create_demo()), self._demo_ready)
        if not self.jobs.running and not self._io:
            self._guard_replace(start)

    def _demo_ready(self, source):
        self._source_opened(source)
        self.jobs.start('generate', chart=source.reference['path'], chart_reference=dict(source.reference),
                        options=dict(settings=dict(fingers=['index', 'middle'], beam_width=4)))

    def generate(self):
        if not self.source:
            return
        dialog = GenerateDialog(self._asset_defaults(), self)
        if dialog.exec() != QDialog.Accepted:
            return
        options = dialog.values()
        if self.document:
            options['pose'] = copy.deepcopy(self.document.pose)
        for key in ('model', 'blender'):
            self.settings.setValue(key, options[key])
        self._guard_replace(lambda: self.jobs.start('generate', chart=self.source.reference['path'],
                            chart_reference=dict(self.source.reference), options=options))

    def save_project(self, checked=False, *, save_as=False, on_saved=None):
        if not self.document:
            return
        document = self.document
        path = document.path if not save_as else None
        if not path:
            PROJECTS.mkdir(parents=True, exist_ok=True)
            default = PROJECTS / (document.title + '.fingering.json')
            name, _ = QFileDialog.getSaveFileName(self, '保存指法工程', str(default), '指法工程 (*.fingering.json)')
            if not name:
                return
            path = Path(name)
            if not path.name.endswith('.fingering.json'):
                path = path.with_name(path.name + '.fingering.json')
        snapshot = document.snapshot(destination=path)
        state, chart, title = document._state, copy.deepcopy(document.chart), document.title
        def saved(_):
            document.mark_saved(path, state=state, chart=chart, title=title)
            logging.getLogger(__name__).info('保存工程：%s', path)
            self._update_title()
            self._progress('工程已保存：' + str(path))
            if on_saved:
                on_saved()
        self._async(lambda: write_json_atomic(path, snapshot), saved)

    def _set_selection(self, ids, additive=False):
        if not self.document:
            return
        ids = set(ids)
        old = self.selected_ids
        self.selected_ids = (self.selected_ids | ids) if additive else ids
        self.canvas.selected_ids = set(self.selected_ids)
        self.timeline.selected_ids = set(self.selected_ids)
        if old != self.selected_ids:
            self.canvas.zone = None
            self.canvas.active_point = 0
        self._sync_selection = True
        selection = QItemSelection()
        ordered = sorted(self.selected_ids)
        if ordered:
            low = previous = ordered[0]
            for row in ordered[1:] + [None]:
                if row is not None and row == previous + 1:
                    previous = row
                    continue
                selection.select(self.table_model.index(low, 0), self.table_model.index(previous, 4))
                if row is not None:
                    low = previous = row
        self.table.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        self._sync_selection = False
        if ordered:
            self.table.scrollTo(self.table_model.index(ordered[0], 0))
        self._refresh_inspector()
        self.canvas.update()
        self.timeline.update()
        self._refresh_actions()

    def _table_selection(self, selected, deselected):
        if self._sync_selection or not self.document:
            return
        ids = {index.row() for index in self.table.selectionModel().selectedRows()}
        self._set_selection(ids)

    def _refresh_inspector(self):
        if not self.document or not self.selected_ids:
            self.selection_label.setText('在动作列表或时间轴中选择动作，开始编辑指法' if self.document else
                                         '生成指法初稿或打开工程后，即可编辑动作')
            self.path_model.set_points([], self.offset)
            self._path_selection_changed()
            return
        ids = self.document.sorted_ids(self.selected_ids)
        c = self.document.contact(ids[0])
        count = len({n for i in ids for n in self.document.contact(i)['note_ids']})
        detail = (f"动作 #{ids[0] + 1} · {finger_name(c)} · {c['start'] + self.offset:.3f}–{c['end'] + self.offset:.3f} s"
                  if len(ids) == 1 else '、'.join(f'#{i + 1} {finger_name(self.document.contact(i))}' for i in ids[:6]))
        if len(ids) > 6:
            detail += f'…（共 {len(ids)} 段）'
        if len(ids) == 1 and len(c['note_ids']) > 1:
            detail += '\n这段动作关联多个音符，修改时请一起检查'
        self.selection_label.setText(f'已选 {len(ids)} 段动作 / {count} 个音符\n' + detail)
        self.hand.setCurrentIndex(self.hand.findData(c['hand']))
        self.finger.setCurrentIndex(self.finger.findData(c['finger']))
        self.touch_mode.setCurrentIndex(self.touch_mode.findData(c.get('touch_mode', 'auto')))
        self.path_model.set_points(c['points'] if len(ids) == 1 else [], self.offset)
        self.canvas.active_point = min(self.canvas.active_point, len(c['points']) - 1) if len(ids) == 1 else 0
        self._path_selection_changed()

    def _edit(self, operation):
        if not self.document or not self.selected_ids:
            return
        if self._io or (self.jobs.running and self.jobs.request['kind'] == 'generate'):
            return
        self.pause()
        try:
            changed = operation()
        except (ValueError, IndexError) as exc:
            self._error(str(exc))
            return
        self._changed(changed)

    def _pose_tab_changed(self, index):
        self.view_tabs.setCurrentIndex(index)
        self.inspector_tabs.setCurrentIndex(index)
        self.track_tabs.setCurrentIndex(index)
        if index == 1:
            self.pose_canvas.set_time(self.music_time - self.offset)

    def _pose_tool_changed(self):
        self.pose_canvas.mode = self.pose_tool.currentData()
        if self.pose_canvas.mode != 'select':
            self._select_pose_target(self.pose_panel.selection[0], 'hand')
        self.pose_canvas.update()

    def _select_pose_target(self, hand, target):
        self.pose_panel.select_target(hand, target)
        self._pose_target_changed(hand, target)
        self.inspector_tabs.setCurrentIndex(1)

    def _pose_target_changed(self, hand, target):
        self.pose_canvas.select_target(hand, target)
        self.pose_timeline.select_target(hand, target)

    def _seek_pose(self, t):
        self.pause()
        self.seek(t + self.offset)
        self.pose_timeline.set_time(t, True)

    def _show_problem_tab(self):
        self.inspector_tabs.setCurrentIndex(0)
        self.tabs.setCurrentIndex(2)

    def _refresh_pose_editor(self):
        self.pose_panel.set_document(self.document, self.offset)
        self.pose_canvas.set_document(self.document, self.source.chart if self.source else None)
        self.pose_timeline.set_document(self.document, self.offset)
        self.pose_canvas.set_time(self.music_time - self.offset)
        self.pose_panel.set_time(self.music_time - self.offset)

    def _edit_pose(self, pose, label):
        if self.document is None or self._io or (self.jobs.running and self.jobs.request['kind'] == 'generate'):
            return
        self.pause()
        try:
            self._changed(self.document.commit_pose(label, pose))
        except ValueError as exc:
            self._error(str(exc))

    def _changed(self, ids):
        if ids == []:
            return
        self._refresh_pose_editor()
        if ids is None:
            self._update_duration()
            self._update_title()
            self._refresh_actions()
            self._progress('修改已应用，可撤销；保存工程可保留当前修改')
            return
        if self.table_model.rowCount() != self._table_contact_count:
            self._table_contact_count = len(self.document)
            selected = self.selected_ids & set(range(len(self.document)))
            self._sync_selection = True
            self.table_model.set_document(self.document, self.offset)
            self._sync_selection = False
            self._set_selection(selected)
        self.table_model.changed(ids)
        self._update_duration()
        self.canvas.zone = None
        point_rows = self._selected_path_rows()
        point_count = self.path_model.rowCount()
        self._refresh_inspector()
        if len(self.selected_ids) == 1 and point_count == self.path_model.rowCount():
            selection = QItemSelection()
            if point_rows:
                low = previous = point_rows[0]
                for row in point_rows[1:] + [None]:
                    if row is not None and row == previous + 1:
                        previous = row
                        continue
                    selection.select(self.path_model.index(low, 0), self.path_model.index(previous, 2))
                    if row is not None:
                        low = previous = row
            self.path_table.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        # A continuation may change the physical DOWN of later segments, and
        # a split may remove rows. Rebuild the inexpensive structural feedback.
        problems = edit_check(self.document, self.source.chart if self.source else None)
        problems.extend(draft_warnings(self.document))
        self._show_problems(problems)
        self.canvas.update()
        self.timeline.update()
        self._update_title()
        self._refresh_actions()
        if any(p['severity'] == 'error' for p in problems):
            self._progress('动作存在需修复项，单击“问题”中的条目可定位可以继续编辑或保存工程')
        elif problems:
            self._progress('修改已应用“问题”页仍有初稿待检查项，点击“检查方案”更新结果；需要保留时请保存工程')
        else:
            self._progress('修改已应用点击“检查方案”查看结果；需要保留时请保存工程')

    def assign(self):
        self._edit(lambda: self.document.assign(self.selected_ids, self.hand.currentData(), self.finger.currentData()))

    def alternate(self):
        self._edit(lambda: self.document.alternate(self.selected_ids, self.finger.currentData(), self.hand.currentData()))

    def swap(self):
        self._edit(lambda: self.document.swap_hands(self.selected_ids))

    def set_lock(self, locked):
        self._edit(lambda: self.document.set_lock(self.selected_ids, locked))

    def reset_selected(self):
        self._edit(lambda: self.document.reset(self.selected_ids))

    def undo(self):
        self._restore_history(False)

    def redo(self):
        self._restore_history(True)

    def _restore_history(self, redo):
        if self.document is None or self._io or (self.jobs.running and self.jobs.request['kind'] == 'generate'):
            return
        stack = self.document.redo_stack if redo else self.document.undo_stack
        if not stack:
            return
        self.pause()
        self.canvas.drag = None
        self.canvas.update()
        change = stack[-1]
        changed = self.document.redo() if redo else self.document.undo()
        if change.kind == 'chart':
            self._restore_chart()
        elif change.kind == 'structure':
            selected = self.selected_ids & set(range(len(self.document)))
            self._set_document(self.document)
            self._set_selection(selected)
        else:
            self._changed(changed)
        self._progress(('已重做：' if redo else '已撤销：') + change.label)

    def _restore_chart(self):
        document = self.document
        reference = document.chart
        if self.source and all(self.source.reference.get(key) == reference.get(key) for key in ('path', 'sha256', 'chart_entry')):
            self.source.reference = copy.deepcopy(reference)
            audio = reference.get('audio')
            url = QUrl.fromLocalFile(audio) if audio else QUrl()
            if self.player.source() != url:
                self.player.setSource(url)
                self.player.setPosition(round(self.music_time * 1000))
            self._changed(None)
            return
        self._set_source(None)
        self._set_document(document)
        if reference.get('path'):
            def restored(source):
                if set(document.note_contacts) != set(range(source.reference['note_count'])):
                    raise ValueError('此方案的音符编号与当前谱面不匹配，请选择正确的谱面')
                source.reference = copy.deepcopy(reference)
                self._set_source(source)
                self._set_document(document)
            self._load_source(reference['path'], restored, expected=reference,
                              audio=reference.get('audio'), background=reference.get('background'))

    def _point_selected(self, contact, point):
        if self.selected_ids != {contact}:
            self._set_selection({contact})
        self.canvas.active_point = point
        if point not in self._selected_path_rows():
            self.path_table.selectRow(point)
        self.tabs.setCurrentIndex(1)
        self.seek(self.document.contact(contact)['points'][point][0] + self.offset)

    def _path_row_selected(self, row):
        if len(self.selected_ids) == 1:
            self.canvas.active_point = row
            self.seek(self.document.contact(next(iter(self.selected_ids)))['points'][row][0] + self.offset)

    def _selected_path_rows(self):
        return sorted(index.row() for index in self.path_table.selectionModel().selectedRows())

    def _path_selection_changed(self, *_):
        rows = self._selected_path_rows()
        self.canvas.selected_points = set(rows)
        self.canvas.update()
        if not self.document or not self.selected_ids:
            self.path_summary.setText('选择一个动作可编辑控制点；选择多个动作可批量修改轨迹')
            return
        count = sum(len(self.document.contact(i)['points']) for i in self.selected_ids)
        self.path_summary.setText(f'{len(self.selected_ids)} 段轨迹 · {count:,} 个控制点' +
                                 (f' · 已选 {len(rows):,} 点' if len(self.selected_ids) == 1 else ' · 可批量修改'))

    def _drag_path_mode(self, enabled):
        self.canvas.drag_whole_path = enabled

    def batch_paths(self):
        if not self.document or not self.selected_ids:
            return
        if self._io or (self.jobs.running and self.jobs.request['kind'] == 'generate'):
            return
        self.pause()
        ids = self.document.sorted_ids(self.selected_ids)
        contacts = [self.document.contact(i) for i in ids]
        rows = self._selected_path_rows() if len(ids) == 1 else []
        reference = contacts[0]['points'][rows[0] if rows else 0][1:]
        dialog = BatchPathDialog(contacts, rows, reference, self)
        if dialog.exec() != QDialog.Accepted:
            return
        options = dialog.values()
        selected = rows if dialog.scope.currentData() == 'rows' else None
        self._edit(lambda: self.document.edit_paths(ids, rows=selected, **options))

    def edit_flicks(self):
        if not self.document or not self.source or not self.selected_ids:
            return
        if self._io or (self.jobs.running and self.jobs.request['kind'] == 'generate'):
            return
        from flowphi.ui.gestures import flick_contacts
        try:
            entries = flick_contacts(self.document, self.source.chart, self.selected_ids)
        except ValueError as exc:
            self._error(str(exc))
            return
        if not entries:
            self._progress('所选动作没有关联 Flick请先在动作列表或时间轴中选择包含 Flick 的动作')
            return
        self.pause()
        dialog = FlickDialog(entries, self.source.chart, self.offset, self)
        if dialog.exec() == QDialog.Accepted:
            self._edit(lambda: self.document.set_flicks(self.selected_ids, self.source.chart, **dialog.values()))

    def edit_release(self):
        if not self.document or not self.selected_ids:
            return
        if self._io or (self.jobs.running and self.jobs.request['kind'] == 'generate'):
            return
        self.pause()
        ids = self.document.sorted_ids(self.selected_ids)
        dialog = ReleaseDialog([self.document.contact(i) for i in ids], self.offset, self)
        if dialog.exec() == QDialog.Accepted:
            self._edit(lambda: self.document.set_release(ids, **dialog.values()))

    def _point_edited(self, contact, point, x, y):
        _, old_x, old_y = self.document.contact(contact)['points'][point]
        if abs(x - old_x) < 1e-12 and abs(y - old_y) < 1e-12:
            return
        rows = self._selected_path_rows()
        if self.canvas.drag_whole_path or (point in rows and len(rows) > 1):
            self._edit(lambda: self.document.edit_paths([contact], operation='translate',
                       rows=None if self.canvas.drag_whole_path else rows, x=x - old_x, y=y - old_y))
            return
        points = [list(p) for p in self.document.contact(contact)['points']]
        points[point][1:] = [x, y]
        self._edit(lambda: self.document.set_points(contact, points))

    def _path_edited(self, points):
        if len(self.selected_ids) == 1:
            i = next(iter(self.selected_ids))
            self._edit(lambda: self.document.set_points(i, points))

    def _add_point_at(self, x, y):
        if len(self.selected_ids) != 1:
            return
        i = next(iter(self.selected_ids))
        c = self.document.contact(i)
        t = round((self.music_time - self.offset) * 1000) / 1000
        if not c['start'] < t < c['end']:
            self._error('请先把播放位置移到这段动作的按下与抬起之间，再添加控制点')
            return
        if any(abs(p[0] - t) < .0005 for p in c['points']):
            self._error('这个时刻已有控制点（时间精确到毫秒），可以直接修改该点')
            return
        points = sorted([*c['points'], [t, x, y]], key=lambda p: p[0])
        self._edit(lambda: self.document.set_points(i, points))

    def add_path_point(self):
        if len(self.selected_ids) == 1:
            c = self.document.contact(next(iter(self.selected_ids)))
            self._add_point_at(*point_at(c['points'], self.music_time - self.offset))

    def remove_path_point(self):
        if len(self.selected_ids) != 1:
            return
        rows = self._selected_path_rows()
        if not any(row > 0 for row in rows):
            self._error('请先在表格中选择要删除的控制点第一个点会自动保留')
            return
        i = next(iter(self.selected_ids))
        self._edit(lambda: self.document.edit_paths([i], operation='delete', rows=rows))

    def show_zone(self):
        if len(self.selected_ids) == 1 and self.source:
            i = next(iter(self.selected_ids))
            row = max(0, min(self.canvas.active_point, len(self.document.contact(i)['points']) - 1))
            t = self.document.contact(i)['points'][row][0]
            self.seek(t + self.offset)
            self.jobs.start('zone', self.document, contact=i, time=t)

    def replan(self):
        if self.document and self.source and self.selected_ids:
            self.jobs.start('replan', self.document, selected=sorted(self.selected_ids),
                            beam_width=self.performance['replan_beam'], max_contacts=self.performance['replan_contacts'])

    def validate_plan(self):
        if self.document and self.source:
            self.jobs.start('validate', self.document)

    def edit_touch_mode(self, mode):
        if self.document and self.selected_ids:
            self._edit(lambda: self.document.set_touch_mode(self.selected_ids, mode))

    def split_heads(self):
        if self.document and self.source and self.selected_ids:
            count = len(self.document)
            self._edit(lambda: self.document.split_heads(self.selected_ids, self.source.chart))
            if len(self.document) > count:
                self._set_selection(self.selected_ids | set(range(count, len(self.document))))

    def export(self):
        if not self.document or not self.source:
            return
        parent = QFileDialog.getExistingDirectory(self, '选择导出位置（会新建目录存放动作数据）', str(OUTPUTS))
        if not parent:
            return
        output = Path(parent) / ('fingering-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.jobs.start('export', self.document, output=str(output))

    def render(self, full):
        if not self.document or not self.source:
            return
        start, end = self.range_start.value(), self.range_end.value()
        dialog = RenderDialog(self._asset_defaults(), start, max(.1, end - start), self, full=full)
        if dialog.exec() != QDialog.Accepted:
            return
        options = dialog.values()
        if not dialog.output.text().strip():
            self._error('请选择一个输出目录')
            return
        output = Path(dialog.output.text().strip()).resolve()
        if any((output / name).exists() for name in ('plan.psap', 'motion-plan.json')):
            self._error('此目录已有导出的动作数据，请选择新的输出目录')
            return
        for key in ('model', 'resources', 'blender'):
            self.settings.setValue(key, options[key])
        if options.get('background'):
            if not self._associate_media('background', options['background']):
                return
        self.jobs.start('render', self.document, output=str(output), options=options)

    def choose_audio(self):
        if not self.source:
            return
        path, _ = QFileDialog.getOpenFileName(self, '关联音乐', str(Path(self.source.reference['path']).parent), '音频 (*.wav *.ogg *.mp3 *.flac)')
        if path:
            self._associate_media('audio', path)

    def choose_background(self):
        if not self.source:
            return
        current = self.source.reference.get('background') or self.source.reference['path']
        path, _ = QFileDialog.getOpenFileName(self, '关联曲绘', str(Path(current).parent),
                                            '曲绘 (*.png *.jpg *.jpeg *.webp);;所有文件 (*)')
        if path and self._associate_media('background', path):
            self._progress('曲绘已关联，将用于下一次视频保存工程可保留此设置')

    def _associate_media(self, key, path):
        label = '关联音乐' if key == 'audio' else '关联曲绘'
        path = Path(path).resolve()
        if not path.is_file():
            self._error(label + '失败，找不到文件，请重新选择')
            return False
        if self.source.reference.get(key) == str(path):
            return True
        reference = self.source.reference | {key: str(path)}
        if self.document is not None:
            self.document.commit_chart(label, reference)
        self.pause()
        self.source.reference = reference
        if key == 'audio':
            self.player.setSource(QUrl.fromLocalFile(str(path)))
            self.player.setPosition(round(self.music_time * 1000))
        self._update_title()
        self._refresh_actions()
        return True

    def configure_interface_scale(self):
        current = QApplication.instance().property('interface_scale') or 100
        dialog = InterfaceScaleDialog(interface_scale(self.settings), current, self)
        if dialog.exec() == QDialog.Accepted:
            value = dialog.scale.value()
            self.settings.setValue('ui_scale', value)
            self.settings.sync()
            self._progress(f'界面缩放已设为 {value}%，重启 FlowPhi 后生效')

    def configure_performance(self):
        dialog = PerformanceDialog(self.performance, self)
        if dialog.exec() == QDialog.Accepted:
            self.performance = dialog.values()
            for key, value in self.performance.items():
                self.settings.setValue(key, value)
            self._apply_performance()

    def _apply_performance(self):
        self.timer.setInterval(max(1, round(1000 / self.performance['preview_fps'])))
        self.canvas.lookahead = float(self.performance['lookahead'])
        if self.canvas.render_width != self.performance['preview_width']:
            self.canvas.render_width = self.performance['preview_width']
            self.canvas.set_source(self.source)
        self.timeline.span = float(self.performance['timeline_span'])
        self.pose_timeline.span = float(self.performance['timeline_span'])
        if self.document:
            self.document.history_limit = self.performance['history_mib'] * 1024 * 1024
        self.canvas.update()
        self.timeline.update()

    def compare_baseline(self, checked=None):
        if checked is not None:
            self.compare_action.setChecked(bool(checked))
        self.canvas.compare = self.compare_action.isChecked()
        self.canvas.update()

    def toggle_play(self):
        if not self.source:
            return
        if self.playing:
            self.pause()
            return
        if self.loop.isChecked() and not self.range_start.value() <= self.music_time < self.range_end.value():
            self.seek(self.range_start.value())
        if self.music_time >= self.duration:
            self.seek(0.)
        self.playing = True
        self._play_rate = float(self.speed.currentData())
        self.anchor_time, self.anchor_clock = self.music_time, time.monotonic()
        self.player.setPlaybackRate(self.speed.currentData())
        if not self.player.source().isEmpty():
            self.player.play()
        self.play_button.setText('暂停 (Space)')
        self.timer.start()

    def pause(self):
        if self.playing:
            self.music_time = min(self.duration, self.anchor_time + (time.monotonic() - self.anchor_clock) * self._play_rate)
        self.playing = False
        self.timer.stop()
        self.player.pause()
        self.play_button.setText('播放 (Space)')
        self._display_time()

    def seek(self, seconds):
        self.music_time = max(0., min(self.duration, float(seconds)))
        self.anchor_time, self.anchor_clock = self.music_time, time.monotonic()
        self._seeking = True
        self.player.setPosition(round(self.music_time * 1000))
        if self.playing and not self.player.source().isEmpty():
            self.player.play()
        self._seeking = False
        self._display_time(follow=True)

    def _tick(self):
        self.music_time = self.anchor_time + (time.monotonic() - self.anchor_clock) * self._play_rate
        if self.loop.isChecked() and self.range_end.value() > self.range_start.value() and self.music_time >= self.range_end.value():
            self.seek(self.range_start.value())
        elif self.music_time >= self.duration:
            self.pause()
            self.seek(self.duration)
        else:
            self._display_time(follow=True)

    def _display_time(self, follow=False):
        self.clock_label.setText(f'{int(self.music_time // 60):02d}:{self.music_time % 60:06.3f}')
        if not self.seek_slider.isSliderDown():
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(round(self.music_time * 1000))
            self.seek_slider.blockSignals(False)
        self.canvas.set_time(self.music_time - self.offset)
        self.timeline.set_time(self.music_time - self.offset, follow)
        self.pose_canvas.set_time(self.music_time - self.offset)
        self.pose_timeline.set_time(self.music_time - self.offset, follow)
        self.pose_panel.set_time(self.music_time - self.offset)

    def _speed_changed(self):
        was_playing = self.playing
        self.pause()
        self.player.setPlaybackRate(self.speed.currentData())
        if was_playing:
            self.toggle_play()

    def _audio_position(self, milliseconds):
        if self.playing and not self._seeking:
            estimated = self.anchor_time + (time.monotonic() - self.anchor_clock) * self._play_rate
            if abs(milliseconds / 1000 - estimated) > .15:
                self.anchor_time, self.anchor_clock = milliseconds / 1000, time.monotonic()

    def _audio_duration(self, milliseconds):
        if milliseconds > 0:
            self._update_duration()

    def _media_status(self, status):
        if status == QMediaPlayer.EndOfMedia and self.playing and not self.loop.isChecked():
            self.pause()

    def _timeline_range(self, start, end):
        self.pause()
        self.range_start.setValue(max(0., start + self.offset))
        self.range_end.setValue(max(0., end + self.offset))

    def select_range(self):
        if self.document:
            start, end = self.range_start.value() - self.offset, self.range_end.value() - self.offset
            self._set_selection(set(self.document.index.query(start, end)))
            self.timeline.start = start
            self.timeline.span = max(.5, end - start)
            self.timeline.update()

    def _show_problems(self, problems):
        self._problems = problems
        self.timeline.conflict_ids = {i for p in problems if p['code'] == 'finger_overlap' for i in p['contacts']}
        self.timeline._layout_key = None
        self.timeline.update()
        self.problem_list.clear()
        for problem in problems[:1000]:
            label = '需修复' if problem['severity'] == 'error' else '提示'
            duration = (f"–{problem['end_time'] + self.offset:.3f}" if 'end_time' in problem else '')
            text = f"{label} · {problem['time'] + self.offset:.3f}{duration} s\n{problem['message']}"
            item = QTreeWidgetItem([text])
            item.setData(0, Qt.UserRole, problem)
            item.setToolTip(0, text)
            self.problem_list.addTopLevelItem(item)
            if self.document:
                for i in self.document.sorted_ids(problem['contacts']):
                    summary = contact_label(i, self.document.contact(i), self.offset)
                    child = QTreeWidgetItem([summary.replace(' · 音符 ', '\n音符 ', 1)])
                    child.setData(0, Qt.UserRole, dict(problem, contacts=[i]))
                    child.setToolTip(0, summary + '\n单击只选择这个动作，保留当前冲突时刻')
                    item.addChild(child)
            item.setExpanded(problem['code'] == 'finger_overlap')
        self.tabs.setTabText(2, f'问题 ({len(problems)})' if problems else '问题')

    def _problem_selected(self, item, column=0):
        problem = item.data(0, Qt.UserRole)
        self._set_selection(set(problem['contacts']))
        self.pause()
        self.seek(problem['time'] + self.offset)
        self.timeline.focus_interval(problem['time'], problem.get('end_time', problem['time']))
        bounds = self.timeline.contact_bounds(problem['contacts'])
        if bounds is not None:
            x, y = round(bounds.center().x()), round(bounds.center().y())
            margin = min(100, round(bounds.height() / 2) + 8)
            QTimer.singleShot(0, lambda: self.timeline_scroll.ensureVisible(x, y, 0, margin))
        self._progress('已定位问题点击展开的动作可单独选择，再修改手指或抬起时间')

    def _refresh_actions(self):
        if not hasattr(self, 'canvas'):
            return
        available = not self.jobs.running and not self._io
        loaded = self.document is not None
        ready = loaded and self.source is not None
        for action in (self.open_chart_action, self.open_project_action, self.demo_action):
            action.setEnabled(available)
        self.generate_action.setEnabled(available and self.source is not None)
        self.save_action.setEnabled(loaded and not self._io)
        self.save_as_action.setEnabled(loaded and not self._io)
        for action in (self.validate_action, self.export_action, self.preview_action, self.render_action):
            action.setEnabled(available and ready)
        self.replan_action.setEnabled(available and ready and bool(self.selected_ids))
        self.replan_button.setEnabled(self.replan_action.isEnabled())
        generating = bool(self._io) or (self.jobs.running and self.jobs.request['kind'] == 'generate')
        self.canvas.editable = not generating
        self.pose_canvas.editable = loaded and not generating
        self.pose_panel.setEnabled(loaded and not generating)
        self.pose_tool.setEnabled(loaded and not generating)
        for action, label, stack in (
            (self.undo_action, '撤销', self.document.undo_stack if loaded else []),
            (self.redo_action, '重做', self.document.redo_stack if loaded else [])):
            action.setEnabled(bool(stack) and not generating)
            action.setToolTip(f'{label}：{stack[-1].label}' if stack else f'没有可{label}的操作')
        self.compare_action.setEnabled(loaded)
        self.audio_action.setEnabled(self.source is not None and available)
        self.background_action.setEnabled(self.source is not None and available)
        self.attach_chart_action.setEnabled(loaded and available)
        self.play_button.setEnabled(self.source is not None)
        for button in self.edit_buttons:
            button.setEnabled(loaded and bool(self.selected_ids) and not generating)
        single = loaded and len(self.selected_ids) == 1 and not generating
        self.add_path_button.setEnabled(single)
        self.remove_path_button.setEnabled(single)
        self.path_table.setEnabled(single)
        self.drag_path.setEnabled(single)
        self.batch_path_button.setEnabled(loaded and bool(self.selected_ids) and not generating)
        self.batch_path_action.setEnabled(self.batch_path_button.isEnabled())
        self.release_button.setEnabled(self.batch_path_button.isEnabled())
        self.release_action.setEnabled(self.release_button.isEnabled())
        self.flick_button.setEnabled(self.batch_path_button.isEnabled() and ready)
        self.flick_action.setEnabled(self.flick_button.isEnabled())
        self.touch_mode.setEnabled(self.release_button.isEnabled())
        self.touch_button.setEnabled(self.release_button.isEnabled())
        for action in self.touch_actions:
            action.setEnabled(self.release_button.isEnabled())
        self.split_heads_action.setEnabled(self.flick_button.isEnabled())
        self.split_heads_button.setEnabled(self.split_heads_action.isEnabled())
        self.zone_button.setEnabled(single and available and ready)

    def _update_title(self):
        if self.document:
            doc = self.document
            self.setWindowTitle(('● ' if doc.dirty else '') + doc.title + ' · flowphi')
            pose_label = '自定义姿态' if any(doc.pose['defaults'].values()) or doc.pose['clips'] else '自动姿态'
            self.summary.setText(f'{doc.title}  ·  {len(doc.note_contacts)} 个音符 / {len(doc)} 段动作  ·  指法修改 {len(doc.edits)} 段  ·  {pose_label}')
        elif self.source:
            self.setWindowTitle('flowphi')
            title = Path(self.source.reference['path']).stem
            if self.source.reference.get('chart_entry'):
                title += ' / ' + self.source.reference['chart_entry']
            self.summary.setText(f"{title}  ·  {self.source.reference['note_count']} 个音符  ·  尚未生成指法")
        else:
            self.summary.setText('打开谱面 → 生成指法初稿 → 调整指法与姿态 → 保存工程或生成视频')

    def _busy_changed(self, busy):
        if busy:
            self.log_view.clear()
            self._job_log = Path(self.jobs.request['work']) / 'job.log'
            self.open_log_button.setEnabled(False)
            self.open_log_button.setToolTip(str(self._job_log))
            kind = self.jobs.request['kind']
            self.log_dock.setWindowTitle('后台任务日志 · ' + TASK_NAMES.get(kind, kind))
            self.task_status.begin(kind)
        self.cancel_button.setVisible(busy)
        self._refresh_actions()

    def _job_progress(self, entry):
        self.task_status.update_progress(entry)

    def show_job_log(self):
        self.log_dock.show()
        self.log_dock.raise_()

    def open_job_log(self):
        if self._job_log and self._job_log.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._job_log)))

    def open_app_log(self):
        if not APP_LOG.is_file():
            self._error('程序日志尚未创建：' + str(APP_LOG))
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(APP_LOG))):
            self._error('无法打开程序日志，请手动查看：' + str(APP_LOG))

    def _progress(self, text):
        self.progress_label.setText(' '.join(text.splitlines()))
        self.progress_label.setToolTip(text)

    def _log(self, text):
        scroll = self.log_view.verticalScrollBar()
        position = scroll.value()
        follow = position >= scroll.maximum() - 2
        self.log_view.appendPlainText(text)
        scroll.setValue(scroll.maximum() if follow else position)
        if self._job_log and self._job_log.is_file():
            self.open_log_button.setEnabled(True)

    def _error(self, message):
        logging.getLogger(__name__).error('%s', message)
        self._progress(message)

    @Slot(object)
    def _job_completed(self, message):
        self.task_status.finish(message)
        if self._pending_close:
            QTimer.singleShot(0, self.close)
            return
        if message.get('cancelled'):
            self._progress('本次后台任务已取消')
            return
        if not message.get('ok'):
            self._error(message.get('error', '后台任务失败请通过“窗口 → 后台任务日志”查看详情'))
            return
        request, result = message['request'], message['result']
        kind = request['kind']
        if kind == 'generate':
            self._async(lambda: read_document(result['project']), self._generated)
            return
        current = self.document and id(self.document) == request.get('document_id') and self.document.revision == request.get('revision')
        if kind == 'replan':
            if not current:
                self._progress('计算期间方案已修改，本次手指分配结果未应用请重新执行“重新分配手指”')
                return
            self._changed(self.document.accept_assignments(result['assignments']))
            self._progress(f"重新分配完成：调整 {result['changed_contacts']} 段动作的手指，已锁定的指法保持")
        elif kind == 'zone':
            if current and self.selected_ids == {result['contact']}:
                self.canvas.zone = result
                self.canvas.update()
                self._progress(f"绿色区域对应 {result['time'] + self.offset:.3f} s 的所选控制点")
        elif kind == 'validate':
            if current:
                self._show_problems(result['issues'])
                self._progress(('触控检查通过' + (f"另有 {len(result['issues'])} 条提示，可在“问题”页查看" if result['issues'] else '') +
                                '导出时会再次检查最终触控数据') if result['valid'] else '触控检查发现需修复项，单击“问题”页中的条目可定位')
                if result['issues']:
                    self._show_problem_tab()
            else:
                self._progress('检查期间方案已修改，请重新点击“检查方案”获取当前结果')
        elif kind in ('export', 'render'):
            if current:
                self._show_problems(result['validation']['issues'])
            if not result.get('exported'):
                self._show_problem_tab()
                self._progress('检查未通过，本次未导出数据或生成视频请处理“问题”中的需修复项后重试')
                return
            self._last_output = result['output']
            suffix = '' if current else '\n此结果对应任务开始时的方案，后续修改未包含在内'
            if result.get('video'):
                answer = QMessageBox.information(self, '视频已生成', result['video'] + suffix + '\n\n现在打开视频？', QMessageBox.Yes | QMessageBox.No)
                if answer == QMessageBox.Yes:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(result['video']))
            else:
                QMessageBox.information(self, '动作数据已导出', result['output'] + suffix + '\n\n包含触控序列、带姿态设置的动作数据、检查报告和可继续编辑的工程')
            self._progress('完成：' + result['output'])

    def _generated(self, document):
        self.pause()
        if self.document is not None:
            self.document.replace_plan(document)
            document = self.document
            self._set_document(document)
        else:
            # This path is a cache snapshot, not the user's chosen save destination.
            document.path = None
            document._saved_state = -1
            self._attach_project(document, self.source)
        pending = sum(c.get('draft_fallback') is not None for c in document.baseline['contacts'])
        if pending:
            self._show_problem_tab()
            self._progress(f'初稿已生成，保留全部音符；{pending} 段动作使用临时指法，需要检查单击“问题”中的条目可定位，修改后点击“检查方案”')
        else:
            self._progress('指法初稿已生成选择动作可修改手指和轨迹；切到“手部姿态”可调整手腕和空闲手指')

    def closeEvent(self, event):
        if self._io:
            self._pending_close = True
            event.ignore()
            return
        if self.jobs.running:
            if QMessageBox.question(self, '结束后台任务', '后台任务仍在运行结束本次任务并退出？', QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                event.ignore()
                return
            self._pending_close = True
            self.jobs.cancel()
            event.ignore()
            return
        if any(t.isRunning() for t in self.jobs.findChildren(QThread)):
            self._pending_close = True
            QTimer.singleShot(100, self.close)
            event.ignore()
            return
        self._pending_close = False
        if self.document and self.document.dirty:
            answer = QMessageBox.warning(self, '保存当前工程', '退出前保存当前工程？',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Save:
                event.ignore()
                self.save_project(on_saved=self.close)
                return
        self.pause()
        self.player.stop()
        event.accept()
