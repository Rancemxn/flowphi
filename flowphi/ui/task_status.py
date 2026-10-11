from __future__ import annotations

import math
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QVBoxLayout


TASK_NAMES = dict(generate='生成指法初稿', replan='重新分配手指', zone='计算可用区域',
                  validate='检查方案', export='导出动作数据', render='生成视频')


def duration_text(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes:02d}:{seconds:02d}'


class TaskStatus(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('taskStatus')
        self.setStyleSheet('QFrame#taskStatus { background: #1a2738; border: 1px solid #3a4b63; border-radius: 5px; }'
                          'QProgressBar { border: 1px solid #40536b; border-radius: 3px; background: #111824;'
                          'color: #e2eaf5; text-align: center; min-height: 18px; }'
                          'QProgressBar::chunk { background: #38699a; }')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 6, 9, 6)
        layout.setSpacing(4)
        top = QHBoxLayout()
        self.title = QLabel('暂无后台任务')
        self.metrics = QLabel()
        top.addWidget(self.title, 1)
        top.addWidget(self.metrics)
        layout.addLayout(top)
        row = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.setValue(0)
        self.bar.setFormat('等待任务')
        self.bar.setToolTip('显示当前阶段的真实进度；总量尚未确定时显示活动进度条')
        self.log_button = QPushButton('查看日志')
        self.cancel_button = QPushButton('取消任务')
        self.cancel_button.hide()
        hide_button = QPushButton('隐藏')
        hide_button.setToolTip('隐藏进度条，任务继续运行；可从“窗口 → 显示任务进度”重新显示')
        hide_button.clicked.connect(self.hide)
        row.addWidget(self.bar, 1)
        row.addWidget(self.log_button)
        row.addWidget(self.cancel_button)
        row.addWidget(hide_button)
        layout.addLayout(row)
        self.detail = QLabel()
        # Reserve each text row so stage changes and long paths cannot resize the workspace.
        for label in (self.title, self.detail):
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            label.setFixedHeight(label.fontMetrics().lineSpacing())
        layout.addWidget(self.detail)
        self.name = '后台任务'
        self.active = False
        self.stage = None
        self.started = self.stage_started = self.updated = 0.
        self.done = self.total = None
        self.warning_count = 0
        self._detail = ''
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._clock)
        self.hide()

    def begin(self, kind):
        self.name = TASK_NAMES.get(kind, kind)
        self.active = True
        self.started = self.stage_started = self.updated = time.monotonic()
        self.stage = None
        self.done = self.total = None
        self.warning_count = 0
        self._detail = ''
        self.title.setText(self.name + ' · 准备任务')
        self.bar.setRange(0, 0)
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.show()
        self._clock()
        self.timer.start()

    def update_progress(self, entry):
        if not self.active:
            return
        now = time.monotonic()
        stage = entry.get('stage') or entry.get('message', '')
        if stage != self.stage:
            self.stage, self.stage_started = stage, now
            self.done = self.total = None
        self.updated = now
        self.title.setText(f"{self.name} · {entry.get('message', '')}")
        self._detail = str(entry.get('detail', ''))
        done, total = entry.get('done'), entry.get('total')
        numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                      for value in (done, total))
        if numeric and total > 0:
            if self.done is not None and done < self.done:
                self.stage_started = now
            self.done, self.total = max(0, min(done, total)), total
            self.bar.setRange(0, 1000)
            self.bar.setValue(round(self.done / total * 1000))
            self.bar.setFormat(f"%p% · {self.done:g}/{total:g} {entry.get('unit', '')}".rstrip())
        else:
            self.done = self.total = None
            self.bar.setRange(0, 0)
        self._clock()

    def note_warning(self):
        self.warning_count += 1
        self._clock()

    def _clock(self):
        if not self.started:
            return
        now = time.monotonic()
        text = '已用 ' + duration_text(now - self.started)
        if self.active and self.done and self.total and self.done < self.total and now - self.stage_started >= 2:
            remaining = (now - self.stage_started) * (self.total - self.done) / self.done
            text += ' · 本阶段预计还需 ' + duration_text(remaining)
        self.metrics.setText(text)
        self.title.setToolTip(self.title.text())
        details = [self._detail] if self._detail else []
        if self.warning_count:
            details.append(f'{self.warning_count} 条警告，详见日志')
        if self.active and now - self.updated >= 10:
            details.append(f'距上次进度更新 {int(now - self.updated)} 秒')
        detail = ' · '.join(details)
        self.detail.setText(' '.join(detail.splitlines()))
        self.detail.setToolTip(detail)

    def finish(self, message):
        self.active = False
        self.timer.stop()
        self.cancel_button.hide()
        result = message.get('result', {})
        if message.get('cancelled'):
            status = '已取消'
        elif not message.get('ok'):
            status = '失败'
            self._detail = message.get('error', '请查看后台任务日志')
        elif result.get('exported') is False or result.get('valid') is False:
            status = '检查未通过'
            self._detail = '请在“问题”页查看需修复项'
        else:
            status = '已完成'
            self._detail = result.get('video') or result.get('output') or ''
        self.bar.setRange(0, 1000)
        self.bar.setValue(1000 if status == '已完成' else max(0, self.bar.value()))
        self.bar.setFormat(status)
        self.title.setText(self.name + ' · ' + status)
        self._clock()
