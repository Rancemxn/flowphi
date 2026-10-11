from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import uuid

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QThread, Signal, Slot

from flowphi.ui.model import write_json_atomic
from flowphi.ui.sources import DEFAULT_CACHE, ROOT
from flowphi.runtime import module_command


class AsyncCall(QThread):
    succeeded = Signal(object)
    failed = Signal(object)

    def __init__(self, key, function, parent=None):
        super().__init__(parent)
        self.key, self.function = key, function

    def run(self):
        try:
            self.succeeded.emit(dict(key=self.key, value=self.function()))
        except Exception as exc:
            self.failed.emit(dict(key=self.key, error=str(exc)))


class JobRunner(QObject):
    progress = Signal(str)
    progress_details = Signal(object)
    log_line = Signal(str)
    warning = Signal(str)
    completed = Signal(object)
    busy_changed = Signal(bool)

    def __init__(self, parent=None, cache=DEFAULT_CACHE):
        super().__init__(parent)
        self.cache = Path(cache)
        self.process = None
        self.writer = None
        self.killer = None
        self.running = False
        self.cancelled = False
        self.request = None
        self.buffer = b''
        self.log = None

    def start(self, kind, document=None, **payload):
        if self.running:
            raise ValueError('请先完成或取消当前后台任务')
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:7]
        work = self.cache / 'jobs' / stamp
        request = dict(kind=kind, work=str(work), cache=str(self.cache), **payload)
        snapshot = document.snapshot() if document else None
        if document:
            request['project'] = str(work / 'input.fingering.json')
            request['revision'] = document.revision
            request['document_id'] = id(document)
        self.request = request
        self.running, self.cancelled, self.buffer = True, False, b''
        self.busy_changed.emit(True)
        self.progress.emit('准备后台任务…')
        self.progress_details.emit(dict(stage='prepare', message='准备任务文件'))

        def prepare():
            work.mkdir(parents=True, exist_ok=True)
            if snapshot:
                write_json_atomic(request['project'], snapshot)
            write_json_atomic(work / 'request.json', request)
            return str(work / 'request.json')

        self.writer = AsyncCall(stamp, prepare, self)
        self.writer.succeeded.connect(self._prepared)
        self.writer.failed.connect(self._prepare_failed)
        self.writer.finished.connect(self.writer.deleteLater)
        self.writer.start()

    @Slot(object)
    def _prepared(self, message):
        if self.cancelled:
            self._complete(dict(ok=False, cancelled=True))
            return
        self.log = (Path(self.request['work']) / 'job.log').open('w', encoding='utf-8')
        self.process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONUTF8', '1')
        environment.insert('PYTHONIOENCODING', 'utf-8')
        environment.insert('FLOWPHI_PROGRESS_JSON', '1')
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        if os.name != 'nt' and hasattr(QProcess, 'UnixProcessParameters'):
            parameters = QProcess.UnixProcessParameters()
            parameters.flags = QProcess.UnixProcessFlag.CreateNewSession
            self.process.setUnixProcessParameters(parameters)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        command = module_command('flowphi.ui.worker', message['value'])
        self.process.start(command[0], command[1:])

    @Slot(object)
    def _prepare_failed(self, message):
        self._complete(dict(ok=False, error=message['error']))

    def _read(self, final=False):
        if self.process is not None:
            self.feed_output(bytes(self.process.readAllStandardOutput()), final=final)

    def feed_output(self, data, *, final=False):
        self.buffer += data
        rows = re.split(rb'\r\n|[\r\n]', self.buffer)
        self.buffer = b'' if final else rows.pop()
        for row in rows:
            if row:
                self._consume_line(row.decode('utf-8', errors='replace'))

    def _consume_line(self, line):
        if self.log:
            self.log.write(line + '\n')
            self.log.flush()
        line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line).rstrip()
        if not line:
            return
        try:
            entry = json.loads(line)
        except (ValueError, TypeError):
            entry = None
        text = line
        if isinstance(entry, dict) and entry.get('event') in ('progress', 'log'):
            text = str(entry.get('message', ''))
            if entry.get('event') == 'progress':
                if entry.get('total') is not None and entry.get('done') is not None:
                    text += f" · {entry['done']}/{entry['total']} {entry.get('unit', '')}".rstrip()
                if entry.get('detail'):
                    text += ' · ' + str(entry['detail'])
                self.progress.emit(text)
                self.progress_details.emit(entry)
            if entry.get('level') == 'warning':
                text = '警告：' + text
                self.warning.emit(text)
            elif entry.get('level') == 'error':
                text = '错误：' + text
        elif line.startswith(('flowphi:', 'HANDCAM_STATUS ')):
            self.progress.emit(line)
        if line.startswith(('Warning:', 'WARNING:', '警告：')):
            self.warning.emit(line)
        self.log_line.emit(f"[{datetime.now():%H:%M:%S}] {text}")

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self._complete(dict(ok=False, error=self.process.errorString()))

    def _finished(self, exit_code, exit_status):
        if not self.running:
            return
        self._read(final=True)
        if self.cancelled:
            result = dict(ok=False, cancelled=True)
        else:
            path = Path(self.request['work']) / 'result.json'
            try:
                result = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                result = dict(ok=False, error=f'后台任务意外结束（退出码 {exit_code}）请通过“窗口 → 后台任务日志”查看详情')
        self._complete(result)

    def _complete(self, result):
        if not self.running:
            return
        if self.log:
            self.log.close()
            self.log = None
        result['request'] = self.request
        self.running = False
        self.busy_changed.emit(False)
        self.completed.emit(result)
        if self.process:
            self.process.deleteLater()
            self.process = None

    def cancel(self):
        if not self.running:
            return
        self.cancelled = True
        self.progress.emit('正在结束本次后台任务…')
        self.progress_details.emit(dict(stage='cancel', message='正在取消任务'))
        if self.process and self.process.state() != QProcess.NotRunning:
            pid = int(self.process.processId())
            if pid <= 0:
                return
            if os.name == 'nt':
                # Kill only the process tree created by this QProcess. This also
                # stops its Blender / FFmpeg child processes and leaves other jobs alone.
                self.killer = AsyncCall('cancel', lambda: subprocess.run(
                    ['taskkill', '/PID', str(pid), '/T', '/F'], stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW, timeout=10, check=True), self)
                self.killer.failed.connect(self._cancel_failed)
                self.killer.finished.connect(self.killer.deleteLater)
                self.killer.start()
            else:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    @Slot(object)
    def _cancel_failed(self, message):
        if not self.running or not self.cancelled:
            return
        self.progress.emit('结束进程失败：' + message['error'])
        if self.process:
            self.process.kill()
