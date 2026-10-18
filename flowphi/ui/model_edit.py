from __future__ import annotations

import copy
import json
from pathlib import Path
import time

from PySide6.QtCore import QByteArray, Qt, QTimer, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QListWidget, QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout, QWidget)

from flowphi.ui.assistant import (SYSTEM_PROMPT, editor_context, execute_tool, python_data,
    reassign_candidate, restore_version, save_version, versions_folder, edit_editor)
from flowphi.ui.chat_view import ChatView, TOOL_NAMES, conversation_entries, error_summary
from flowphi.ui.edit_history import EditHistory
from flowphi.ui.jobs import AsyncCall, JobRunner
from flowphi.ui.model import write_json_atomic
from flowphi.ui.model_api import (StreamReply, build_request, compaction_input,
    compaction_tail, estimate_tokens, parse_reply, tool_results, wire_messages)
from flowphi.ui.preferences import current_preferences


class PromptEdit(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not event.modifiers() & Qt.ShiftModifier:
            self.submitted.emit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def insertFromMimeData(self, source):
        if source.hasText():
            self.insertPlainText(source.text())


class ModelEditPanel(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.messages, self.pending = [], []
        self.reply = None
        self.active = False
        self.generation = 0
        self.calls = []
        self.identity = None
        self.usage = {}
        self.context_tokens = 0
        self.compacting = False
        self.scope = {}
        self.version_cache = {}
        self.version_thread = None
        self.versions_dirty = False
        self.current_call = None
        self.results = []
        self.network = QNetworkAccessManager(self)
        self.jobs = JobRunner(self)
        self.jobs.completed.connect(self.job_finished)
        self.jobs.progress.connect(lambda text: self.status.setText(text) if self.active else None)
        self.timeout = QTimer(self)
        self.timeout.setSingleShot(True)
        self.timeout.timeout.connect(self.calculation_timeout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.detach = QPushButton('独立窗口')
        self.detach.clicked.connect(self.toggle_floating)
        editor.model_dock.topLevelChanged.connect(self.floating_changed)
        layout.addWidget(self.detach, 0, Qt.AlignRight)
        tabs = self.tabs = QTabWidget()
        layout.addWidget(tabs)
        chat = QWidget()
        box = QVBoxLayout(chat)
        self.selection = QLabel()
        self.selection.setWordWrap(True)
        box.addWidget(self.selection)
        self.transcript = ChatView()
        self.transcript.setPlaceholderText('直接描述要改的时间段、指法或手部动作。修改会自动保存版本。')
        self.transcript.seek.connect(editor.seek)
        box.addWidget(self.transcript, 1)
        self.usage_bar = QLabel('上下文已用 0.0K')
        self.usage_bar.setStyleSheet('padding: 4px; background: #213146; border-radius: 3px;')
        self.usage_bar.setWordWrap(True)
        box.addWidget(self.usage_bar)
        self.status = QLabel('就绪')
        self.status.setWordWrap(True)
        box.addWidget(self.status)
        self.prompt = PromptEdit()
        self.prompt.setPlaceholderText('例如：12–16 秒只用左右食指\nEnter 发送 · Shift+Enter 换行 · 工作中也可补充要求')
        self.prompt.setMaximumHeight(115)
        self.prompt.submitted.connect(self.submit)
        box.addWidget(self.prompt)
        self.fields = {}
        row = QHBoxLayout()
        self.fields['model'] = QLineEdit(editor.settings.value('ai/model', '', str))
        self.fields['model'].setPlaceholderText('模型名称')
        self.fields['model'].setAccessibleName('模型名称')
        row.addWidget(self.fields['model'], 1)
        self.effort = QComboBox()
        for label, value in [('默认推理', 'default'), ('不推理', 'none'), ('极低', 'minimal'), ('低', 'low'),
                             ('中', 'medium'), ('高', 'high'), ('极高', 'xhigh'), ('最高', 'max')]:
            self.effort.addItem(label, value)
        self.effort.setCurrentIndex(max(0, self.effort.findData(editor.settings.value('ai/reasoning_effort', 'default', str))))
        self.effort.setToolTip('推理强度支持范围由模型和网关决定；默认不附加参数')
        row.addWidget(self.effort)
        box.addLayout(row)
        row = QHBoxLayout()
        self.send = QPushButton('发送')
        self.send.clicked.connect(self.submit)
        self.stop = QPushButton('停止')
        self.stop.clicked.connect(self.cancel)
        self.stop.setEnabled(False)
        clear = QPushButton('新对话')
        clear.clicked.connect(self.new_conversation)
        self.compact = QPushButton('压缩上下文')
        self.compact.setToolTip('点击后压缩上下文，保留要求和修改进度；聊天记录保留。')
        self.compact.clicked.connect(self.compact_context)
        for button in (clear, self.compact, self.stop, self.send):
            row.addWidget(button)
        box.addLayout(row)
        tabs.addTab(chat, '对话')

        connection = QWidget()
        box = QVBoxLayout(connection)
        form = QFormLayout()
        self.protocol = QComboBox()
        for title, key in [('OpenAI · Chat Completions', 'openai'), ('OpenAI · Responses', 'responses'),
                           ('Anthropic · Messages', 'anthropic')]:
            self.protocol.addItem(title, key)
        self.protocol.setCurrentIndex(max(0, self.protocol.findData(editor.settings.value('ai/protocol', 'openai', str))))
        form.addRow('协议', self.protocol)
        for key, label in [('base_url', '网关地址'), ('api_key', 'API 密钥')]:
            field = QLineEdit(editor.settings.value('ai/' + key, '', str))
            if key == 'api_key':
                field.setEchoMode(QLineEdit.Password)
            self.fields[key] = field
            form.addRow(label, field)
        box.addLayout(form)
        box.addWidget(QLabel('补充提示词'))
        saved_prompt = editor.settings.value('ai/system_prompt', '', str)
        if saved_prompt.startswith('你是 FlowPhi 的编辑助手。') or saved_prompt == SYSTEM_PROMPT:
            saved_prompt = ''
        self.system = QPlainTextEdit(saved_prompt)
        self.system.setPlaceholderText('可补充自己的编辑偏好；内置工具和判定说明会自动提供。')
        box.addWidget(self.system, 1)
        note = QLabel('发送对话文字及当前工程的相关动作和设置。密钥仅保存在本机连接设置中。推理强度需模型支持。')
        note.setWordWrap(True)
        box.addWidget(note)
        save = QPushButton('保存连接与提示词')
        save.clicked.connect(self.save_connection)
        box.addWidget(save)
        tabs.addTab(connection, '连接')

        self.history = EditHistory(editor)
        tabs.addTab(self.history, '修改记录')

        versions = QWidget()
        box = QVBoxLayout(versions)
        self.versions = QListWidget()
        box.addWidget(self.versions, 1)
        row = QHBoxLayout()
        self.create_version = QPushButton('保存版本')
        self.create_version.clicked.connect(self.manual_version)
        self.restore = QPushButton('恢复所选版本')
        self.restore.clicked.connect(self.restore_selected)
        row.addWidget(self.create_version)
        row.addWidget(self.restore)
        box.addLayout(row)
        self.versions_tab = versions
        tabs.addTab(versions, '版本')
        tabs.currentChanged.connect(self.tab_changed)
        self.context_timer = QTimer(self)
        self.context_timer.timeout.connect(self.refresh_context)
        self.context_timer.start(500)
        self.refresh_context()

    def config(self):
        return {key: field.text().strip() for key, field in self.fields.items()} | dict(
            protocol=self.protocol.currentData(), reasoning_effort=self.effort.currentData(), system_prompt=self.system.toPlainText())

    def save_connection(self):
        for key, value in self.config().items():
            self.editor.settings.setValue('ai/' + key, value)
        self.editor.settings.sync()
        if self.editor.settings.status() != self.editor.settings.Status.NoError:
            raise OSError('无法保存模型连接设置')

    def history_path(self):
        return versions_folder(self.editor).parent.parent / 'model-conversations' / (versions_folder(self.editor).name + '.json')

    def save_history(self, path=None):
        if self.identity:
            try:
                write_json_atomic(path or self.identity, dict(messages=self.messages, entries=self.transcript.entries,
                    pending=self.pending, usage=self.usage, context_tokens=self.context_tokens))
            except (OSError, ValueError) as exc:
                self.status.setText('对话保存失败：' + str(exc))
                return False
        return True

    def archive_history(self):
        if self.identity and (self.messages or self.transcript.entries or self.pending):
            path = self.identity.parent / self.identity.stem / f'{time.time_ns()}.json'
            if not self.save_history(path):
                raise OSError('旧对话归档失败，原记录已保留')

    def refresh_context(self):
        path = self.history_path()
        if path != self.identity:
            self.cancel()
            self.save_history()
            self.identity = path
            self.messages, self.pending, self.usage = [], [], {}
            self.context_tokens = 0
            self.transcript.reset()
            self.version_cache.clear()
            try:
                if path.exists() and path.stat().st_size <= 128 * 1024 * 1024:
                    data = json.loads(path.read_text(encoding='utf-8'))
                    self.messages, self.usage = data['messages'], data.get('usage', {})
                    # Old conversations can contain attachments; retain only their text.
                    for message in self.messages:
                        if message.get('role') == 'user' and isinstance(message.get('content'), list):
                            message['content'] = '\n'.join(b.get('text', '') for b in message['content'] if b.get('type') == 'text')
                    self.pending = data.get('pending', [])
                    self.context_tokens = data.get('context_tokens', estimate_tokens(self.messages))
                    self.transcript.reset(data.get('entries', conversation_entries(self.messages)))
                    if self.pending:
                        self.status.setText(f'有 {len(self.pending)} 条要求待处理，点击发送可继续')
            except (OSError, ValueError, KeyError, TypeError):
                self.status.setText('无法读取上次对话，可开始新对话')
            self.show_usage()
        selected = len(self.editor.selected_ids)
        self.selection.setText(f'已选 {selected} 段动作 · 范围 {self.editor.range_start.value():g}–{self.editor.range_end.value():g} 秒')
        if self.tabs.currentWidget() is self.history:
            self.history.refresh()

    def show_usage(self):
        inp = self.usage.get('input_tokens', self.usage.get('prompt_tokens', 0))
        inp += self.usage.get('cache_read_input_tokens', 0) + self.usage.get('cache_creation_input_tokens', 0)
        out = self.usage.get('output_tokens', self.usage.get('completion_tokens', 0))
        count = inp + out if self.usage else self.context_tokens
        self.usage_bar.setText(f'上下文已用 {"约 " if not self.usage and count else ""}{count / 1000:.1f}K')
        self.usage_bar.setToolTip('模型当前携带的上下文，含系统说明、相关编辑器数据和对话；不是累计计费量。可点击“压缩上下文”手动整理。未返回用量时为估算。')

    def say(self, text, role='notice', **kwargs):
        return self.transcript.add(role, text, **kwargs)

    def toggle_floating(self):
        dock = self.editor.model_dock
        dock.setFloating(not dock.isFloating())
        if dock.isFloating():
            dock.resize(650, 800)

    def floating_changed(self, floating):
        self.detach.setText('停靠侧栏' if floating else '独立窗口')

    def tab_changed(self, *_):
        if self.tabs.currentWidget() is self.versions_tab:
            self.refresh_versions()
        elif self.tabs.currentWidget() is self.history:
            self.history.refresh()

    def state(self):
        doc = self.editor.document
        preferences = current_preferences(self.editor)
        for key in ('range_start', 'range_end'):
            preferences.pop(key)
        return (id(doc), doc.revision if doc else None, preferences)

    def set_active(self, active):
        self.active = active
        self.stop.setEnabled(active)
        self.compact.setEnabled(not active)
        self.restore.setEnabled(not active)
        self.send.setText('补充要求' if active else '发送')

    def submit(self):
        text = self.prompt.toPlainText().strip()
        if not text and not self.pending:
            return
        self.refresh_context()
        music_time = self.editor.music_time
        scope = (dict(ids=[i + 1 for i in sorted(self.editor.selected_ids)]) if self.editor.selected_ids else
                 dict(time_range=[self.editor.range_start.value(), self.editor.range_end.value()]))
        if self.active:
            if text:
                self.pending.append(dict(text=text, scope=scope, music_time=music_time))
                self.say(text, 'user')
                self.prompt.clear()
                self.status.setText(f'已收到 {len(self.pending)} 条补充要求，当前步骤完成后处理')
                self.save_history()
            return
        if self.jobs.running:
            self.status.setText('正在结束上一次计算，输入内容已保留，请稍后发送')
            return
        try:
            self.save_connection()
            config = self.config()
            build_request(config, [])
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self.active_config = config
        self.scope = scope
        self.flush_pending()
        if text:
            self.scope = scope
            self.messages.append(dict(role='user', content=text, _scope=scope, _time=music_time))
            self.say(text, 'user')
        self.prompt.clear()
        self.set_active(True)
        self.start_request()

    def flush_pending(self):
        for message in self.pending:
            self.messages.append(dict(role='user', content=message['text'], _scope=message['scope'], _time=message.get('music_time')))
            self.scope = message['scope']
        self.pending.clear()

    def compact_context(self):
        if self.active or not self.messages:
            return
        self.refresh_context()
        if self.jobs.running or not self.messages:
            return
        try:
            self.save_connection()
            self.active_config = self.config()
            build_request(self.active_config, [])
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self.scope = next((m['_scope'] for m in reversed(self.messages) if '_scope' in m), {})
        self.set_active(True)
        self.start_request(compact=True)

    def start_request(self, *, compact=False):
        if not self.active:
            return
        try:
            self.expected = self.state()
            context = editor_context(self.editor, self.scope)
            context['request_scope'] = self.scope
            endpoint, headers, body = build_request(self.active_config, self.messages, context=context, stream=True)
            self.context_tokens = estimate_tokens(body)
            self.compacting = compact
            if compact:
                self.compact_before = self.context_tokens
                self.compact_tail = compaction_tail(self.messages)
                endpoint, headers, body = build_request(self.active_config, compaction_input(self.messages), stream=True, compact=True)
            self.usage = {}
            self.show_usage()
            request = QNetworkRequest(QUrl(endpoint))
            request.setAttribute(QNetworkRequest.RedirectPolicyAttribute, QNetworkRequest.ManualRedirectPolicy)
            request.setTransferTimeout(180_000)
            for key, value in headers.items():
                request.setRawHeader(key.encode(), value.encode())
            self.stream = StreamReply(self.active_config['protocol'])
            self.raw = b''
            self.reply_entry = None
            self.thinking_entry = self.say('正在压缩上下文…' if compact else '正在思考…', 'thinking',
                detail='整理关键要求和修改进度，聊天记录会保留。' if compact else '模型尚未返回可显示的思考内容。')
            self.current_call = None
            self.calls, self.results = [], []
            self.reply = self.network.post(request, QByteArray(body))
            reply = self.reply
            self.reply.readyRead.connect(lambda: self.read_reply() if self.reply is reply else None)
            self.reply.finished.connect(lambda: self.finished_reply() if self.reply is reply else None)
            self.status.setText('正在压缩上下文…可继续补充要求' if compact else '正在思考…可继续选中动作、编辑或发送补充要求')
            self.save_history()
        except Exception as exc:
            self.fail(exc)

    def read_reply(self):
        if self.reply is None:
            return
        chunk = bytes(self.reply.readAll())
        try:
            content_type = bytes(self.reply.rawHeader('Content-Type')).decode('ascii', errors='ignore')
            status = self.reply.attribute(QNetworkRequest.HttpStatusCodeAttribute) or 0
            if 'text/event-stream' in content_type and status == 200:
                delta = self.stream.feed(chunk)
                if self.stream.reasoning and not self.compacting:
                    self.transcript.update_entry(self.thinking_entry, text='正在思考 · 可展开', detail=self.stream.reasoning)
                if delta and not self.compacting:
                    if self.reply_entry is None:
                        self.reply_entry = self.say('', 'assistant')
                    self.transcript.update_entry(self.reply_entry, text=self.stream.text)
                    self.status.setText('正在回复…')
            else:
                self.raw += chunk
                if len(self.raw) > 32 * 1024 * 1024:
                    raise ValueError('网关响应超过 32 MiB')
        except Exception as exc:
            self.fail(exc)

    def finished_reply(self):
        reply = self.reply
        if reply is None:
            return
        self.read_reply()
        if self.reply is None:
            return
        self.reply = None
        reply.deleteLater()
        try:
            status = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute) or 0
            if reply.error() != QNetworkReply.NoError or status != 200:
                detail = self.raw[:2000].decode('utf-8', errors='replace') or reply.errorString()
                raise ValueError(f'请求失败（HTTP {status}）：{detail}')
            result = parse_reply(self.active_config['protocol'], json.loads(self.raw)) if self.raw else self.stream.finish()
            if self.compacting:
                self.finish_compaction(result)
                return
            self.messages.append(result['message'])
            self.transcript.update_entry(self.thinking_entry,
                text='思考完成 · 可展开' if result['reasoning'] else '思考完成 · 未提供详细过程',
                detail=result['reasoning'] or '模型服务没有返回可显示的思考内容。')
            if result['text']:
                if self.reply_entry is None:
                    self.reply_entry = self.say(result['text'], 'assistant')
                else:
                    self.transcript.update_entry(self.reply_entry, text=result['text'])
            self.usage = result['usage']
            self.show_usage()
            self.calls = list(result['calls'])
            self.results = []
            self.run_next_tool()
        except Exception as exc:
            self.fail(exc)

    def finish_compaction(self, result):
        summary = result['text'].strip()
        if not summary or result['calls']:
            raise ValueError('未收到有效的交接记忆，原上下文已保留')
        messages = [dict(role='user', _summary=True,
            content='此前对话的交接记忆（历史记录；具体动作以当前工程为准，接着完成未完成的要求）：\n' + summary), *self.compact_tail]
        context = editor_context(self.editor, self.scope)
        context['request_scope'] = self.scope
        _, _, body = build_request(self.active_config, messages, context=context, stream=True)
        size = estimate_tokens(body)
        if estimate_tokens(wire_messages(messages, self.active_config['protocol'])) >= estimate_tokens(wire_messages(self.messages, self.active_config['protocol'])):
            raise ValueError('本次整理未缩小上下文，原记录已保留')
        self.archive_history()
        self.messages = messages
        self.compacting = False
        self.usage = {}
        self.context_tokens = size
        self.transcript.update_entry(self.thinking_entry,
            text=f'上下文已压缩 · 约 {self.compact_before / 1000:.1f}K → {size / 1000:.1f}K', detail=summary)
        self.show_usage()
        self.save_history()
        if self.pending:
            self.flush_pending()
            self.start_request()
        else:
            self.set_active(False)
            self.status.setText('上下文已压缩，可以继续对话')

    def arguments(self, call):
        args = json.loads(call['arguments']) if isinstance(call['arguments'], str) else copy.deepcopy(call['arguments'])
        if not isinstance(args, dict):
            raise ValueError('工具参数必须是对象')
        def scope_default(value):
            scope = value.get('scope') or {}
            if not any(k in scope for k in ('ids', 'all', 'time_range')):
                value['scope'] = self.scope | scope
        if call['name'] in ('get_contacts', 'run_python', 'reassign_contacts'):
            scope_default(args)
        elif call['name'] == 'edit_editor':
            for operation in args.get('operations', []):
                scope_default(operation)
        return args

    def stale(self):
        return self.state() != self.expected

    def run_next_tool(self):
        if not self.active:
            return
        if self.pending and self.calls:
            # Preserve one result per call before inserting the new user messages.
            for call in self.calls:
                self.results.append((call['id'], dict(error='用户补充了新要求，本步骤尚未执行；请结合新要求重新安排')))
            self.say('已收到补充要求，接下来的步骤会按新要求安排', 'tool',
                     detail='尚未开始的操作已交回模型重新考虑；正在执行的步骤已结束。')
            self.calls.clear()
        if not self.calls:
            if self.results:
                self.messages.extend(tool_results(self.active_config['protocol'], self.results))
                self.results = []
                self.flush_pending()
                self.start_request()
            elif self.pending:
                self.flush_pending()
                self.start_request()
            else:
                self.set_active(False)
                self.status.setText('完成')
                self.save_history()
            return
        call = self.calls.pop(0)
        self.current_call = call
        self.tool_arguments = call['arguments']
        self.tool_entry = self.say(TOOL_NAMES.get(call['name'], '处理动作') + '…', 'tool', detail='正在执行')
        try:
            if self.stale():
                self.finish_tool(dict(error='stale：编辑器已改变，请按刷新后的上下文重新读取并计算。'))
                return
            args = self.arguments(call)
            name = call['name']
            self.status.setText(TOOL_NAMES.get(name, '处理动作') + '…')
            self.tool_arguments = args
            if name == 'run_python':
                self.jobs.start('model_python', code=args['code'], data=python_data(self.editor, args.get('scope')))
                # Script execution remains bounded; reassignment is cancellable without a fixed deadline.
                self.timeout.start(60_000)
            elif name == 'reassign_contacts':
                self.candidate, ids = reassign_candidate(self.editor, **args)
                prefs = current_preferences(self.editor)
                self.jobs.start('replan', self.candidate, selected=ids, hands=args.get('hands'), fingers=args.get('fingers'),
                                beam_width=prefs['replan_beam'], max_contacts=max(prefs['replan_contacts'], len(self.candidate)))
            else:
                self.finish_tool(execute_tool(self.editor, name, args))
        except Exception as exc:
            self.finish_tool(dict(error=str(exc)))

    def job_finished(self, message):
        self.timeout.stop()
        if not self.active:
            return
        try:
            if self.stale():
                raise ValueError('stale：计算期间编辑器已改变，结果未应用；请重新计算')
            if not message.get('ok'):
                raise ValueError(message.get('error') or '计算已取消或超过 60 秒')
            result = message['result']
            if self.current_call['name'] == 'run_python':
                args = dict(settings=result['settings'], operations=result['operations'])
                args = self.arguments(dict(name='edit_editor', arguments=args))
                applied = edit_editor(self.editor, **args)
                applied.update(result=result['result'], stdout=result['stdout'])
            else:
                self.candidate.accept_assignments(result['assignments'])
                if self.candidate.snapshot() != self.editor.document.snapshot():
                    save_version(self.editor, '模型安排手指前')
                    self.editor._changed(self.editor.document.replace_plan(self.candidate, '模型安排手指'))
                applied = result
            self.finish_tool(applied)
        except Exception as exc:
            self.finish_tool(dict(error=str(exc)))

    def finish_tool(self, result):
        self.results.append((self.current_call['id'], result))
        title = TOOL_NAMES.get(self.current_call['name'], '处理动作')
        if result.get('error'):
            title += ' · ' + error_summary(result['error'])
        elif result.get('changed') or result.get('changed_contacts'):
            count = len(result.get('contacts', [])) or result.get('changed_contacts', 0)
            title += f' · 已修改 {count} 段动作' if count else ' · 设置已更新'
        elif 'total' in result:
            title += f" · 已读取 {len(result['contacts'])} / {result['total']} 段"
        else:
            title += ' · 完成'
        detail = dict(tool=self.current_call['name'], arguments=getattr(self, 'tool_arguments', self.current_call['arguments']), result=result)
        self.transcript.update_entry(self.tool_entry, text=title, detail='```json\n' + json.dumps(detail, ensure_ascii=False, indent=2) + '\n```')
        if self.tabs.currentWidget() is self.history:
            self.history.refresh()
        elif self.tabs.currentWidget() is self.versions_tab and (result.get('changed') or result.get('changed_contacts')):
            self.refresh_versions()
        # Stale siblings must also be rejected; only successful tools advance the revision.
        if not result.get('error'):
            self.expected = self.state()
        generation = self.generation
        QTimer.singleShot(0, lambda: self.run_next_tool() if generation == self.generation else None)

    def calculation_timeout(self):
        self.status.setText('计算超过 60 秒，正在停止')
        self.jobs.cancel()

    def cancel(self):
        self.generation += 1
        was_active = self.active
        self.set_active(False)
        self.compacting = False
        self.timeout.stop()
        if self.reply is not None:
            reply, self.reply = self.reply, None
            reply.abort()
            reply.deleteLater()
        if self.jobs.running:
            self.jobs.cancel()
        if was_active:
            done = {identity for identity, _ in getattr(self, 'results', [])}
            pending = getattr(self, 'results', []) + [(c['id'], dict(error='用户已停止，未执行'))
                for c in ([self.current_call] if getattr(self, 'current_call', None) and self.current_call['id'] not in done else []) + self.calls
                if c['id'] not in done]
            if pending and self.messages and self.messages[-1].get('tool_calls'):
                self.messages.extend(tool_results(self.active_config['protocol'], pending))
            self.calls = []
            if hasattr(self, 'thinking_entry') and self.reply_entry is None and not self.current_call:
                self.transcript.update_entry(self.thinking_entry, text='已停止思考')
            if self.current_call and self.current_call['id'] not in done:
                self.transcript.update_entry(self.tool_entry, text='操作已停止 · 未应用未完成的计算')
            self.status.setText('已停止；已经应用的修改可撤销或恢复版本')
            self.save_history()

    def fail(self, error):
        key = getattr(self, 'active_config', {}).get('api_key', '')
        text = str(error).replace(key, '[密钥]') if key else str(error)
        self.cancel()
        self.say('这次未完成 · 展开查看原因', detail=text)
        self.status.setText('未完成，可以继续发送')
        self.save_history()

    def new_conversation(self):
        self.cancel()
        try:
            self.archive_history()
        except OSError as exc:
            self.status.setText(str(exc))
            return
        self.messages.clear()
        self.pending.clear()
        self.transcript.reset()
        self.usage = {}
        self.context_tokens = 0
        self.show_usage()
        self.save_history()

    def refresh_versions(self):
        if self.version_thread is not None:
            self.versions_dirty = True
            return
        self.versions_dirty = False
        folder, cached = versions_folder(self.editor), self.version_cache.copy()
        def read_versions():
            result = {}
            for path in folder.glob('*.json'):
                try:
                    stamp = path.stat().st_mtime_ns
                    if path in cached and cached[path][0] == stamp:
                        result[path] = cached[path]
                    else:
                        data = json.loads(path.read_text(encoding='utf-8'))
                        result[path] = (stamp, data['created'].replace('T', ' ') + '  ' + data['label'])
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            return folder, result
        thread = self.version_thread = AsyncCall('versions', read_versions, self)
        thread.succeeded.connect(self.versions_loaded)
        thread.failed.connect(lambda _: self.status.setText('版本列表读取失败，可重新打开“版本”重试'))
        thread.finished.connect(self.versions_finished)
        thread.start()

    def versions_loaded(self, message):
        folder, result = message['value']
        if folder != versions_folder(self.editor):
            return
        selected = self.versions.currentItem()
        selected = selected.data(Qt.UserRole) if selected else None
        self.version_cache = result
        self.versions.clear()
        for path, (_, label) in sorted(result.items(), reverse=True):
            self.versions.addItem(label)
            item = self.versions.item(self.versions.count() - 1)
            item.setData(Qt.UserRole, str(path))
            if str(path) == selected:
                self.versions.setCurrentItem(item)

    def versions_finished(self):
        self.version_thread.deleteLater()
        self.version_thread = None
        if self.versions_dirty:
            self.refresh_versions()

    def manual_version(self):
        label, ok = QInputDialog.getText(self, '保存版本', '版本名称')
        if ok:
            try:
                save_version(self.editor, label.strip() or '手动保存')
                self.refresh_versions()
            except OSError as exc:
                self.status.setText(str(exc))

    def restore_selected(self):
        item = self.versions.currentItem()
        if item and not self.active:
            try:
                label = restore_version(self.editor, item.data(Qt.UserRole))
                self.refresh_versions()
                self.say('已恢复版本：' + label)
                self.messages.append(dict(role='user', content='编辑器已恢复版本：' + label + '。后续请以当前数据为准。'))
                self.save_history()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.status.setText(str(exc))
