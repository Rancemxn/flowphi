from __future__ import annotations

from html import escape
import json
import re

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QColor, QTextCursor, QTextDocument, QTextFrameFormat
from PySide6.QtWidgets import QTextBrowser


TOOL_NAMES = {'get_contacts': '读取动作', 'edit_editor': '修改方案', 'run_python': '计算并整理动作',
              'reassign_contacts': '重新安排手指', 'sample_motion': '查看手部位置'}
TIME = re.compile(r'(?<![\w.:])(?:\d{1,3}:[0-5]\d(?:\.\d{1,3})?|'
                  r'\d+(?:\.\d+)?(?=\s*(?:秒|s\b))|'
                  r'\d+(?:\.\d+)?(?=\s*[–—~～至-]\s*\d+(?:\.\d+)?\s*(?:秒|s\b)))(?![\d.:])')


class TextDocument(QTextDocument):
    def loadResource(self, *_):
        return None


def markdown_html(text):
    doc = TextDocument()
    doc.setMarkdown(text, QTextDocument.MarkdownDialectGitHub | QTextDocument.MarkdownNoHTML)
    block = doc.begin()
    while block.isValid():
        block_text = block.text()
        for match in TIME.finditer(block_text):
            cursor = QTextCursor(doc)
            # QTextCursor positions count UTF-16 units, including both halves of emoji.
            cursor.setPosition(block.position() + len(block_text[:match.start()].encode('utf-16-le')) // 2)
            cursor.setPosition(block.position() + len(block_text[:match.end()].encode('utf-16-le')) // 2, QTextCursor.KeepAnchor)
            fmt = cursor.charFormat()
            if fmt.isAnchor() or fmt.fontFixedPitch():
                continue
            value = match.group()
            seconds = sum(float(part) * 60 ** i for i, part in enumerate(reversed(value.split(':'))))
            fmt.setAnchor(True)
            fmt.setAnchorHref(f'time:{seconds:g}')
            fmt.setForeground(QColor('#79baff'))
            fmt.setFontUnderline(True)
            cursor.mergeCharFormat(fmt)
        block = block.next()
    html = doc.toHtml()
    return html[html.index('<body'):].split('>', 1)[1].rsplit('</body>', 1)[0]


def error_summary(error):
    if 'stale' in error:
        return '方案已更新，本次结果未应用'
    if '未找到' in error or '没有找到' in error or '可用指法' in error:
        return '暂未排出合适的指法'
    if '接续' in error or '持续按住' in error or '继续按住' in error:
        return '有一段动作没接上前面的触点'
    if '锁定' in error:
        return '部分动作已锁定，尚未调整'
    return '这一步未完成'


class ChatView(QTextBrowser):
    seek = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries = []
        self.fragments = {}
        self.frames = []
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self.open_link)
        self.setStyleSheet('QTextBrowser {background:#111b29; color:#e2eaf5; border:1px solid #304158; padding:8px;}')
        self.setAccessibleName('对话：你的消息与模型回复分开显示；时间可点击，过程可展开')
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(120)
        self.render_timer.timeout.connect(self.render)

    def loadResource(self, *_):
        return None

    def add(self, role, text='', *, detail=None):
        index = len(self.entries)
        entry = dict(role=role, text=text)
        if detail is not None:
            entry.update(detail=detail, expanded=False)
        self.entries.append(entry)
        self.schedule()
        return index

    def update_entry(self, index, **values):
        if index is None or not 0 <= index < len(self.entries):
            return
        entry = self.entries[index]
        visible_change = any(entry.get(k) != v for k, v in values.items() if k != 'detail' or entry.get('expanded'))
        entry.update(values)
        if visible_change:
            self.fragments.pop(index, None)
            self.schedule()

    def schedule(self):
        if not self.render_timer.isActive():
            self.render_timer.start()

    def reset(self, entries=()):
        self.entries = list(entries)
        self.fragments.clear()
        self.frames.clear()
        self.clear()
        self.render()

    def open_link(self, url):
        if url.scheme() == 'time':
            try:
                self.seek.emit(float(url.path()))
            except ValueError:
                pass
        elif url.scheme() == 'detail':
            try:
                index = int(url.path())
                self.update_entry(index, expanded=not self.entries[index].get('expanded'))
            except (ValueError, IndexError):
                pass

    def render(self):
        self.render_timer.stop()
        bar = self.verticalScrollBar()
        position, at_bottom = bar.value(), bar.maximum() - bar.value() < 24
        for index, entry in enumerate(self.entries):
            if index not in self.fragments:
                role = entry['role']
                label, color = {'user': ('你', '#203d59'), 'assistant': ('模型', '#182a3c'),
                                'thinking': ('思考', '#1c2837'), 'tool': ('操作', '#1c2837'),
                                'notice': ('提示', '#352d28')}.get(role, ('记录', '#1c2837'))
                body = markdown_html(entry['text']) if role == 'assistant' else escape(entry['text']).replace('\n', '<br>')
                if 'detail' in entry:
                    body = f'<a href="detail:{index}" style="color:#acbdd4">' + ('▾ ' if entry.get('expanded') else '▸ ') + body + '</a>'
                    if entry.get('expanded'):
                        body += '<hr>' + markdown_html(entry['detail'])
                self.fragments[index] = (f'<table width="100%" cellspacing="0" cellpadding="10" bgcolor="{color}">'
                    f'<tr><td><b>{label}</b><br>{body}</td></tr></table><p style="margin:4px"></p>')
                if index == len(self.frames):
                    cursor = QTextCursor(self.document())
                    cursor.movePosition(QTextCursor.End)
                    fmt = QTextFrameFormat()
                    fmt.setLeftMargin(24 if role == 'user' else 0)
                    fmt.setRightMargin(0 if role == 'user' else 8)
                    self.frames.append(cursor.insertFrame(fmt))
                frame = self.frames[index]
                cursor = frame.firstCursorPosition()
                cursor.setPosition(frame.lastPosition(), QTextCursor.KeepAnchor)
                cursor.insertHtml(self.fragments[index])
        bar.setValue(bar.maximum() if at_bottom else position)


def conversation_entries(messages):
    entries, calls = [], {}
    for message in messages:
        role = message.get('role')
        content = message.get('content') or ''
        if isinstance(content, list):
            content = '\n'.join(b.get('text', '') for b in content if b.get('type') == 'text')
        if role == 'tool':
            name = calls.get(message.get('tool_call_id'), '操作')
            try:
                result = json.loads(content)
                summary = error_summary(result['error']) if result.get('error') else '已记录'
            except (ValueError, TypeError, AttributeError):
                summary = '已记录'
            entries.append(dict(role='tool', text=TOOL_NAMES.get(name, '操作') + ' · ' + summary, detail=content, expanded=False))
        else:
            reasoning = message.get('reasoning_content') or '\n'.join(
                b.get('thinking', '') for b in message.get('_anthropic', []) if b.get('type') == 'thinking')
            reasoning += '\n'.join(c.get('text', '') for b in message.get('_responses', [])
                                   if b.get('type') == 'reasoning' for c in b.get('summary', []))
            if reasoning:
                entries.append(dict(role='thinking', text='思考完成 · 可展开', detail=reasoning, expanded=False))
            if content:
                entries.append(dict(role=role, text=content))
            for call in message.get('tool_calls', []):
                calls[call['id']] = call['function']['name']
    return entries
