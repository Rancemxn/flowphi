from __future__ import annotations

import copy
import math
import time

import skia
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QToolTip, QWidget

from flowphi.ui.model import TimeIndex
from flowphi.ui.contact_state import resolve_contacts
from flowphi.motion import FINGER_ORDER, point_at
from flowphi.rendering.preview import ChartRenderer

FINGER_NAMES = {'index': '食指', 'middle': '中指', 'ring': '无名指', 'thumb': '拇指', 'little': '小指'}
HAND_NAMES = {'left': '左', 'right': '右'}
KIND_NAMES = {'tap': 'Tap', 'hold': 'Hold', 'drag': 'Drag', 'flick': 'Flick'}
COLORS = {'left': ['#64afff', '#53cfdd', '#65d8b1', '#a99cf5', '#7e9ed9'],
          'right': ['#ff9980', '#f8bc6b', '#ead476', '#ec9cd1', '#ed929d']}


def finger_name(c):
    return HAND_NAMES[c['hand']] + FINGER_NAMES[c['finger']]


def contact_color(c):
    return QColor(COLORS[c['hand']][FINGER_ORDER.index(c['finger'])])


def contact_label(index, c, offset=0.):
    notes = ', '.join(map(str, c['note_ids'][:6]))
    if len(c['note_ids']) > 6:
        notes += f"…（共 {len(c['note_ids'])} 个）"
    mode = dict(auto='自动接续', press='重新按下', **{'continue': '继续按住'})[c.get('touch_mode', 'auto')]
    return (f"动作 #{index + 1} · {finger_name(c)} · {KIND_NAMES.get(c['kind'], c['kind'])} · {mode} · 音符 {notes}"
            f" · {c['start'] + offset:.3f}–{c['end'] + offset:.3f} s")


def choose_contact(parent, document, indices, offset, global_position):
    ids = document.sorted_ids(indices)
    if len(ids) == 1:
        return ids[0]
    if not ids:
        return None
    menu = QMenu(parent)
    menu.addSection('这里有多个动作，请选择')
    for i in ids:
        action = menu.addAction(contact_label(i, document.contact(i), offset))
        action.setData(i)
    selected = menu.exec(global_position)
    result = selected.data() if selected is not None else None
    menu.deleteLater()
    return result


class ContactTableModel(QAbstractTableModel):
    HEADERS = ('按下 / s', '关联音符', '类型', '手指', '锁定')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document = None
        self.offset = 0.

    def set_document(self, document, offset=0.):
        self.beginResetModel()
        self.document, self.offset = document, offset
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or self.document is None else len(self.document)

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        if role == Qt.DisplayRole and orientation == Qt.Vertical:
            return f'#{section + 1}'
        if role == Qt.ToolTipRole and orientation == Qt.Vertical:
            return '动作编号，与时间轴和问题列表中的编号一致'
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self.document is None:
            return None
        c = self.document.contact(index.row())
        if role == Qt.DisplayRole:
            ids = ', '.join(map(str, c['note_ids'][:4])) + ('…' if len(c['note_ids']) > 4 else '')
            return (f"{c['start'] + self.offset:.3f}", ids, KIND_NAMES.get(c['kind'], c['kind']), finger_name(c),
                    ('指法 ' if c['assignment_locked'] else '') + ('轨迹' if c['path_locked'] else ''))[index.column()]
        if role == Qt.ForegroundRole and index.column() == 3:
            return contact_color(c)
        if role == Qt.ToolTipRole:
            return contact_label(index.row(), c, self.offset)
        return None

    def changed(self, indices):
        ordered = sorted(set(indices))
        if not ordered:
            return
        low = previous = ordered[0]
        for row in ordered[1:] + [None]:
            if row is not None and row == previous + 1:
                previous = row
                continue
            self.dataChanged.emit(self.index(low, 0), self.index(previous, len(self.HEADERS) - 1))
            if row is not None:
                low = previous = row


class PathTableModel(QAbstractTableModel):
    edited = Signal(object)
    invalid = Signal(str)
    HEADERS = ('音乐时间 / s', '横向 X / %', '纵向 Y / %')
    TIPS = ('音乐播放到第几秒时经过该点第一个点的时间就是按下时刻，不能在此修改',
            '横向位置：0% 在屏幕最左侧，100% 在最右侧',
            '纵向位置：0% 在屏幕最上方，100% 在最下方')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.offset = 0.

    def set_points(self, points, offset=0.):
        self.beginResetModel()
        self.points, self.offset = points, offset
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.points)

    def columnCount(self, parent=QModelIndex()):
        return 3

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        if role == Qt.ToolTipRole and orientation == Qt.Horizontal:
            return self.TIPS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role in (Qt.DisplayRole, Qt.EditRole):
            value = self.points[index.row()][index.column()]
            return f'{value + self.offset:.3f}' if index.column() == 0 else f'{value * 100:.3f}'
        return None

    def flags(self, index):
        result = super().flags(index)
        if index.isValid() and not (index.row() == 0 and index.column() == 0):
            result |= Qt.ItemIsEditable
        return result

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid():
            return False
        try:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError('请输入有效数字，例如 0 或 0.5')
            points = [list(p) for p in self.points]
            points[index.row()][index.column()] = round((number - self.offset) * 1000) / 1000 if index.column() == 0 else number / 100
            self.edited.emit(points)
            return True
        except (TypeError, ValueError):
            self.invalid.emit('请输入有效数字，例如 0 或 0.5')
            return False
        except OverflowError:
            self.invalid.emit('数值过大，请输入较小的数值')
            return False


class IndexedPreview:
    def __init__(self, chart, width=960):
        view = copy.copy(chart)
        view.lines = [copy.copy(line) for line in chart.lines]
        self.visuals = [list(line.visual_notes) for line in chart.lines]
        self.indices = [TimeIndex((v.note.seconds, v.note.seconds + max(v.note.hold, .08), i)
                                 for i, v in enumerate(rows)) for rows in self.visuals]
        self.renderer = ChartRenderer(view, width, max(1, round(width * chart.height / chart.width)))
        self.lines = view.lines
        self.note_ids = {id(note): i for i, note in enumerate(n for line in chart.lines for n in line.notes)}
        self.visible = []
        self.surface = skia.Surface(self.renderer.width, self.renderer.height)
        self.cached_key = None
        self.cached_image = None
        self.last_ms = 0.
        self.block_signature = None
        self.block_path = None

    def frame(self, t, lookahead):
        key = round(t, 6), lookahead
        if self.cached_key == key:
            return self.cached_image
        start = time.perf_counter()
        self.visible = []
        for line, rows, index in zip(self.lines, self.visuals, self.indices):
            line.visual_notes = [rows[i] for i in index.query(t - .1, t + lookahead)]
            self.visible.extend((line, v) for v in line.visual_notes)
        canvas = self.surface.getCanvas()
        canvas.clear(skia.Color(16, 23, 35))
        self.renderer.draw(canvas, t)
        blocks = self.renderer.chart.block_areas
        if blocks:
            signature = blocks.active(t)
            if signature != self.block_signature:
                area = blocks.forbidden(t)
                parts = [area] if area.geom_type == 'Polygon' else getattr(area, 'geoms', [])
                path = skia.Path()
                path.setFillType(skia.PathFillType.kEvenOdd)
                for polygon in parts:
                    if polygon.geom_type != 'Polygon':
                        continue
                    for ring in [polygon.exterior, *polygon.interiors]:
                        for i, (x, y) in enumerate(ring.coords):
                            point = x / blocks.width * self.renderer.width, y / blocks.height * self.renderer.height
                            path.moveTo(*point) if i == 0 else path.lineTo(*point)
                        path.close()
                self.block_path, self.block_signature = path, signature
            if self.block_path is not None:
                canvas.drawPath(self.block_path, skia.Paint(Color=skia.ColorSetARGB(110, 116, 84, 150), AntiAlias=True))
                canvas.drawPath(self.block_path, skia.Paint(Color=skia.ColorSetARGB(210, 181, 147, 219),
                    Style=skia.Paint.kStroke_Style, StrokeWidth=1.5, AntiAlias=True))
        image = self.surface.makeImageSnapshot()
        self.cached_image = QImage(image.tobytes(), image.width(), image.height(), QImage.Format_ARGB32_Premultiplied).copy()
        self.cached_key = key
        self.last_ms = (time.perf_counter() - start) * 1000
        return self.cached_image

    def hit_note(self, x, y, t, tolerance):
        best = None
        for line, visual in self.visible:
            nid = self.note_ids.get(id(visual.note))
            if nid is None:
                continue
            state = line.note_state(t, visual)
            if state is None:
                continue
            nx = state.head.real / self.renderer.chart.width
            ny = state.head.imag / self.renderer.chart.height
            distance = math.hypot(nx - x, ny - y)
            if distance < tolerance and (best is None or distance < best[0]):
                best = distance, nid
        return best[1] if best else None


class ChartCanvas(QWidget):
    selected = Signal(object, bool)
    point_selected = Signal(int, int)
    point_edited = Signal(int, int, float, float)
    add_point = Signal(float, float)
    pause_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 235)
        self.setFocusPolicy(Qt.StrongFocus)
        self.document = self.source = self.preview = None
        self.selected_ids = set()
        self.time = 0.
        self.lookahead = 8.
        self.render_width = 960
        self.compare = False
        self.editable = True
        self.drag = None
        self.drag_whole_path = False
        self.selected_points = set()
        self.drag_rows = set()
        self.drag_origin = (0., 0.)
        self.drag_bounds = (0., 0., 0., 0.)
        self.zone = None
        self.handles = []
        self.active_point = 0
        self._touch_key = None
        self._touch_contacts = []
        self._touch_index = TimeIndex([])
        self.setToolTip('触点重叠时，右键点击可按动作编号选择；左键拖动所选轨迹的控制点')

    def set_source(self, source):
        self.source = source
        self.preview = IndexedPreview(source.chart, self.render_width) if source else None
        self.update()

    def set_document(self, document):
        self.document = document
        self.selected_ids.clear()
        self.selected_points.clear()
        self.drag = None
        self.zone = None
        self.update()

    def set_time(self, t):
        self.time = t
        if self.zone and abs(self.zone['time'] - t) > .001:
            self.zone = None
        self.update()

    def viewport(self):
        ratio = self.source.chart.width / self.source.chart.height if self.source else 16 / 9
        width = min(self.width() - 24, (self.height() - 24) * ratio)
        height = width / ratio
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def screen_point(self, x, y):
        rect = self.viewport()
        return QPointF(rect.left() + x * rect.width(), rect.top() + y * rect.height())

    def normalized(self, point):
        rect = self.viewport()
        return ((point.x() - rect.left()) / rect.width(), (point.y() - rect.top()) / rect.height())

    def displayed_point(self, contact, row, x, y):
        if self.drag and self.drag[0] == contact and (self.drag_rows is None or row in self.drag_rows):
            return x + self.drag[2] - self.drag_origin[0], y + self.drag[3] - self.drag_origin[1]
        return x, y

    def contacts_at(self, position):
        if not self.document:
            return []
        x, y = self.normalized(position)
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return []
        rect = self.viewport()
        hits = []
        if self.compare:
            rows = [(i, self.document.baseline['contacts'][i]) for i in self.document.baseline_index.query(self.time - .10, self.time + .10)]
        else:
            self._resolve_touches()
            rows = [(i, self._touch_contacts[i]) for i in self._touch_index.query(self.time - .10, self.time + .10)]
        for i, c in rows:
            px, py = point_at(c['points'], self.time)
            if math.hypot((px - x) * rect.width(), (py - y) * rect.height()) < 24:
                hits.extend([i] if self.compare else c['editor_contacts'])
        return list(dict.fromkeys(hits))

    def _resolve_touches(self):
        key = id(self.document), self.document.revision, id(self.source)
        if key != self._touch_key:
            self._touch_contacts, _, _ = resolve_contacts(self.document, self.source.chart if self.source else None)
            self._touch_index = TimeIndex((c['start'], c['end'], i) for i, c in enumerate(self._touch_contacts))
            self._touch_key = key

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#111824'))
        rect = self.viewport()
        painter.setPen(QPen(QColor('#334359'), 1))
        painter.drawRoundedRect(rect, 7, 7)
        if not self.preview:
            return
        painter.drawImage(rect, self.preview.frame(self.time, self.lookahead))
        painter.save()
        painter.setClipRect(rect)
        if self.zone and not self.compare:
            path = QPainterPath()
            path.setFillRule(Qt.OddEvenFill)
            for polygon in self.zone.get('polygons', []):
                for ring in [polygon['exterior'], *polygon['holes']]:
                    if not ring:
                        continue
                    path.moveTo(self.screen_point(*ring[0]))
                    for point in ring[1:]:
                        path.lineTo(self.screen_point(*point))
                    path.closeSubpath()
            painter.fillPath(path, QColor(84, 218, 166, 42))
            painter.setPen(QPen(QColor('#54daa6'), 1, Qt.DashLine))
            painter.drawPath(path)
        self.handles = []
        if self.document:
            for i in sorted(self.selected_ids)[:100]:
                c = self.document.baseline['contacts'][i] if self.compare else self.document.contact(i)
                points = c['points']
                # Bound drawing work; numeric editing still exposes every point.
                stride = max(1, math.ceil(len(points) / 400))
                chosen = sorted(set(range(0, len(points), stride)) | {len(points) - 1,
                                min(self.active_point, len(points) - 1)})
                path = QPainterPath()
                for n, j in enumerate(chosen):
                    _, x, y = points[j]
                    x, y = self.displayed_point(i, j, x, y)
                    pos = self.screen_point(x, y)
                    path.moveTo(pos) if n == 0 else path.lineTo(pos)
                painter.setPen(QPen(contact_color(c), 2))
                painter.drawPath(path)
                if len(self.selected_ids) == 1 and not self.compare:
                    for j in chosen:
                        _, x, y = points[j]
                        x, y = self.displayed_point(i, j, x, y)
                        pos = self.screen_point(x, y)
                        self.handles.append((i, j, pos))
                        painter.setBrush(QColor('#ffffff') if j == self.active_point or j in self.selected_points else contact_color(c))
                        painter.setPen(QPen(QColor('#111824'), 1))
                        painter.drawEllipse(pos, 5 if j == self.active_point else 3.5, 5 if j == self.active_point else 3.5)
            if self.compare:
                active = [([i], self.document.baseline['contacts'][i]) for i in self.document.baseline_index.query(self.time, self.time)]
            else:
                self._resolve_touches()
                active = [(self._touch_contacts[i]['editor_contacts'], self._touch_contacts[i])
                          for i in self._touch_index.query(self.time, self.time)]
            for ids, c in sorted(active, key=lambda row: (bool(set(row[0]) & self.selected_ids), row[0])):
                x, y = point_at(c['points'], self.time)
                pos = self.screen_point(x, y)
                color = contact_color(c)
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 100))
                painter.setPen(QPen(color, 2.5))
                painter.drawEllipse(pos, 11, 11)
                painter.setPen(QColor('#ffffff'))
                label = ' / '.join(f'#{i + 1}' for i in ids[:3]) + ('…' if len(ids) > 3 else '')
                painter.drawText(pos + QPointF(15, -8), f'{label} {finger_name(c)}')
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or not self.document:
            return
        self.setFocus()
        self.pause_requested.emit()
        if not self.compare and self.editable:
            hits = []
            for i, j, pos in self.handles:
                if i not in self.selected_ids or j >= len(self.document.contact(i)['points']):
                    continue
                distance = math.hypot(event.position().x() - pos.x(), event.position().y() - pos.y())
                if distance <= 10:
                    hits.append((distance, j != self.active_point, j not in self.selected_points, i, j))
            if hits:
                _, _, _, i, j = min(hits)
                _, x, y = self.document.contact(i)['points'][j]
                self.drag = i, j, x, y
                self.active_point = j
                self.point_selected.emit(i, j)
                self.drag_origin = (x, y)
                self.drag_rows = None if self.drag_whole_path else set(self.selected_points) | {j}
                points = self.document.contact(i)['points']
                rows = range(len(points)) if self.drag_rows is None else self.drag_rows
                selected = [points[row] for row in rows]
                self.drag_bounds = (-min(p[1] for p in selected), 1 - max(p[1] for p in selected),
                                    -min(p[2] for p in selected), 1 - max(p[2] for p in selected))
                self.update()
                return
        x, y = self.normalized(event.position())
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return
        hits = self.contacts_at(event.position())
        offset = self.source.chart.offset if self.source else 0.
        found = choose_contact(self, self.document, hits, offset, event.globalPosition().toPoint())
        if hits and found is None:
            return
        ids = {found} if found is not None else set()
        if not ids and self.preview:
            nid = self.preview.hit_note(x, y, self.time, .06)
            if nid is not None:
                ids = set(self.document.note_contacts.get(nid, []))
        if ids:
            self.selected.emit(ids, bool(event.modifiers() & Qt.ControlModifier))

    def contextMenuEvent(self, event):
        hits = self.contacts_at(event.pos())
        if not hits:
            return
        self.pause_requested.emit()
        offset = self.source.chart.offset if self.source else 0.
        found = choose_contact(self, self.document, hits, offset, event.globalPos())
        if found is not None:
            self.selected.emit({found}, False)

    def mouseMoveEvent(self, event):
        if self.drag:
            x, y = self.normalized(event.position())
            ox, oy = self.drag_origin
            low_x, high_x, low_y, high_y = self.drag_bounds
            dx = min(high_x, max(low_x, x - ox))
            dy = min(high_y, max(low_y, y - oy))
            self.drag = self.drag[:2] + (ox + dx, oy + dy)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag:
            i, j, x, y = self.drag
            self.drag = None
            self.point_edited.emit(i, j, x, y)
            self.update()

    def mouseDoubleClickEvent(self, event):
        if len(self.selected_ids) == 1 and not self.compare and self.editable:
            x, y = self.normalized(event.position())
            if 0 <= x <= 1 and 0 <= y <= 1:
                self.add_point.emit(x, y)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.drag:
            self.drag = None
            self.update()
        else:
            super().keyPressEvent(event)


class FingeringTimeline(QWidget):
    selected = Signal(object, bool)
    seek = Signal(float)
    range_selected = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(277)
        self.setMouseTracking(True)
        self.document = None
        self.selected_ids = set()
        self.offset = self.time = self.start = 0.
        self.span = 12.
        self.drag_start = self.drag_end = None
        self.rectangles = []
        self.row_bounds = []
        self.overlapping_ids = set()
        self.conflict_ids = set()
        self._layout_key = None
        self.gutter = 72

    def set_document(self, document, offset=0.):
        self.document, self.offset = document, offset
        self.start = -offset
        self.selected_ids.clear()
        self._layout_key = None
        self.update()

    def set_time(self, t, follow=False):
        self.time = t
        if follow and not (self.start + self.span * .08 <= t <= self.start + self.span * .85):
            self.start = max(-self.offset, t - self.span * .25)
        self.update()

    def x_at(self, t):
        return self.gutter + (t - self.start) / self.span * (self.width() - self.gutter - 10)

    def time_at(self, x):
        return self.start + (x - self.gutter) / max(1, self.width() - self.gutter - 10) * self.span

    def row_at(self, y):
        self._ensure_layout()
        return next((row for row, (top, bottom) in enumerate(self.row_bounds) if y < bottom), 9)

    def _ensure_layout(self):
        key = (id(self.document), self.document.revision if self.document else None,
               self.start, self.span, self.width())
        if key == self._layout_key:
            return
        self._layout_key = key
        groups = [[] for _ in range(10)]
        if self.document:
            for i in self.document.sorted_ids(self.document.index.query(self.start, self.start + self.span)):
                c = self.document.contact(i)
                row = (0 if c['hand'] == 'left' else 5) + FINGER_ORDER.index(c['finger'])
                groups[row].append((i, c))
        self.rectangles, self.row_bounds = [], []
        self.overlapping_ids = set(self.conflict_ids)
        y = 27.
        for group in groups:
            lanes, placed = [], []
            for i, c in group:
                # Separate both real overlaps and short actions whose minimum
                # clickable width would otherwise cover the preceding action.
                left = max(float(self.gutter), self.x_at(c['start']))
                right = max(left + 7., min(self.width() - 10., self.x_at(c['end'])))
                lane = next((n for n, end in enumerate(lanes) if end + 2 <= left), len(lanes))
                if lane == len(lanes):
                    lanes.append(right)
                else:
                    lanes[lane] = right
                placed.append((i, QRectF(left, y + lane * 24 + 2, right - left, 20)))
            height = max(1, len(lanes)) * 24
            self.row_bounds.append((y, y + height))
            self.rectangles.extend(placed)
            y += height
        self.setMinimumHeight(math.ceil(y + 10))

    def focus_interval(self, start, end):
        middle = (start + end) / 2
        self.span = min(120., max(.5, (end - start) * 1.6))
        self.start = max(-self.offset, middle - self.span * .5)
        self._ensure_layout()
        self.update()

    def contact_bounds(self, indices):
        self._ensure_layout()
        ids = set(indices)
        result = None
        for i, rect in self.rectangles:
            if i in ids:
                result = QRectF(rect) if result is None else result.united(rect)
        return result

    def _hits(self, position):
        self._ensure_layout()
        if position.x() < self.gutter:
            return []
        exact = [i for i, rect in self.rectangles if rect.contains(position)]
        return exact or [i for i, rect in self.rectangles if rect.adjusted(-3, -1, 3, 1).contains(position)]

    def paintEvent(self, event):
        self._ensure_layout()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#121c2a'))
        painter.setRenderHint(QPainter.Antialiasing)
        for row, (y, bottom) in enumerate(self.row_bounds):
            height = bottom - y
            if row % 2 == 0:
                painter.fillRect(QRectF(self.gutter, y, self.width() - self.gutter, height), QColor('#182332'))
            painter.setPen(QColor('#a8b8cd'))
            painter.drawText(QRectF(6, y, self.gutter - 10, height), Qt.AlignVCenter, ('左' if row < 5 else '右') + FINGER_NAMES[FINGER_ORDER[row % 5]])
            painter.setPen(QPen(QColor('#29374a'), 1))
            painter.drawLine(QPointF(self.gutter, bottom), QPointF(self.width(), bottom))
        step = next((v for v in (.1, .25, .5, 1, 2, 5, 10, 20, 30) if self.span / v <= 18), 60)
        first = math.ceil((self.start + self.offset) / step) * step
        value = first
        while value <= self.start + self.offset + self.span:
            x = self.x_at(value - self.offset)
            painter.setPen(QPen(QColor('#29374a'), 1))
            painter.drawLine(QPointF(x, 25), QPointF(x, self.height() - 7))
            painter.setPen(QColor('#94a6bf'))
            painter.drawText(QPointF(x + 3, 17), f'{value:g}')
            value += step
        painter.save()
        painter.setClipRect(QRectF(self.gutter, 25, self.width() - self.gutter, self.height() - 25))
        if self.document:
            for i, rect in self.rectangles:
                c = self.document.contact(i)
                color = contact_color(c)
                painter.setBrush(color)
                border = QColor('#ff7484') if i in self.overlapping_ids else color
                painter.setPen(QPen(QColor('#ffffff') if i in self.selected_ids else border,
                                    2 if i in self.selected_ids or i in self.overlapping_ids else 1))
                painter.drawRoundedRect(rect, 3, 3)
                if c['assignment_locked']:
                    painter.setPen(QPen(QColor('#172230'), 2))
                    painter.drawLine(rect.topLeft() + QPointF(3, 3), rect.bottomLeft() + QPointF(3, -3))
                if rect.width() > 44:
                    painter.setPen(QColor('#142132'))
                    text = f"#{i + 1} {KIND_NAMES.get(c['kind'], c['kind'])}"
                    painter.drawText(rect.adjusted(7, 0, -2, 0), Qt.AlignVCenter,
                                     painter.fontMetrics().elidedText(text, Qt.ElideRight, round(rect.width() - 9)))
        x = self.x_at(self.time)
        painter.setPen(QPen(QColor('#f4f8ff'), 1.5))
        painter.drawLine(QPointF(x, 25), QPointF(x, self.height()))
        if self.drag_start is not None and self.drag_end is not None:
            area = QRectF(self.drag_start, self.drag_end).normalized()
            painter.fillRect(area, QColor(105, 171, 255, 40))
            painter.setPen(QPen(QColor('#69abff'), 1))
            painter.drawRect(area)
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or event.position().x() < self.gutter:
            return
        self.drag_start = self.drag_end = event.position()
        QToolTip.hideText()

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            self.drag_end = event.position()
            self.update()
        elif self.document:
            hits = self._hits(event.position())
            if hits:
                text = '\n'.join(contact_label(i, self.document.contact(i), self.offset) for i in hits)
                QToolTip.showText(event.globalPosition().toPoint(), text, self)
            else:
                QToolTip.hideText()

    def mouseReleaseEvent(self, event):
        if self.drag_start is None:
            return
        start, end = self.drag_start, event.position()
        self.drag_start = self.drag_end = None
        additive = bool(event.modifiers() & Qt.ControlModifier)
        if math.hypot(end.x() - start.x(), end.y() - start.y()) < 5:
            hits = self._hits(end)
            found = choose_contact(self, self.document, hits, self.offset, event.globalPosition().toPoint()) if self.document else None
            if found is not None:
                self.selected.emit({found}, additive)
            self.seek.emit(self.time_at(end.x()))
        elif self.document:
            low, high = sorted((self.time_at(start.x()), self.time_at(end.x())))
            first, last = sorted((self.row_at(start.y()), self.row_at(end.y())))
            ids = set()
            for i in self.document.index.query(low, high):
                c = self.document.contact(i)
                row = (0 if c['hand'] == 'left' else 5) + FINGER_ORDER.index(c['finger'])
                if first <= row <= last:
                    ids.add(i)
            self.selected.emit(ids, additive)
            self.range_selected.emit(low, high)
        self.update()

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if event.position().x() < self.gutter:
            # The containing scroll area handles vertical movement over labels.
            event.ignore()
            return
        anchor = self.time_at(event.position().x())
        ratio = (anchor - self.start) / self.span
        if event.modifiers() & Qt.ControlModifier:
            self.span = min(120., max(.5, self.span * (1.25 if event.angleDelta().y() < 0 else .8)))
            self.start = max(-self.offset, anchor - ratio * self.span)
        else:
            self.start = max(-self.offset, self.start - event.angleDelta().y() / 120 * self.span * .1)
        self.update()
