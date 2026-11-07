import hashlib
import logging
from dataclasses import dataclass

import numpy as np
from fonticon_fa6 import FA6S
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QCursor
from PyQt6.QtWidgets import QMenu, QMessageBox, QListWidgetItem
from superqt.fonticon import icon
from superqt.utils import qthrottled, signals_blocked

from .roles import LINE_DATA_ROLE, PROVENANCE_ROLE, UNITS_ROLE
from .table import DamnitTableModel, prettify_notation
from .table_filter import FilterMenu, FilterProxy, FilterStatus
from .util import StatusbarStylesheet, delete_variable

log = logging.getLogger(__name__)

ROW_HEIGHT = 30
STATIC_COLUMNS = ["Status", "Proposal", "Run", "Timestamp", "Comment"]
STATIC_COLUMN_IDS = ['Status', 'proposal', 'run', 'start_time', 'comment']


class FilterHeaderView(QtWidgets.QHeaderView):
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.filtered_columns = set()
        self.filter_icon = icon(FA6S.filter)
        # Cache split titles per section so we only parse once per repaint
        self._levels_by_section = {}
        self._max_levels = 1
        self._bound_model = None
        self._hierarchy_enabled = True
        base_size = super().sizeHint().height()
        self._base_section_height = base_size if base_size > 0 else ROW_HEIGHT

    def setModel(self, model):
        if self._bound_model is not None:
            self._bound_model.headerDataChanged.disconnect(self._handle_header_change)
            self._bound_model.columnsInserted.disconnect(self._handle_model_change)
            self._bound_model.columnsRemoved.disconnect(self._handle_model_change)
            self._bound_model.modelReset.disconnect(self._handle_model_change)

        super().setModel(model)
        self._bound_model = model

        if model is not None:
            model.headerDataChanged.connect(self._handle_header_change)
            model.columnsInserted.connect(self._handle_model_change)
            model.columnsRemoved.connect(self._handle_model_change)
            model.modelReset.connect(self._handle_model_change)

        self._rebuild_hierarchy()

    def sizeHint(self):
        size = super().sizeHint()
        base = self._base_section_height or size.height()
        levels = self._max_levels if self._hierarchy_enabled else 1
        size.setHeight(base * max(1, levels))
        return size

    def _draw_filter_hint(self, painter, rect):
        icon_size = 16
        padding = 4
        icon_rect = QtCore.QRect(
            rect.left() + rect.width() - icon_size - padding,
            rect.top() + rect.height() - icon_size - padding,
            icon_size,
            icon_size,
        )
        painter.save()
        painter.drawPixmap(icon_rect, self.filter_icon.pixmap(icon_size))
        painter.restore()

    def paintSection(self, painter, rect, logicalIndex):
        if not rect.isValid():
            return

        if not self._hierarchy_enabled:
            painter.save()
            super().paintSection(painter, rect, logicalIndex)
            painter.restore()
            if logicalIndex in self.filtered_columns:
                self._draw_filter_hint(painter, rect)
            return

        painter.save()

        # paint the section, keeping styles/themes intact
        option = QtWidgets.QStyleOptionHeader()
        self.initStyleOption(option)
        option.rect = rect
        option.text = ""
        option.section = logicalIndex

        self.style().drawControl(QtWidgets.QStyle.ControlElement.CE_HeaderSection, option, painter, self)

        if logicalIndex in self.filtered_columns:
            self._draw_filter_hint(painter, rect)

        painter.restore()

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self._hierarchy_enabled:
            return

        if not self._levels_by_section and self.model() is not None:
            self._rebuild_hierarchy()

        # Second pass overlays the text so parent bands can span multiple columns
        painter = QtGui.QPainter(self.viewport())
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        painter.setPen(self.palette().color(QtGui.QPalette.ColorRole.ButtonText))

        header_rect = self.viewport().rect()
        row_tops, row_heights = self._row_positions(header_rect)

        for logical_index in range(self.count()):
            if self.isSectionHidden(logical_index):
                continue

            levels = self._levels_by_section.get(logical_index)
            if levels is None:
                title = self.model().headerData(logical_index, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
                levels = self._split_title(title)
                self._levels_by_section[logical_index] = levels
            for level_idx, text in enumerate(levels):
                if not text:
                    continue

                span_rows = self._row_span(levels, level_idx)
                top = row_tops[level_idx]
                height = sum(row_heights[level_idx: level_idx + span_rows])
                if height <= 0:
                    continue

                span_columns = self._group_span_columns(logical_index, level_idx)
                is_group = span_columns > 1
                if is_group and not self._is_group_lead(logical_index, level_idx):
                    continue

                if is_group:
                    left, width = self._group_geometry(logical_index, level_idx)
                else:
                    left = self.sectionViewportPosition(logical_index)
                    width = self.sectionSize(logical_index)

                group_rect = QtCore.QRect(left, top, width, height)
                visible_rect = group_rect.intersected(header_rect)
                if not visible_rect.isValid() or visible_rect.width() <= 0:
                    continue

                if is_group:
                    self._paint_group_background(painter, group_rect, clip_rect=visible_rect)

                text_rect = group_rect.adjusted(4, 0, -4, 0)
                if text_rect.width() <= 0:
                    continue

                painter.save()
                painter.setClipRect(visible_rect)
                painter.drawText(
                    text_rect,
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
                    text,
                )
                painter.restore()

    def update_filtered_columns(self, filtered_cols):
        """Update the set of filtered columns and trigger repaint"""
        self.filtered_columns = set(filtered_cols)
        self.viewport().update()

    def _handle_header_change(self, orientation, _first, _last):
        if orientation == Qt.Orientation.Horizontal:
            self._rebuild_hierarchy()

    def _handle_model_change(self, *args, **kwargs):
        self._rebuild_hierarchy()

    def _rebuild_hierarchy(self):
        model = self.model()
        self._levels_by_section = {}
        max_levels = 1
        if model is None:
            self._max_levels = 1
            self.viewport().update()
            self.updateGeometry()
            return

        section_count = self.count()
        for index in range(section_count):
            title = model.headerData(index, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
            levels = self._split_title(title)
            self._levels_by_section[index] = levels
            max_levels = max(max_levels, len(levels))

        self._max_levels = max(1, max_levels)
        self.updateGeometry()
        self.viewport().update()

    def _split_title(self, title):
        if title is None:
            return [""]
        title_str = str(title)
        parts = [part.strip() for part in title_str.split('/') if part.strip()]
        return parts or [title_str]

    def _row_positions(self, rect):
        levels = max(1, self._max_levels)
        total_height = max(1, rect.height())
        base = total_height // levels
        extra = total_height % levels
        heights = []
        for idx in range(levels):
            heights.append(base + (1 if idx < extra else 0))
        tops = []
        running = rect.top()
        for height in heights:
            tops.append(running)
            running += height
        return tops, heights

    def _row_span(self, levels, level_idx):
        if level_idx >= len(levels):
            return 0
        if level_idx == len(levels) - 1:
            return max(1, self._max_levels - level_idx)
        return 1

    def _is_group_lead(self, logical_index, level_idx):
        prefix = self._prefix(logical_index, level_idx)
        if prefix is None:
            return False

        visual = self.visualIndex(logical_index)
        if visual <= 0:
            return True

        for vi in range(visual - 1, -1, -1):
            prev_logical = self.logicalIndex(vi)
            if prev_logical < 0 or self.isSectionHidden(prev_logical):
                continue
            return self._prefix(prev_logical, level_idx) != prefix
        return True

    def _prefix(self, logical_index, level_idx):
        levels = self._levels_by_section.get(logical_index)
        if not levels or level_idx >= len(levels):
            return None
        return tuple(levels[: level_idx + 1])

    def _group_geometry(self, logical_index, level_idx):
        if self.isSectionHidden(logical_index):
            return 0, 0

        prefix = self._prefix(logical_index, level_idx)
        if prefix is None:
            return 0, 0

        # Start drawing from the visual leader so the merged rect covers all siblings
        lead_index = self._group_lead_index(logical_index, level_idx)
        start = self.sectionViewportPosition(lead_index)

        width = 0
        visual = self.visualIndex(lead_index)
        count = self.count()
        for vi in range(visual, count):
            next_logical = self.logicalIndex(vi)
            if next_logical < 0 or self.isSectionHidden(next_logical):
                continue
            if self._prefix(next_logical, level_idx) != prefix:
                break
            width += self.sectionSize(next_logical)

        return start, width

    def _group_span_columns(self, logical_index, level_idx):
        if self.isSectionHidden(logical_index):
            return 0

        prefix = self._prefix(logical_index, level_idx)
        if prefix is None:
            return 1

        lead_index = self._group_lead_index(logical_index, level_idx)
        visual = self.visualIndex(lead_index)
        if visual < 0:
            return 1

        span = 0
        count = self.count()
        for vi in range(visual, count):
            next_logical = self.logicalIndex(vi)
            if next_logical < 0 or self.isSectionHidden(next_logical):
                continue
            if self._prefix(next_logical, level_idx) != prefix:
                break
            span += 1

        return max(1, span)

    def _group_lead_index(self, logical_index, level_idx):
        if self.isSectionHidden(logical_index):
            return logical_index

        prefix = self._prefix(logical_index, level_idx)
        if prefix is None:
            return logical_index

        visual = self.visualIndex(logical_index)
        if visual < 0:
            return logical_index

        lead = logical_index
        for vi in range(visual - 1, -1, -1):
            prev_logical = self.logicalIndex(vi)
            if prev_logical < 0 or self.isSectionHidden(prev_logical):
                continue
            if self._prefix(prev_logical, level_idx) != prefix:
                break
            lead = prev_logical
        return lead

    def _paint_group_background(self, painter, rect, clip_rect=None):
        # Ask the style engine for a header section so merged cells keep theme gradients/borders
        painter.save()
        if clip_rect is not None:
            painter.setClipRect(clip_rect)
        option = QtWidgets.QStyleOptionHeader()
        self.initStyleOption(option)
        option.rect = rect
        option.text = ""
        option.state |= QtWidgets.QStyle.StateFlag.State_Raised
        option.state &= ~(QtWidgets.QStyle.StateFlag.State_Sunken | QtWidgets.QStyle.StateFlag.State_On)
        self.style().drawControl(QtWidgets.QStyle.ControlElement.CE_HeaderSection, option, painter, self)
        painter.restore()

    def set_hierarchical_enabled(self, enabled: bool):
        if self._hierarchy_enabled == enabled:
            return
        self._hierarchy_enabled = enabled
        if enabled:
            self._rebuild_hierarchy()
        else:
            self._levels_by_section.clear()
            self._max_levels = 1

        parent = self.parent()
        if parent is not None:
            if hasattr(parent, "updateGeometries"):
                parent.updateGeometries()
            parent.viewport().update()


class ItemRendererDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Pre-compute equaly spaced color hues for provenance indicator
        self.n_colors = 16
        self.hues = [i / self.n_colors for i in range(self.n_colors)]
        self.label_colors = {}

    def _provenance_color(self, provenance: str) -> QtGui.QColor:
        if provenance not in self.label_colors:
            digest = hashlib.sha256(provenance.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.n_colors
            hue = self.hues[index]
            color = QtGui.QColor.fromHsl(int(255 * hue), 170, 200, 220)
            self.label_colors[provenance] = color
        return self.label_colors[provenance]

    def _paint_provenance_marker(self, painter, option, index):
        provenance = index.data(PROVENANCE_ROLE)
        if not provenance:
            return
        rect = option.rect
        color = self._provenance_color(provenance)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        # draw corner triangle
        x = rect.right()
        y = rect.top() + 1
        painter.setBrush(color)
        painter.setPen(QtGui.QPen(color.darker(130)))
        tri = QtGui.QPolygonF([
            QtCore.QPointF(x, y),
            QtCore.QPointF(x - 10, y),
            QtCore.QPointF(x, y + 10),
        ])
        painter.drawPolygon(tri)
        painter.restore()

    def _paint_sparkline(self, data, painter, option, index):
        # Base item rendering (selection background, etc.)
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""

        style = opt.widget.style() if opt.widget else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)

        # Pixel bounds for the sparkline drawing area.
        rect = opt.rect.adjusted(2, 2, -2, -2)
        if rect.width() < 4 or rect.height() < 4:
            return

        # Validate/prepare data.
        if data.ndim != 2 or data.shape[0] != 2 or data.shape[1] < 2:
            return

        x = np.asarray(data[0], dtype=np.float64)
        y = np.asarray(data[1], dtype=np.float64)

        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            return
        x = x[finite]
        y = y[finite]
        if x.size < 2:
            return

        # Normalize data to [0,1] and map to pixel coordinates.
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))

        span_x = x_max - x_min
        span_y = y_max - y_min

        if span_x == 0:
            x_norm = np.linspace(0, 1, x.size)
        else:
            x_norm = (x - x_min) / span_x

        if span_y == 0:
            y_norm = np.full_like(y, 0.5)
        else:
            y_norm = (y - y_min) / span_y

        x_pix = rect.left() + x_norm * (rect.width() - 1)
        y_pix = rect.top() + (1 - y_norm) * (rect.height() - 1)

        # Build the polyline path.
        poly = QtGui.QPolygonF()
        for xi, yi in zip(x_pix, y_pix):
            poly.append(QtCore.QPointF(float(xi), float(yi)))

        # Draw the sparkline.
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        line_color = QtGui.QColor(0, 120, 215)    # blue
        hover_line_color = QtGui.QColor(181, 181, 181)  # gray
        text_color = QtGui.QColor(0, 0, 0)        # black
        dot_color = QtGui.QColor(220, 0, 0)       # red
        pen = QtGui.QPen(line_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawPolyline(poly)

        # Hover overlay: snap to extrema if near, otherwise interpolate.
        view = opt.widget
        hover_index = view._sparkline_hover_index
        hover_t = view._sparkline_hover_t
        if hover_index is not None and hover_t is not None and hover_index == index:
            x_line = rect.left() + hover_t * (rect.width() - 1)

            snap_idx = None
            if x.size >= 3:
                extrema = []
                for i in range(1, x.size - 1):
                    if (y[i] >= y[i - 1] and y[i] >= y[i + 1]) or (
                        y[i] <= y[i - 1] and y[i] <= y[i + 1]
                    ):
                        extrema.append(i)
                if extrema:
                    extrema = np.asarray(extrema, dtype=int)
                    dists = np.abs(x_pix[extrema] - x_line)
                    nearest = int(extrema[np.argmin(dists)])
                    if dists.min() <= 6.0:
                        snap_idx = nearest

            if snap_idx is not None:
                x_line = float(x_pix[snap_idx])
                y_val = float(y[snap_idx])
                y_marker = float(y_pix[snap_idx])
            else:
                try:
                    if span_x == 0:
                        y_val = float(np.nanmean(y))
                    else:
                        x_target = x_min + hover_t * span_x
                        order = np.argsort(x)
                        x_sorted = x[order]
                        y_sorted = y[order]
                        y_val = float(np.interp(x_target, x_sorted, y_sorted))
                except Exception:
                    y_val = None

                if y_val is not None and np.isfinite(y_val):
                    if span_y == 0:
                        y_norm_val = 0.5
                    else:
                        y_norm_val = (y_val - y_min) / span_y
                    y_norm_val = max(0.0, min(1.0, float(y_norm_val)))
                    y_marker = rect.top() + (1 - y_norm_val) * (rect.height() - 1)
                else:
                    y_marker = None

            painter.setPen(QtGui.QPen(hover_line_color))
            painter.drawLine(QtCore.QPointF(x_line, rect.top()),
                             QtCore.QPointF(x_line, rect.bottom()))

            # Marker + value label.
            if y_val is not None and np.isfinite(y_val) and y_marker is not None:
                painter.save()
                painter.setBrush(dot_color)
                painter.setPen(QtGui.QPen(dot_color))
                painter.drawEllipse(QtCore.QPointF(x_line, y_marker), 2.5, 2.5)
                painter.restore()

                text = prettify_notation(y_val)
                metrics = painter.fontMetrics()
                text_w = metrics.horizontalAdvance(text)
                text_h = metrics.height()
                box_w = text_w + 6
                box_h = text_h + 4
                text_left = x_line + 4
                if text_left + box_w > rect.right():
                    text_left = x_line - box_w - 4
                text_left = max(rect.left(), min(text_left, rect.right() - box_w))
                text_top = rect.top() + 2
                if text_top + box_h > rect.bottom():
                    text_top = rect.bottom() - box_h - 2
                box_rect = QtCore.QRectF(text_left, text_top, box_w, box_h)
                bg = QtGui.QColor(255, 255, 255, 210)
                painter.fillRect(box_rect, bg)
                painter.setPen(text_color)
                painter.drawText(
                    box_rect.adjusted(3, 2, -3, -2),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    text,
                )
        painter.restore()

    def _paint_with_units(self, painter, option, index):
        option = QtWidgets.QStyleOptionViewItem(option)
        value = index.data(Qt.ItemDataRole.DisplayRole)
        if (units := index.data(UNITS_ROLE)) is None:
            super().paint(painter, option, index)
            return

        self.initStyleOption(option, index)

        # Draw everything except the text
        option.text = ""
        style = option.widget.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, option, painter, option.widget)

        if option.state & QtWidgets.QStyle.StateFlag.State_Selected:
            value_color = option.palette.color(QtGui.QPalette.ColorRole.HighlightedText)
            unit_color = option.palette.color(QtGui.QPalette.ColorRole.HighlightedText)
        else:
            value_color = option.palette.color(QtGui.QPalette.ColorRole.Text)
            unit_color = option.palette.color(QtGui.QPalette.ColorRole.Text)

        doc = QtGui.QTextDocument()
        doc.setDefaultFont(option.font)
        doc.setHtml(
            f'<span style="color:{value_color.name()};">{value}</span> '
            f'<span style="color:{unit_color.name()}; font-style:italic; opacity:0.6;">{units}</span>'
        )

        # Get the exact text rect Qt would have used
        text_rect = style.subElementRect(QtWidgets.QStyle.SubElement.SE_ItemViewItemText, option, option.widget)

        painter.save()
        painter.translate(text_rect.topLeft())
        doc.setTextWidth(text_rect.width())
        doc.drawContents(painter)
        painter.restore()

    def paint(self, painter, option, index):
        if (data := index.data(LINE_DATA_ROLE)) is not None:
            self._paint_sparkline(data, painter, option, index)
        else:
            self._paint_with_units(painter, option, index)

        self._paint_provenance_marker(painter, option, index)


@dataclass
class ColumnState:
    id: str
    title: str
    visible: bool = True

    def to_item(self) -> QListWidgetItem:
        item = QtWidgets.QListWidgetItem(self.title)
        item.setData(Qt.ItemDataRole.UserRole, self.id)
        item.setCheckState(
            Qt.CheckState.Checked if self.visible else Qt.CheckState.Unchecked
        )
        return item

class ColumnsState(QtCore.QObject):
    # Tracks column order & visibility in 2 QListWidgets
    #   This only tracks specifically hidden columns, not columns hidden by a tag filter
    visibility_changed = QtCore.pyqtSignal(str, bool)  # column_id, visible
    column_moved = QtCore.pyqtSignal(int, int)  # from_ix, to_ix

    def __init__(self, tbl_view: 'TableView'):
        super().__init__(tbl_view)
        self.table_view = tbl_view
        self.column_items = {}  # {column_id: QListWidgetItem}
        self._columns_widget = QtWidgets.QListWidget()
        self._static_columns_widget = QtWidgets.QListWidget()

        self._columns_widget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self._columns_widget.itemChanged.connect(self.item_changed)
        self._columns_widget.model().rowsMoved.connect(self.item_moved)

        self._columns_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._columns_widget.customContextMenuRequested.connect(self.show_delete_menu)

        self._static_columns_widget.itemChanged.connect(self.item_changed)
        self._static_columns_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,
                                                  QtWidgets.QSizePolicy.Policy.Minimum)
        self._static_columns_widget.setStyleSheet("QListWidget {padding: 0px;} QListWidget::item { margin: 5px; }")
        self._columns_widget.setStyleSheet("QListWidget {padding: 0px;} QListWidget::item { margin: 5px; }")

    def set_model(self, model: 'DamnitTableModel'):
        cols = [ColumnState(cid, title, model.column_settings_loaded.get(cid, True))
                for cid, title in zip(model.column_ids, model.column_titles)]

        self._columns_widget.clear()
        self._static_columns_widget.clear()
        self.column_items.clear()

        moveable_cols = []
        for cs in cols:
            if cs.id in STATIC_COLUMN_IDS:
                item = self.column_items[cs.id] = cs.to_item()
                self._static_columns_widget.addItem(item)
            else:
                moveable_cols.append(cs)

        self.add_new_columns(moveable_cols)

    def add_new_columns(self, cols: list[ColumnState], positions=None):
        if positions is None:
            rows_count = self._columns_widget.count()
            positions = [ii + rows_count for ii in range(len(cols))]

        for cs, position in zip(cols, positions):
            item = self.column_items[cs.id] = cs.to_item()
            self._columns_widget.insertItem(position, item)

    def on_columns_inserted(self, _parent, first, last):
        model = self.table_view.damnit_model
        self.add_new_columns([ColumnState(
            id=model.column_id(i),
            title=model.column_title(i),
        ) for i in range(first, last + 1)])

    def on_columns_about_to_be_removed(self, _parent, first, last):
        model = self.table_view.damnit_model
        col_ids = [model.column_id(i) for i in range(first, last + 1)]

        for cid in col_ids:
            if (item := self.column_items.pop(cid, None)):
                row = self._columns_widget.row(item)
                self._columns_widget.takeItem(row)

    def get_column_states(self) -> dict[str, bool]:
        # We're also preserving the order here, so we can't just iterate through
        # self.column_items
        column_states = {}

        for widget in [self._static_columns_widget, self._columns_widget]:
            for row in range(widget.count()):
                item = widget.item(row)
                col_id = item.data(Qt.ItemDataRole.UserRole)
                column_states[col_id] = item.checkState() == Qt.CheckState.Checked

        return column_states

    def item_changed(self, item: QListWidgetItem):
        col_id = item.data(Qt.ItemDataRole.UserRole)
        state = item.checkState() == Qt.CheckState.Checked
        self.visibility_changed.emit(col_id, state)

    def item_moved(self, srcparent, srcstart, srcend, dstparent, dststart):
        # Take account of the static columns
        col_offset = self._static_columns_widget.count()
        # Qt gives us a destination index for the insert position *before* the
        # item is moved from its previous position, e.g. moving the top item 1
        # place down goes from 0 -> 2. Convert to the position after the move.
        if dststart > srcstart:
            dststart -= 1
        # Only 1 item can be selected & moved at a time
        self.column_moved.emit(srcstart + col_offset, dststart + col_offset)

    def show_delete_menu(self, pos):
        item = self._columns_widget.itemAt(pos)
        if item is None:
            # This happens if the user clicks on blank space inside the widget
            return

        global_pos = self._columns_widget.mapToGlobal(pos)
        menu = QtWidgets.QMenu()
        menu.addAction("Delete")
        action = menu.exec(global_pos)
        if action is not None:
            name = item.data(Qt.ItemDataRole.UserRole)
            self.confirm_delete_variable(name)

    def confirm_delete_variable(self, name):
        button = QMessageBox.warning(
            self.table_view, "Confirm deletion",
            f"You are about to permanently delete the variable <b>'{name}'</b> "
            "from the database and HDF5 files. This cannot be undone. "
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            defaultButton=QMessageBox.StandardButton.No
        )
        if button == QMessageBox.StandardButton.Yes:
            model = self.table_view.damnit_model
            delete_variable(model.db, name)
            model.removeColumn(model.find_column(name, by_title=False))

    def make_dialog(self, parent):
        dialog = QtWidgets.QDialog(parent)
        dialog.setWindowTitle("Column settings")
        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(QtWidgets.QLabel(
            "These columns can be hidden but not reordered or deleted:"
        ))
        layout.addWidget(self._static_columns_widget)
        layout.addWidget(QtWidgets.QLabel(
            "Drag these columns to reorder them, right-click to delete:"
        ))
        layout.addWidget(self._columns_widget)
        dialog.setLayout(layout)
        return dialog


class TableView(QtWidgets.QTableView):
    settings_changed = QtCore.pyqtSignal()
    log_view_requested = QtCore.pyqtSignal(int, int)  # proposal, run
    model_updated = QtCore.pyqtSignal()
    hierarchical_header_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlternatingRowColors(False)
        self.hierarchical_header_enabled = True
        self._restoring_column_widths = False
        self._sparkline_hover_index = QtCore.QModelIndex()
        self._sparkline_hover_t = None

        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )

        self.verticalHeader().setMinimumSectionSize(ROW_HEIGHT)
        self.setItemDelegate(ItemRendererDelegate(self))
        self.setMouseTracking(True)

        self.columns_state = ColumnsState(self)
        self.columns_state.visibility_changed.connect(self.column_visibility_changed)
        self.columns_state.column_moved.connect(self.column_reordered)

        self.context_menu = QtWidgets.QMenu(self)
        self.copy_markdown_action = QtGui.QAction('Copy selection as Markdown', self)
        self.copy_markdown_action.setEnabled(False)
        self.context_menu.addAction(self.copy_markdown_action)
        self.zulip_action = QtGui.QAction('Export table to the Logbook', self)
        self.context_menu.addAction(self.zulip_action)
        self.show_logs_action = QtGui.QAction('View processing logs')
        self.show_logs_action.setEnabled(False)  # Enabled with selection
        self.show_logs_action.triggered.connect(self.show_run_logs)
        self.context_menu.addAction(self.show_logs_action)
        self.process_action = QtGui.QAction('Reprocess runs')
        self.context_menu.addAction(self.process_action)
        self.cancel_jobs_action = QtGui.QAction('Cancel processing')
        self.cancel_jobs_action.setEnabled(False)
        self.context_menu.addAction(self.cancel_jobs_action)

        # Add tag filtering support
        self._current_tag_filter = set()  # Change to set for multiple tags
        self._tag_filter_button = QtWidgets.QPushButton("Variables by Tag")
        self._tag_filter_button.clicked.connect(self._show_tag_filter_menu)
        # add column values filter support
        self._filter_status = FilterStatus(self, parent)

    def setModel(self, model: 'DamnitTableModel'):
        """
        Overload
        """
        if (old_sel_model := self.selectionModel()) is not None:
            old_sel_model.deleteLater()
        if (old_model := self.model()) is not None:
            old_model.deleteLater()

        # to simplify some units test setup, we set the TableView model to
        # a standard QStandardItemModel. This allows us to test the TableView
        # without needing to create a full DamnitTableModel.
        if isinstance(model, DamnitTableModel):
            self.damnit_model = model
        else:
            self.damnit_model = None

        sfpm = FilterProxy(self)
        sfpm.setSourceModel(model)
        sfpm.setSortRole(Qt.ItemDataRole.UserRole)  # Numeric sort where relevant
        super().setModel(sfpm)

        # When loading a new model, the saved column order is applied at the
        # model level (changing column logical indices). So we need to reset
        # any reordering from the view level, which maps logical indices to
        # different visual indices, to show the columns as in the model.
        self.setHorizontalHeader(FilterHeaderView(self))
        header = self.horizontalHeader()
        header.set_hierarchical_enabled(self.hierarchical_header_enabled)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self.show_horizontal_header_menu)
        header.sectionResized.connect(self._on_section_resized)
        # header.setSectionsMovable(True)  # TODO need to update variable order in the table / emit settings_changed

        if model is not None:
            self.model().rowsInserted.connect(self.resize_new_rows)
            self.model().columnsRemoved.connect(self.on_columns_removed)
            self.model().filterChanged.connect(
                lambda: header.update_filtered_columns(self.model().filters))
            self.resizeRowsToContents()
        if self.damnit_model is not None:
            self.columns_state.set_model(self.damnit_model)
            for col_id, vis in self.columns_state.get_column_states().items():
                self.set_column_visibility(col_id, vis)
            self.model().columnsInserted.connect(self.columns_state.on_columns_inserted)
            self.model().columnsAboutToBeRemoved.connect(self.columns_state.on_columns_about_to_be_removed)
            self.damnit_model.processing_jobs.run_jobs_changed.connect(
                lambda *_: self.update_cancel_processing_action()
            )

        self.selectionModel().selectionChanged.connect(self.selection_changed)

        self.model_updated.emit()

    def _set_sparkline_hover(self, index, t):
        self.setCursor(Qt.CursorShape.BlankCursor)
        old_index = self._sparkline_hover_index
        old_t = self._sparkline_hover_t
        if old_index == index and old_t == t:
            return
        self._sparkline_hover_index = index
        self._sparkline_hover_t = t
        if old_index.isValid():
            self.viewport().update(self.visualRect(old_index))
        if index.isValid():
            self.viewport().update(self.visualRect(index))

    def _clear_sparkline_hover(self):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if not self._sparkline_hover_index.isValid():
            self._sparkline_hover_t = None
            return
        old_index = self._sparkline_hover_index
        self._sparkline_hover_index = QtCore.QModelIndex()
        self._sparkline_hover_t = None
        if old_index.isValid():
            self.viewport().update(self.visualRect(old_index))

    def mouseMoveEvent(self, event):
        index = self.indexAt(event.pos())
        if index.isValid() and index.data(LINE_DATA_ROLE) is not None:
            rect = self.visualRect(index)
            if rect.width() > 1:
                t = (event.pos().x() - rect.left()) / (rect.width() - 1)
            else:
                t = 0.0
            t = max(0.0, min(1.0, float(t)))
            self._set_sparkline_hover(index, t)
        else:
            self._clear_sparkline_hover()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._clear_sparkline_hover()
        super().leaveEvent(event)

    def _on_section_resized(self, logical_index, old_size, new_size):
        if self._restoring_column_widths:
            return
        if old_size == new_size:
            return
        header = self.horizontalHeader()
        if header.sectionResizeMode(logical_index) == QtWidgets.QHeaderView.ResizeMode.ResizeToContents:
            return
        self._queue_settings_changed()

    @qthrottled(timeout=200, leading=False)
    def _queue_settings_changed(self):
        self.settings_changed.emit()

    def selected_rows(self):
        """Get indices of selected rows in the DamnitTableModel"""
        proxy_rows = self.selectionModel().selectedRows()
        # Translate indices in sorted proxy model back to the underlying model
        proxy = self.model()
        return [proxy.mapToSource(ix) for ix in proxy_rows]

    def selected_proposal_runs(self):
        return [
            self.damnit_model.row_to_proposal_run(ix.row())
            for ix in self.selected_rows()
        ]

    def update_cancel_processing_action(self):
        sel_prop_run = self.selected_proposal_runs()
        slurm_jobs = self.damnit_model.processing_jobs.active_slurm_jobs_for_runs(sel_prop_run)
        self.cancel_jobs_action.setEnabled(bool(slurm_jobs))

    def selection_changed(self):
        has_sel = self.selectionModel().hasSelection()
        self.copy_markdown_action.setEnabled(has_sel)
        self.show_logs_action.setEnabled(has_sel)
        self.update_cancel_processing_action()

    def column_visibility_changed(self, col_id, visible):
        # Checkbox toggled in column selection dialog
        self.set_column_visibility(col_id, visible)
        self.settings_changed.emit()

    def set_column_visibility(self, name, visible):
        """
        Make a column visible or not. This function should be used instead of the lower-level
        setColumnHidden().

        The main use-cases are hiding/showing a column when the user clicks a
        checkbox, and hiding a column programmatically when loading the users
        settings. In the first case we want to emit a signal to save the
        settings, and in the second we want the checkbox for that column to be
        deselected. The `for_restore` argument lets you specify which behaviour
        you want.
        """
        try:
            column_index = self.damnit_model.find_column(name, by_title=False)
        except KeyError:
            log.error("Could not find column %r to set visibility", name)
            return

        self.setColumnHidden(column_index, not visible)

    def apply_column_widths(self, column_widths):
        if not column_widths or not hasattr(self, "damnit_model") or self.damnit_model is None:
            return
        header = self.horizontalHeader()
        self._restoring_column_widths = True
        try:
            for logical_idx, title in enumerate(self.damnit_model.column_titles):
                width = column_widths.get(title)
                if not isinstance(width, int):
                    continue
                if width <= 0:
                    continue
                if header.sectionResizeMode(logical_idx) == QtWidgets.QHeaderView.ResizeMode.ResizeToContents:
                    continue
                self.setColumnWidth(logical_idx, int(width))
        finally:
            self._restoring_column_widths = False

    def column_reordered(self, from_ix, to_ix):
        self.horizontalHeader().moveSection(from_ix, to_ix)
        self.settings_changed.emit()

    def on_columns_removed(self, _parent, _first, _last):
        self.apply_tag_filter(self._current_tag_filter)
        self.apply_tag_filter.flush()

        self.settings_changed.emit()  # Save column order

    def get_column_states(self):
        return self.columns_state.get_column_states()

    def get_column_widths(self):
        if not hasattr(self, "damnit_model") or self.damnit_model is None:
            return {}
        header = self.horizontalHeader()
        widths = {}
        for logical_idx, title in enumerate(self.damnit_model.column_titles):
            width = header.sectionSize(logical_idx)
            if width > 0:
                widths[title] = int(width)
        return widths

    def resize_new_rows(self, parent, first, last):
        for row in range(first, last + 1):
            self.resizeRowToContents(row)

    def _get_columns_status(self, widget):
        res = {}
        for ii in range(widget.count()):
            ci = widget.item(ii)
            res[ci.text()] = ci.isSelected()
        return res

    def get_movable_columns(self):
        return self._get_columns_status(self._columns_widget)

    def get_movable_columns_count(self):
        return self._columns_widget.count()

    def get_static_columns(self):
        return self._get_columns_status(self._static_columns_widget)

    def get_static_columns_count(self):
        return self._static_columns_widget.count()

    def contextMenuEvent(self, event):
        self.context_menu.popup(QtGui.QCursor.pos())

    def show_run_logs(self):
        # Get first selected row
        try:
            row = self.selected_rows()[0].row()
        except IndexError:
            self.damnit_model._main_window.show_status_message(
                "No row selected",
                timeout=7000,
                stylesheet=StatusbarStylesheet.ERROR
            )
            return
        prop, run = self.damnit_model.row_to_proposal_run(row)
        self.log_view_requested.emit(prop, run)

    def _show_tag_filter_menu(self):
        """Show a menu to select tag filtering."""
        if not hasattr(self, 'damnit_model') or not self.damnit_model:
            return

        menu = QtWidgets.QMenu(self)

        # Add "Show All" option
        show_all_action = menu.addAction("Show All Variables")
        show_all_action.triggered.connect(lambda: self.apply_tag_filter(set()))
        if not self._current_tag_filter:
            show_all_action.setEnabled(False)

        menu.addSeparator()

        # Add checkable actions for each tag
        for tag in sorted(self.damnit_model.db.get_all_tags()):
            action = menu.addAction(tag)
            action.setCheckable(True)
            action.setChecked(tag in self._current_tag_filter)
            action.triggered.connect(lambda checked, t=tag: self._toggle_tag_filter(t))

        menu.exec(QtGui.QCursor.pos())

    def _toggle_tag_filter(self, tag_name: str):
        """Toggle a tag in the filter set and apply the filter."""
        if tag_name in self._current_tag_filter:
            self._current_tag_filter.remove(tag_name)
        else:
            self._current_tag_filter.add(tag_name)
        self.apply_tag_filter(self._current_tag_filter)

    @qthrottled(timeout=50, leading=False)
    def apply_tag_filter(self, tag_names: set):
        """Filter columns to show only variables with selected tags."""
        self._current_tag_filter = tag_names

        # Get user's column visibility preferences
        column_states = self.get_column_states()

        if not tag_names:
            # Show all columns that were checked in the column widgets
            for col, state in column_states.items():
                self.set_column_visibility(col, state)

            self._tag_filter_button.setText("Variables by Tag")
        else:
            # Get all variables that have any of the selected tags
            tagged_vars = set()
            for tag in tag_names:
                tagged_vars.update(self.damnit_model.db.get_variables_by_tag(tag))

            # Hide/show columns based on whether they're tagged AND checked in column widgets
            for idx, (col, state) in enumerate(column_states.items()):
                is_static = idx < self.get_static_columns_count()
                is_tagged = col in tagged_vars

                # Column should be visible if:
                # 1. It's a static column that's checked in column widgets, or
                # 2. It's a non-static column that's both tagged and checked in column widgets
                show = state and (is_static or is_tagged)
                self.set_column_visibility(col, show)

            # Update button text
            if len(tag_names) == 1:
                self._tag_filter_button.setText(f"Variables: {next(iter(tag_names))}")
            else:
                self._tag_filter_button.setText(f"Variables: {len(tag_names)} tags")

    def get_toolbar_widgets(self):
        """Return widgets to be added to the toolbar."""
        return [self._tag_filter_button, self._filter_status]

    def show_horizontal_header_menu(self, position):
        pos = QCursor.pos()
        index = self.horizontalHeader().logicalIndexAt(position)
        menu = QMenu(self)
        sort_asc_action = QAction(icon(FA6S.arrow_up_short_wide), "Sort Ascending", self)
        sort_desc_action = QAction(icon(FA6S.arrow_down_wide_short), "Sort Descending", self)
        filter_action = QAction(icon(FA6S.filter), "Filter", self)

        menu.addAction(sort_asc_action)
        menu.addAction(sort_desc_action)
        menu.addSeparator()
        menu.addAction(filter_action)

        sort_asc_action.triggered.connect(lambda: self.sortByColumn(index, Qt.SortOrder.AscendingOrder))
        sort_desc_action.triggered.connect(lambda: self.sortByColumn(index, Qt.SortOrder.DescendingOrder))
        filter_action.triggered.connect(lambda: FilterMenu(index, self.model(), self).popup(pos))

        menu.exec(pos)

    def set_hierarchical_header_enabled(self, enabled: bool, emit_signal=True):
        enabled = bool(enabled)
        if self.hierarchical_header_enabled == enabled:
            return
        self.hierarchical_header_enabled = enabled
        header = self.horizontalHeader()
        if isinstance(header, FilterHeaderView):
            header.set_hierarchical_enabled(enabled)
        if emit_signal:
            self.hierarchical_header_changed.emit(enabled)
            self.settings_changed.emit()
