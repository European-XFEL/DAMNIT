import json
import logging
import time
from base64 import b64encode
from itertools import groupby

import numpy as np
from fonticon_fa6 import FA6S
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QAction, QHeaderView, QMenu, QMessageBox, QWidgetAction, QListWidgetItem
from superqt import QSearchableListWidget
from superqt.fonticon import icon

from ..backend.db import BlobTypes, DamnitDB, ReducedData
from ..backend.user_variables import value_types_by_name
from ..util import StatusbarStylesheet, delete_variable, timestamp2str
from .table_header import HeaderView, FilterProxy, SearchMenu

log = logging.getLogger(__name__)

ROW_HEIGHT = 30
THUMBNAIL_SIZE = 35
COMMENT_ID_ROLE = Qt.ItemDataRole.UserRole + 1


# movable columns
# resiseable columns
# filter by column values
# group columns


# def check_type_consistency(model, column):
#     """
#     Check if all items in a specific column have the same data type.
    
#     Args:
#         model: QStandardItemModel
#         column: int, column index to check
    
#     Returns:
#         tuple: (bool, type or None) - (is_consistent, common_type)
#     """
#     if model.rowCount() == 0:
#         return True, None

#     types = set()
#     for row in range(model.rowCount()):
#         item = model.index(row, column)
#         item_data = item.data(Qt.ItemDataRole.UserRole)
#         if item and item_data is not None:
#             types.add(type(item_data))

#     if len(types) == 1:
#         return True, types.pop()
#     return False, None


# def check_type_numeric(model, column):
#     """Check if all non-null values in column can be converted to float"""
#     for row in range(model.rowCount()):
#         item = model.item(row, column)
#         item_data = item.data(Qt.ItemDataRole.UserRole)
#         if item and item_data:
#             try:
#                 float(item_data)
#             except ValueError:
#                 return False
#     return True


# class FilterProxy(QtCore.QSortFilterProxyModel):
#     filterChanged = QtCore.pyqtSignal()

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.filters = {}

#     def set_filter(self, column, expression=None):
#         print('set_filter', column, expression)
#         if expression is not None:
#             self.filters[column] = expression
#         elif self.filters.get(column) is not None:
#             print('clear filter:', column)
#             del self.filters[column]
#         self.invalidateFilter()
#         self.filterChanged.emit()

#     def clear_filter(self):
#         self.filters = {}
#         self.invalidateFilter()
#         self.filterChanged.emit()

#     def filterAcceptsRow(self, source_row, source_parent):
#         for col, expr in self.filters.items():
#             data = self.sourceModel().index(source_row, col, source_parent).data()
#             if not expr(data):
#                 return False
#         return True


# class NumericalRangeFilterWidget(QtWidgets.QDialog):
#     filterChanged = QtCore.pyqtSignal(float, float, bool)

#     def __init__(self, data, parent=None):
#         super().__init__(parent)

#         self.data = np.asarray(
#             [d for d in data if isinstance(d, (float, int))], dtype=np.float64)

#         self.allow_nans = QtWidgets.QCheckBox('Allow NaNs')
#         self.allow_nans.setChecked(True)
#         self.allow_nans.stateChanged.connect(self.rangeChanged)

#         self.clear_filter = QtWidgets.QPushButton('Clear')
#         self.clear_filter.clicked.connect(self.clear)
#         self.ok = QtWidgets.QPushButton('Done')
#         self.ok.clicked.connect(self.close)

#         hbox = QtWidgets.QHBoxLayout()
#         hbox.addWidget(self.clear_filter)
#         hbox.addWidget(self.ok)

#         layout = QtWidgets.QVBoxLayout()
#         layout.addWidget(self.allow_nans)
#         layout.addLayout(hbox)
#         self.setLayout(layout)
#         self.show()

#     @QtCore.pyqtSlot()
#     def rangeChanged(self):
#         allow_nans = self.allow_nans.isChecked()
#         self.filterChanged.emit(self.data.min(), self.data.max(), allow_nans)
        
#     def clear(self):
#         self.filterChanged.emit(-np.inf, np.inf, True)
#         self.close()


# class StringFilterMenu(QtWidgets.QMenu):
#     def __init__(self, data, column_index, parent=None):
#         super().__init__(parent)
#         self.column_index = column_index

#         self.signalMapper = QtCore.QSignalMapper(self)

#         actionAll = QtWidgets.QAction("All", self)
#         actionAll.triggered.connect(self.onActionAllTriggered)
#         self.addAction(actionAll)
#         self.addSeparator()

#         for idx, value in enumerate(sorted(set(data))):              
#             action = QtWidgets.QAction(value, self)
#             self.signalMapper.setMapping(action, idx)  
#             action.triggered.connect(self.signalMapper.map)  
#             self.addAction(action)

#         self.signalMapper.mapped.connect(self.onSignalMapperMapped)  

#     @QtCore.pyqtSlot()
#     def onActionAllTriggered(self):
#         self.parent().model().set_filter(self.column_index, lambda *args: True)

#     @QtCore.pyqtSlot(int)
#     def onSignalMapperMapped(self, idx):
#         str_action = self.signalMapper.mapping(idx).text()
#         str_filter = QtCore.QRegExp(str_action, QtCore.Qt.CaseSensitive, QtCore.QRegExp.FixedString)
#         self.parent().model().set_filter(self.column_index, lambda data: str_filter.exactMatch(data))


# class SearchMenu(QMenu):
#     def __init__(self, column, model, parent=None):
#         super().__init__(parent)
#         self.column = column
#         self.model = model

#         action_all = QAction("All", self)
#         action_all.triggered.connect(self.onActionAllTriggered)
#         self.addAction(action_all)

#         self.item_list = QSearchableListWidget()
#         self.item_list.filter_widget.setPlaceholderText("Search")
#         self.item_list.layout().setContentsMargins(0, 0, 0, 0)
#         self.item_list.itemChanged.connect(self.selectionChanged)

#         unique_data = set()
#         for row in range(model.rowCount()):
#             _item = model.index(row, column)
#             if _item is None:
#                 continue
#             data = _item.data()
#             print(f'row:{row}, col:{self.column}, data:{data}')
#             unique_data.add(data)

#         for data in sorted(unique_data, key=lambda item: str(item) if item is not None else ''):
#             item = QListWidgetItem()
#             item.setData(Qt.UserRole, data)
#             item.setData(Qt.DisplayRole, str(data or ''))
#             item.setCheckState(Qt.Checked)
#             self.item_list.addItem(item)

#         list_action = QWidgetAction(self)
#         list_action.setDefaultWidget(self.item_list)
#         self.addAction(list_action)

#     @QtCore.pyqtSlot()
#     def onActionAllTriggered(self):
#         print('SearchMenu.onActionAllTriggered')
#         self.model.set_filter(self.column, None)

#     @QtCore.pyqtSlot()
#     def selectionChanged(self):
#         print('SearchMenu.selectionChanged')
#         selection = set()
#         for idx in range(self.item_list.count()):
#             item = self.item_list.item(idx)
#             if item.checkState() == Qt.Checked:
#                 data = item.data(Qt.DisplayRole)
#                 selection.add(item.data(Qt.DisplayRole))

#         self.model.set_filter(self.column, lambda data: data in selection)


# class HeaderView(QHeaderView):
#     def __init__(self, orientation, parent=None):
#         super().__init__(orientation, parent)
#         # self.setSectionsMovable(True)
#         self.setSectionResizeMode(QHeaderView.Interactive)
#         self.setSectionsClickable(True)
#         self.hovered_section = -1
#         self.button_hovered = False

#     def mouseMoveEvent(self, event):
#         super().mouseMoveEvent(event)
#         hovered_section = self.logicalIndexAt(event.pos())
#         old_hovered = self.hovered_section
#         old_button_hovered = self.button_hovered

#         self.hovered_section = hovered_section
#         self.button_hovered = self.isButtonHovered(event.pos())

#         if old_hovered != self.hovered_section or old_button_hovered != self.button_hovered:
#             self.updateSection(old_hovered)
#             self.updateSection(self.hovered_section)

#     def leaveEvent(self, event):
#         super().leaveEvent(event)
#         old_hovered = self.hovered_section
#         self.hovered_section = -1
#         self.button_hovered = False
#         self.updateSection(old_hovered)

#     def paintSection(self, painter, rect, logicalIndex):
#         painter.save()
#         super().paintSection(painter, rect, logicalIndex)
#         painter.restore()

#         # If this section is being hovered, draw the button
#         if logicalIndex == self.hovered_section:
#             button_rect = self.getButtonRect(rect)
            
#             # Set up the painter for the button
#             painter.save()
            
#             # Draw the "V" character
#             font = painter.font()
#             font.setPointSize(10)
#             painter.setFont(font)
            
#             if self.button_hovered:
#                 # Use a different color when button is hovered
#                 painter.setPen(QColor(60, 60, 60))
#             else:
#                 painter.setPen(QColor(120, 120, 120))
                
#             painter.drawText(button_rect, Qt.AlignCenter, "⋮")
            
#             painter.restore()

#     def mousePressEvent(self, event):
#         index = self.logicalIndexAt(event.pos())
#         if index == self.hovered_section and self.isButtonHovered(event.pos()):
#             # self.showHeaderMenu(index, event.globalPos())
#             self.parent().show_horizontal_header_menu(index, event.globalPos())
#         else:
#             super().mousePressEvent(event)

#     def showHeaderMenu(self, index, pos):
#         menu = QMenu(self)
#         sort_asc_action = QAction("Sort Ascending", self)
#         sort_desc_action = QAction("Sort Descending", self)
#         filter_action = QAction("Filter", self)

#         menu.addAction(sort_asc_action)
#         menu.addAction(sort_desc_action)
#         menu.addSeparator()
#         menu.addAction(filter_action)

#         sort_asc_action.triggered.connect(lambda: self.parent().sortByColumn(index, Qt.AscendingOrder))
#         sort_desc_action.triggered.connect(lambda: self.parent().sortByColumn(index, Qt.DescendingOrder))
#         filter_action.triggered.connect(lambda: SearchMenu(index, self.parent().model(), self).popup(pos))

#         menu.exec_(pos)

#     def filterColumn(self, column):
#         # Implement your filter logic here
#         print(f"Filtering column {column}")

#         model = self.parent().model()

#         types = set()
#         for row in range(model.rowCount()):
#             item = model.index(row, column)
#             if item and item.data() is not None:
#                 types.add(type(item.data(Qt.ItemDataRole.UserRole)))
#         print('types', types)

#         # column_data = [model.data(model.index(row, column)) for row in range(model.rowCount())]

#         # if all(isinstance(col, (str, type(None))) for col in column_data):
#         #     data = [c for c in column_data if c is not None]
#         #     self.filter_menu = StringFilterMenu(data, column, self)
#         #     self.filter_menu.popup(QtGui.QCursor.pos())
#         #     return

#     def getButtonRect(self, rect):
#         return QRect(rect.left() + 2, rect.top() + 2, 18, 18)

#     def isButtonHovered(self, pos):
#         index = self.logicalIndexAt(pos)
#         if index != -1:
#             rect = self.rect()
#             rect.setLeft(self.sectionViewportPosition(index))
#             rect.setRight(self.sectionViewportPosition(index) + self.sectionSize(index) - 1)
#             button_rect = self.getButtonRect(rect)
#             return button_rect.contains(pos)
#         return False


class TableView(QtWidgets.QTableView):
    settings_changed = QtCore.pyqtSignal()
    log_view_requested = QtCore.pyqtSignal(int, int)  # proposal, run
    model_updated = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setAlternatingRowColors(False)

        # self.setSortingEnabled(True)
        self.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )

        # self.horizontalHeader().setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # self.horizontalHeader().customContextMenuRequested.connect(self.show_filter_menu)
        self.setHorizontalHeader(HeaderView(Qt.Horizontal, self))
        self.filter_menu = None

        self.verticalHeader().setMinimumSectionSize(ROW_HEIGHT)
        self.verticalHeader().setStyleSheet("QHeaderView"
                                            "{"
                                            "background:white;"
                                            "}")

        # Add the widgets to be used in the column settings dialog
        self._columns_widget = QtWidgets.QListWidget()
        self._static_columns_widget = QtWidgets.QListWidget()

        self._columns_widget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self._columns_widget.itemChanged.connect(self.item_changed)
        self._columns_widget.model().rowsMoved.connect(self.item_moved)

        self._columns_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self._columns_widget.customContextMenuRequested.connect(self.show_delete_menu)

        self._static_columns_widget.itemChanged.connect(self.item_changed)
        self._static_columns_widget.setStyleSheet("QListWidget {padding: 0px;} QListWidget::item { margin: 5px; }")
        self._columns_widget.setStyleSheet("QListWidget {padding: 0px;} QListWidget::item { margin: 5px; }")

        self.context_menu = QtWidgets.QMenu(self)
        self.zulip_action = QtWidgets.QAction('Export table to the Logbook', self)
        self.context_menu.addAction(self.zulip_action)
        self.show_logs_action = QtWidgets.QAction('View processing logs')
        self.show_logs_action.triggered.connect(self.show_run_logs)
        self.context_menu.addAction(self.show_logs_action)
        self.process_action = QtWidgets.QAction('Reprocess runs')
        self.context_menu.addAction(self.process_action)

    def setModel(self, model: 'DamnitTableModel'):
        """
        Overload of setModel() to make sure that we restyle the comment rows
        when the model is updated.
        """
        if (old_sel_model := self.selectionModel()) is not None:
            old_sel_model.deleteLater()
        if (old_model := self.model()) is not None:
            old_model.deleteLater()

        self.damnit_model = model
        
        sfpm = FilterProxy(self)
        sfpm.setSourceModel(model)
        sfpm.setSortRole(Qt.ItemDataRole.UserRole)  # Numeric sort where relevant
        super().setModel(sfpm)
        # TODO self.clear_filter_button.clicked.connect(self.model().clearFilter)
        # TODO model.columnsChanged.connect(self.populateColumnToggle)

        # When loading a new model, the saved column order is applied at the
        # model level (changing column logical indices). So we need to reset
        # any reordering from the view level, which maps logical indices to
        # different visual indices, to show the columns as in the model.
        # self.setHorizontalHeader(QtWidgets.QHeaderView(Qt.Horizontal, self))
        # self.horizontalHeader().sortIndicatorChanged.connect(self.style_comment_rows)
        # self.horizontalHeader().setSectionsClickable(True)
        if model is not None:
            self.model().rowsInserted.connect(self.style_comment_rows)
            self.model().rowsInserted.connect(self.resize_new_rows)
            self.model().columnsInserted.connect(self.on_columns_inserted)
            self.model().columnsRemoved.connect(self.on_columns_removed)
            self.resizeRowsToContents()
        
        self.model_updated.emit()

    def selected_rows(self):
        """Get indices of selected rows in the DamnitTableModel"""
        proxy_rows = self.selectionModel().selectedRows()
        # Translate indices in sorted proxy model back to the underlying model
        proxy = self.model()
        return [proxy.mapToSource(ix) for ix in proxy_rows]

    def item_changed(self, item):
        state = item.checkState()
        self.set_column_visibility(item.text(), state == Qt.Checked)

    def set_column_visibility(self, name, visible, for_restore=False):
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
            column_index = self.damnit_model.find_column(name, by_title=True)
        except KeyError:
            log.error("Could not find column %r to set visibility", name)
            return

        self.setColumnHidden(column_index, not visible)

        if for_restore:
            widget = self._columns_widget if \
                len(self._columns_widget.findItems(name, Qt.MatchExactly)) == 1 else \
                self._static_columns_widget

            # Try to find the column. Some, like 'comment_id' will not be in the
            # list shown to the user.
            matching_items = widget.findItems(name, Qt.MatchExactly)
            if len(matching_items) == 1:
                item = matching_items[0]
                item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
        else:
            self.settings_changed.emit()

    def item_moved(self, parent, start, end, destination, row):
        # Take account of the static columns
        col_offset = self._static_columns_widget.count()

        col_from = start + col_offset
        col_to = self._columns_widget.currentIndex().row() + col_offset

        self.horizontalHeader().moveSection(col_from, col_to)

        self.settings_changed.emit()

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
            name = self.damnit_model.column_title_to_id(item.text())
            self.confirm_delete_variable(name)

    def confirm_delete_variable(self, name):
        button = QMessageBox.warning(self, "Confirm deletion",
                                     f"You are about to permanently delete the variable <b>'{name}'</b> "
                                     "from the database and HDF5 files. This cannot be undone. "
                                     "Are you sure you want to continue?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     defaultButton=QMessageBox.No)
        if button == QMessageBox.Yes:
            model = self.damnit_model
            delete_variable(model.db, name)
            model.removeColumn(model.find_column(name, by_title=False))

    def add_new_columns(self, columns, statuses, positions = None):
        if positions is None:
            rows_count = self._columns_widget.count()
            positions = [ii + rows_count for ii in range(len(columns))]

        for column, status, position in zip(columns, statuses, positions):
            if column == "comment_id":
                continue

            item = QtWidgets.QListWidgetItem(column)
            self._columns_widget.insertItem(position, item)
            item.setCheckState(Qt.Checked if status else Qt.Unchecked)

    def on_columns_inserted(self, _parent, first, last):
        titles = [self.damnit_model.column_title(i) for i in range(first, last + 1)]
        self.add_new_columns(titles, [True for _ in list(titles)])

    def on_columns_removed(self, _parent, _first, _last):
        col_header_view = self.horizontalHeader()
        cols = []
        for logical_idx, title in enumerate(self.damnit_model.column_titles):
            visual_idx = col_header_view.visualIndex(logical_idx)
            visible = not col_header_view.isSectionHidden(logical_idx)
            cols.append((visual_idx, title, visible))

        # Put titles in display order (sort by visual indices)
        cols.sort()

        self.set_columns([c[1] for c in cols], [c[2] for c in cols])

    def set_columns(self, columns, statuses):
        self._columns_widget.clear()
        self._static_columns_widget.clear()

        static_columns = ["Status", "Proposal", "Run", "Timestamp", "Comment"]
        for column, status in zip(columns, statuses):
            if column in static_columns:
                item = QtWidgets.QListWidgetItem(column)
                self._static_columns_widget.addItem(item)
                item.setCheckState(Qt.Checked if status else Qt.Unchecked)
        self._static_columns_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
                                                  QtWidgets.QSizePolicy.Minimum)

        # Remove the static columns
        new_columns = []
        new_statuses = []
        for col, status in zip(columns, statuses):
            if col in static_columns:
                continue
            new_columns.append(col)
            new_statuses.append(status)

        self.add_new_columns(new_columns, new_statuses)

    def get_column_states(self):
        column_states = { }

        def add_column_states(widget):
            for row in range(widget.count()):
                item = widget.item(row)
                column_states[item.text()] = item.checkState() == Qt.Checked

        add_column_states(self._static_columns_widget)
        add_column_states(self._columns_widget)

        return column_states

    def style_comment_rows(self, *_):
        self.clearSpans()
        model : DamnitTableModel = self.damnit_model
        comment_col = model.find_column("Comment", by_title=True)
        timestamp_col = model.find_column("Timestamp", by_title=True)

        proxy_mdl = self.model()
        for row_ix in model.standalone_comment_rows():
            ix = proxy_mdl.mapFromSource(self.damnit_model.createIndex(row_ix, 0))
            self.setSpan(ix.row(), 0, 1, timestamp_col)
            self.setSpan(ix.row(), comment_col, 1, 1000)

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
        row = self.selected_rows()[0].row()
        prop, run = self.damnit_model.row_to_proposal_run(row)
        self.log_view_requested.emit(prop, run)

    def show_horizontal_header_menu(self, index, pos):
        menu = QMenu(self)
        sort_asc_action = QAction(icon(FA6S.arrow_up_short_wide), "Sort Ascending", self)
        sort_desc_action = QAction(icon(FA6S.arrow_down_wide_short), "Sort Descending", self)
        filter_action = QAction(icon(FA6S.filter), "Filter", self)
        hide_action = QAction(icon(FA6S.eye_slash), "Hide column", self)

        menu.addAction(sort_asc_action)
        menu.addAction(sort_desc_action)
        menu.addSeparator()
        menu.addAction(filter_action)
        menu.addSeparator()
        menu.addAction(hide_action)

        sort_asc_action.triggered.connect(lambda: self.sortByColumn(index, Qt.AscendingOrder))
        sort_desc_action.triggered.connect(lambda: self.sortByColumn(index, Qt.DescendingOrder))
        filter_action.triggered.connect(lambda: SearchMenu(index, self.model(), self).popup(pos))
        hide_action.triggered.connect(lambda: print("TODO"))

        menu.exec_(pos)


class DamnitTableModel(QtGui.QStandardItemModel):
    value_changed = QtCore.pyqtSignal(int, int, str, object)
    time_comment_changed = QtCore.pyqtSignal(int, str)
    run_visibility_changed = QtCore.pyqtSignal(int, bool)

    def __init__(self, db: DamnitDB, column_settings: dict, parent):
        self.column_ids, self.column_titles = self._load_columns(db, column_settings)
        n_run_rows = db.conn.execute("SELECT count(*) FROM run_info").fetchone()[0]
        n_cmnt_rows = db.conn.execute("SELECT count(*) FROM time_comments").fetchone()[0]
        log.info(f"Table will have {n_run_rows} runs & {n_cmnt_rows} standalone comments")

        super().__init__(n_run_rows + n_cmnt_rows, len(self.column_ids), parent)
        self.setHorizontalHeaderLabels(self.column_titles)
        self._main_window = parent
        self.is_sorted_by = ""
        self.is_sorted_order = None
        self.db = db
        self.column_index = {c: i for (i, c) in enumerate(self.column_ids)}
        self.run_index = {}  # {(proposal, run): row}
        self.standalone_comment_index = {}

        self._bold_font = QtGui.QFont()
        self._bold_font.setBold(True)

        # Empty spaces are not editable by default
        proto = QtGui.QStandardItem()
        proto.setEditable(False)
        self.setItemPrototype(proto)

        self.user_variables = db.get_user_variables()
        self.editable_columns = {"comment"} | {
            vv.name for vv in self.user_variables.values()
        }
        for col_id in self.editable_columns:
            col_ix = self.find_column(col_id)
            for r in range(self.rowCount()):
                # QStandardItem is editable by default
                self.setItem(r, col_ix, QtGui.QStandardItem())

        # Set up status column with checkboxes for runs
        checkbox_proto = self.itemPrototype().clone()
        checkbox_proto.setCheckable(True)
        checkbox_proto.setCheckState(Qt.Checked)
        for r in range(n_run_rows):
            self.setItem(r, 0, checkbox_proto.clone())

        self._load_from_db()

    @staticmethod
    def _load_columns(db: DamnitDB, col_settings):
        t0 = time.perf_counter()
        column_ids = (
                ["Status", "proposal", "run", "start_time", "comment"]
                + sorted(set(db.variable_names()) - {'comment'})
        )
        col_id_to_title = {
            "run": "Run",
            "proposal": "Proposal",
            "start_time": "Timestamp",
            "comment": "Comment",
        } | dict(
            db.conn.execute("""SELECT name, title FROM variables WHERE title NOT NULL""")
        )
        col_title_to_id = {t: n for (n, t) in col_id_to_title.items()}

        # Column settings store human friendly titles - convert to IDs
        saved_col_order = [col_title_to_id.get(c, c) for c in col_settings]

        # Strip missing columns
        saved_col_order = [col for col in saved_col_order if col in column_ids]

        # Sort columns such that all static columns (proposal, run, etc) are at
        # the beginning, followed by all the columns that have saved settings,
        # followed by all the other columns (i.e. comment_id and any new columns
        # added in between the last save and now).
        non_static_cols = column_ids[5:]
        sorted_cols = column_ids[:5]
        # Static columns are saved too to store their visibility, but we filter
        # them out here because they've already been added to the list.
        sorted_cols.extend([col for col in saved_col_order if col not in sorted_cols])
        # Add all other unsaved columns
        sorted_cols.extend([col for col in non_static_cols if col not in saved_col_order])

        column_titles = [col_id_to_title.get(c, c) for c in sorted_cols]

        t1 = time.perf_counter()
        log.info(f"Got columns in {t1 - t0:.3f} s")
        return sorted_cols, column_titles

    def text_item(self, value, display=None):
        if display is None:
            if value is None:
                display = ''
            elif isinstance(value, float):
                display = prettify_notation(value)
            else:
                display = str(value)
        item = self.itemPrototype().clone()
        # We will use UserRole data for sorting
        item.setData(value, role=Qt.ItemDataRole.UserRole)
        item.setData(display, role=Qt.ItemDataRole.DisplayRole)
        return item

    def image_item(self, png_data: bytes):
        item = self.itemPrototype().clone()
        item.setData(self.generateThumbnail(png_data), role=Qt.DecorationRole)
        item.setToolTip(
            f'<img src="data:image/png;base64,{b64encode(png_data).decode()}">'
        )
        return item

    def comment_item(self, text, comment_id=None):
        item = QtGui.QStandardItem(text)  # Editable by default
        item.setToolTip(text)
        if comment_id is not None:
            # For standalone comments, integer ID
            item.setData(comment_id, COMMENT_ID_ROLE)
        return item

    def new_item(self, value, column_id, max_diff, attrs):
        if is_png_bytes(value):
            return self.image_item(value)
        elif column_id == 'comment':
            return self.comment_item(value)
        elif column_id == 'start_time':
            return self.text_item(value, timestamp2str(value))
        else:
            item = self.text_item(value)
            item.setEditable(column_id in self.editable_columns)
            bold = attrs.get("bold")
            if bold is None:
                bold = (max_diff is not None) and max_diff > 1e-9
            if bold:
                item.setFont(self._bold_font)
            if (bg := attrs.get('background')) is not None:
                item.setBackground(QtGui.QBrush(QtGui.QColor(*bg)))
            return item

    def _load_from_db(self):
        t0 = time.perf_counter()

        row_headers = []
        row_ix = -1

        for row_ix, (prop, run, ts) in enumerate(self.db.conn.execute("""
            SELECT proposal, run, start_time FROM run_info ORDER BY proposal, run
        """).fetchall()):
            row_headers.append(str(run))
            self.run_index[(prop, run)] = row_ix
            self.setItem(row_ix, 1, self.text_item(prop))
            self.setItem(row_ix, 2, self.text_item(run))
            self.setItem(row_ix, 3, self.text_item(ts, timestamp2str(ts)))

        for (prop, run), grp in groupby(self.db.conn.execute("""
            SELECT proposal, run, name, value, max_diff, attributes FROM run_variables
            ORDER BY proposal, run
        """).fetchall(), key=lambda r: r[:2]):  # Group by proposal & run
            row_ix = self.run_index[(prop, run)]
            for *_, name, value, max_diff, attr_json in grp:
                col_ix = self.column_index[name]
                if name in self.user_variables:
                    value = self.user_variables[name].get_type_class().from_db_value(value)
                attrs = json.loads(attr_json) if attr_json else {}
                self.setItem(row_ix, col_ix, self.new_item(value, name, max_diff, attrs))

        comments_start = row_ix + 1
        comment_rows = self.db.conn.execute("""
            SELECT rowid, timestamp, comment FROM time_comments
        """).fetchall()
        for row_ix, (cid, ts, comment) in enumerate(comment_rows, start=comments_start):
            self.setItem(row_ix, 3, self.text_item(ts, timestamp2str(ts)))
            self.setItem(
                row_ix, self.column_index["comment"], self.comment_item(comment, cid)
            )
            row_headers.append('')
            self.standalone_comment_index[cid] = row_ix

        self.setVerticalHeaderLabels(row_headers)
        t1 = time.perf_counter()
        log.info(f"Filled rows in {t1 - t0:.3f} s")

    def has_column(self, name, by_title=False):
        if by_title:
            return name in self.column_titles
        else:
            return name in self.column_index

    def find_column(self, name, by_title=False):
        if by_title:
            try:
                return self.column_titles.index(name)
            except ValueError:  # Convert to KeyError, matching dict
                raise KeyError(name)
        else:
            return self.column_index[name]

    def column_title(self, col_ix):
        return self.column_titles[col_ix]

    def column_id(self, col_ix):
        return self.column_ids[col_ix]

    def column_title_to_id(self, title):
        return self.column_id(self.find_column(title, by_title=True))

    def computed_columns(self, by_title=False):
        for i, col_id in enumerate(self.column_ids[5:], start=5):
            if col_id not in self.editable_columns:
                if by_title:
                    yield self.column_titles[i]
                else:
                    yield col_id

    def find_row(self, proposal, run):
        return self.run_index[(proposal, run)]

    def row_to_proposal_run(self, row_ix):
        prop_col, run_col = 1, 2
        prop_it, run_it = self.item(row_ix, prop_col), self.item(row_ix, run_col)
        if prop_it is None:
            return None, None
        return prop_it.data(Qt.UserRole), run_it.data(Qt.UserRole)

    def row_to_comment_id(self, row):
        comment_col = 4
        item = self.item(row, comment_col)
        return item and item.data(COMMENT_ID_ROLE)

    def standalone_comment_rows(self):
        return sorted(self.standalone_comment_index.values())

    def precreate_runs(self, n_runs: int):
        proposal = self.db.metameta["proposal"]
        start_run = max([r for (p, r) in self.run_index if p == proposal]) + 1
        for run in range(start_run, start_run + n_runs):
            # To precreate the run we add it to the `run_info` table, and
            # the `run_variables` table with an empty comment. Adding it to
            # both ensures that the run will show up in the `runs` view.
            self.db.ensure_run(proposal, run)
            self.db.set_variable(proposal, run, "comment", ReducedData(None))

            self.insert_run_row(proposal, run, {}, {}, {})

    def insert_columns(self, before: int, titles, column_ids=None, type_cls=None, editable=False):
        if column_ids is None:
            column_ids = titles
        else:
            assert len(column_ids) == len(titles)

        self.column_ids[before:before] = column_ids
        self.column_titles[before:before] = titles
        self.column_index = {c: i for (i, c) in enumerate(self.column_ids)}
        if editable:
            self.editable_columns.update(column_ids)

        self.insertColumns(before, len(column_ids))

        for i, title in enumerate(titles, start=before):
            self.setHorizontalHeaderItem(before, QtGui.QStandardItem(title))

    def removeColumn(self, column: int, parent=QtCore.QModelIndex()):
        del self.column_ids[column]
        del self.column_titles[column]
        self.column_index = {c: i for (i, c) in enumerate(self.column_ids)}
        super().removeColumn(column, parent)

    def insert_run_row(self, proposal, run, contents: dict, max_diffs: dict, attrs: dict):
        status_item = self.itemPrototype().clone()
        status_item.setCheckable(True)
        status_item.setCheckState(Qt.Checked)
        row = [status_item, self.text_item(proposal), self.text_item(run)]

        for column_id in self.column_ids[3:]:
            if (value := contents.get(column_id, None)) is not None:
                item = self.new_item(
                    value, column_id, max_diffs.get(column_id) or 0, attrs.get(column_id) or {}
                )
            elif column_id in self.editable_columns:
                item = QtGui.QStandardItem()  # Editable by default
            else:
                item = None
            row.append(item)
        # We add new rows at the end, a QSortFilterProxyModel shows them in
        # the correct position given the current sort.
        self.run_index[(proposal, run)] = row_ix = self.rowCount()
        self.appendRow(row)
        self.setVerticalHeaderItem(row_ix, QtGui.QStandardItem(str(run)))

    def insert_comment_row(self, comment_id: int, comment: str, timestamp: float):
        blank = self.itemPrototype().clone()
        ts_item = self.text_item(timestamp, display=timestamp2str(timestamp))
        row = [blank, blank, blank, ts_item, self.comment_item(comment, comment_id)]
        self.standalone_comment_index[comment_id] = row_ix = self.rowCount()
        self.appendRow(row)
        self.setVerticalHeaderItem(row_ix, QtGui.QStandardItem(''))

    def handle_run_values_changed(self, proposal, run, values: dict):
        known_col_ids = set(self.column_ids)
        new_col_ids = [c for c in values if c not in known_col_ids]

        if new_col_ids:
            log.info("New columns for table: %s", new_col_ids)
            # TODO: retrieve titles for new columns
            self.insert_columns(self.columnCount(), new_col_ids)

        try:
            row_ix = self.find_row(proposal, run)
        except KeyError:
            row_ix = None

        max_diffs = {}
        attrs = {}
        for name, max_diff, attr_json in self.db.conn.execute("""
            SELECT name, max_diff, attributes FROM run_variables WHERE proposal=? AND run=?
        """, (proposal, run)):
            max_diffs[name] = max_diff
            attrs[name] = json.loads(attr_json) if attr_json else {}

        col_id_to_ix = {c: i for (i, c) in enumerate(self.column_ids)}

        if row_ix is not None:
            log.debug("Update existing row %s for run %s", row_ix, run)
            for column_id, value in values.items():
                col_ix = col_id_to_ix[column_id]
                self.setItem(row_ix, col_ix, self.new_item(
                    value, column_id, max_diffs.get(column_id) or 0, attrs.get(column_id) or {}
                ))
        else:
            self.insert_run_row(proposal, run, values, max_diffs, attrs)

    def handle_variable_set(self, var_info: dict):
        col_id = var_info['name']
        title = var_info['title'] or col_id
        try:
            col_ix = self.find_column(col_id)
        except KeyError:
            # New column
            end = self.columnCount()
            if var_info['type'] is None:
                self.insert_columns(end, [title], [col_id])
            else:
                type_cls = value_types_by_name[var_info['type']]
                self.insert_columns(
                    end, [title], [col_id], type_cls=type_cls, editable=True
                )
        else:
            # Update existing column
            old_title = self.column_title(col_ix)
            if title != old_title:
                self.column_titles[col_ix] = title
                self.setHorizontalHeaderItem(col_ix, QtGui.QStandardItem(title))

    def add_editable_column(self, name):
        if name == "Status":
            return
        self.editable_columns.add(name)

    def remove_editable_column(self, name):
        if name == "comment":
            return
        self.editable_columns.remove(name)

    @staticmethod
    def generateThumbnail(data: bytes) -> QtGui.QPixmap:
        """
        Helper function to generate a thumbnail from PNG data.
        """
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(data, "PNG")
        if max(pixmap.height(), pixmap.width()) > THUMBNAIL_SIZE:
            pixmap = pixmap.scaled(
                THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.KeepAspectRatio
            )
        return pixmap

    def numbers_for_plotting(self, *cols, by_title=True):
        col_ixs = [self.find_column(c, by_title) for c in cols]
        res = [[]  for _ in cols]
        for r in range(self.rowCount()):
            status_item = self.item(r, 0)
            if status_item is None or status_item.checkState() != Qt.Checked:
                continue

            vals = [self.get_value_at_rc(r, ci) for ci in col_ixs]
            if all(isinstance(val, (int, float)) for val in vals):
                for res_list, val in zip(res, vals):
                    res_list.append(val)

        return res

    def get_value_at(self, index):
        """Get the value for programmatic use, not for display"""
        return self.itemFromIndex(index).data(Qt.UserRole)

    def get_value_at_rc(self, row, col):
        item = self.item(row, col)
        return item.data(Qt.UserRole) if item is not None else None

    def setData(self, index, value, role=None) -> bool:
        if not index.isValid():
            return False

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            # A cell was edited in the table
            changed_column = self.column_id(index.column())
            if changed_column == "comment":
                if not super().setData(index, value, role):
                    return False
                parsed = value
            else:
                variable_type_class = self.user_variables[changed_column].get_type_class()

                try:
                    parsed = variable_type_class.parse(value) if value != '' else None
                except Exception:
                    self._main_window.show_status_message(
                        f'Value {value!r} is not valid for the {variable_type_class} column "{self.column_titles[index.column()]}"',
                        timeout=5000,
                        stylesheet=StatusbarStylesheet.ERROR
                    )
                    return False
                else:
                    if not super().setData(index, parsed, Qt.ItemDataRole.UserRole):
                        return False

                    if parsed is None:
                        display = ''
                    elif isinstance(parsed, float):
                        display = prettify_notation(value)
                    else:
                        display = str(parsed)
                    if not super().setData(index, display, role):
                        return False

            # Send appropriate signals if we edited a standalone comment or an
            # editable column.
            if comment_id := self.row_to_comment_id(index.row()):
                self.time_comment_changed.emit(comment_id, value)
            else:
                prop, run = self.row_to_proposal_run(index.row())
                self.value_changed.emit(int(prop), int(run), changed_column, parsed)

            return True

        elif role == Qt.ItemDataRole.CheckStateRole:
            if not super().setData(index, value, role):
                return False
            # CHeckboxes are only on the status column
            self.run_visibility_changed.emit(
                index.row(), (value == Qt.CheckState.Checked)
            )
            return True

        return super().setData(index, value, role)

    def dataframe_for_export(self, column_titles, rows=None, drop_image_cols=False):
        """Create a cleaned-up dataframe to be saved as a spreadsheet"""
        import pandas as pd
        col_titles_to_ixs = {t: i for (i, t) in enumerate(self.column_titles)}
        column_ixs = [col_titles_to_ixs[t] for t in column_titles]

        if rows is None:
            rows = range(self.rowCount())

        if drop_image_cols:
            filtered_ixs = []
            for col_ix in column_ixs:
                for row_ix in range(self.rowCount()):
                    item = self.item(row_ix, col_ix)
                    if item and is_png_bytes(item.data(Qt.UserRole)):
                        break
                else:
                    filtered_ixs.append(col_ix)
            column_ixs = filtered_ixs

        # Put the dtype options in an order so we can promote columns to more
        # general types based on their values.
        col_dtype_options = [
            pd.Float64Dtype(),  # Used if no values in selection
            pd.BooleanDtype(),
            pd.Int64Dtype(),
            pd.Float64Dtype(),
            pd.StringDtype(),   # For everything that's not covered above
        ]
        value_types = [type(None), bool, int, float]

        cols_dict = {}
        for col_ix in column_ixs:
            values = []
            col_type_ix = 0
            for row_ix in rows:
                item = self.item(row_ix, col_ix)
                if self.column_ids[col_ix] == 'start_time':
                    # Include timestamp as string
                    val = item and item.data(Qt.DisplayRole)
                else:
                    val = item and item.data(Qt.UserRole)
                    if val is None and item and item.data(Qt.DecorationRole):
                        val = "<image>"

                values.append(val)
                for i, pytype in enumerate(value_types):
                    if isinstance(val, pytype):
                        col_type_ix = max(col_type_ix, i)
                        break
                else:
                    col_type_ix = 4

            cols_dict[self.column_titles[col_ix]] = pd.Series(
                values, dtype=col_dtype_options[col_type_ix]
            )

        df = pd.DataFrame(cols_dict)

        row_labels = [self.headerData(r, Qt.Vertical) for r in rows]
        df.index = row_labels

        return df


def prettify_notation(value):
    if isinstance(value, float):
        if value % 1 == 0 and abs(value) < 10_000:
            # If it has no decimal places, display it as an int
            return f"{int(value)}"
        elif 0.0001 < abs(value) < 10_000:
            # If it's an easily-recognized range for numbers, display as a float
            return f"{value:.4f}"
        else:
            # Otherwise, display in scientific notation
            return f"{value:.3e}"
    return f"{value}"

def is_png_bytes(obj):
    return isinstance(obj, bytes) and BlobTypes.identify(obj) is BlobTypes.png
