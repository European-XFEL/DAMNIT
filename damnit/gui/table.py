import json
import logging
import time
from base64 import b64encode
from itertools import groupby

from fonticon_fa6 import FA6S
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QAction, QMenu, QMessageBox
from superqt.fonticon import icon
from superqt.utils import qthrottled

from ..backend.db import BlobTypes, DamnitDB, ReducedData, blob2complex
from ..backend.extraction_control import ExtractionJobTracker
from ..backend.user_variables import value_types_by_name
from ..util import timestamp2str
from .table_filter import FilterMenu, FilterProxy, FilterStatus
from .util import delete_variable, StatusbarStylesheet

log = logging.getLogger(__name__)

ROW_HEIGHT = 30
THUMBNAIL_SIZE = (100, 35)  # w, h (pixels)


class FilterHeaderView(QtWidgets.QHeaderView):
    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.filtered_columns = set()
        self.filter_icon = icon(FA6S.filter)

    def paintSection(self, painter, rect, logicalIndex):
        painter.save()
        
        # Draw the default header section
        super().paintSection(painter, rect, logicalIndex)
        
        # If column is filtered, draw the icon
        if logicalIndex in self.filtered_columns:
            # Calculate icon position
            icon_size = 16
            padding = 4
            icon_rect = QtCore.QRect(
                rect.right() - icon_size - padding,
                rect.center().y() - icon_size//2,
                icon_size,
                icon_size
            )
            # Draw the icon using QPainter's drawIcon
            painter.setClipRect(rect)
            painter.drawPixmap(icon_rect, self.filter_icon.pixmap(icon_size))
            painter.setClipRect(rect)

        painter.restore()

    def update_filtered_columns(self, filtered_cols):
        """Update the set of filtered columns and trigger repaint"""
        self.filtered_columns = set(filtered_cols)
        self.viewport().update()


class TableView(QtWidgets.QTableView):
    settings_changed = QtCore.pyqtSignal()
    log_view_requested = QtCore.pyqtSignal(int, int)  # proposal, run
    model_updated = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlternatingRowColors(False)

        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )

        self.verticalHeader().setMinimumSectionSize(ROW_HEIGHT)

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
        self.show_logs_action.setEnabled(False)  # Enabled with selection
        self.show_logs_action.triggered.connect(self.show_run_logs)
        self.context_menu.addAction(self.show_logs_action)
        self.process_action = QtWidgets.QAction('Reprocess runs')
        self.context_menu.addAction(self.process_action)

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

        self.damnit_model = model

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
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self.show_horizontal_header_menu)
        # header.setSectionsMovable(True)  # TODO need to update variable order in the table / emit settings_changed

        if model is not None:
            self.model().rowsInserted.connect(self.resize_new_rows)
            self.model().columnsInserted.connect(self.on_columns_inserted)
            self.model().columnsRemoved.connect(self.on_columns_removed)
            self.model().filterChanged.connect(
                lambda: header.update_filtered_columns(self.model().filters))
            self.resizeRowsToContents()

        self.selectionModel().selectionChanged.connect(self.selection_changed)

        self.model_updated.emit()

    def selected_rows(self):
        """Get indices of selected rows in the DamnitTableModel"""
        proxy_rows = self.selectionModel().selectedRows()
        # Translate indices in sorted proxy model back to the underlying model
        proxy = self.model()
        return [proxy.mapToSource(ix) for ix in proxy_rows]

    def selection_changed(self):
        has_sel = self.selectionModel().hasSelection()
        self.show_logs_action.setEnabled(has_sel)

    def item_changed(self, item):
        state = item.checkState()
        self.set_column_visibility(item.text(), state == Qt.Checked)

    def set_column_visibility(self, name, visible, for_restore=False, save_settings=True):
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
        elif save_settings:
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

        menu.exec_(QtGui.QCursor.pos())

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
                self.set_column_visibility(col, state, save_settings=False)

            self._tag_filter_button.setText("Variables by Tag")
        else:
            # Get all variables that have any of the selected tags
            tagged_vars = set()
            for tag in tag_names:
                tagged_vars.update(self.damnit_model.db.get_variables_by_tag(tag))

            # Hide/show columns based on whether they're tagged AND checked in column widgets
            for idx, (col, state) in enumerate(column_states.items()):
                is_static = idx < self.get_static_columns_count()
                is_tagged = self.damnit_model.column_title_to_id(col) in tagged_vars

                # Column should be visible if:
                # 1. It's a static column that's checked in column widgets, or
                # 2. It's a non-static column that's both tagged and checked in column widgets
                show = state and (is_static or is_tagged)
                self.set_column_visibility(col, show, save_settings=False)

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

        sort_asc_action.triggered.connect(lambda: self.sortByColumn(index, Qt.AscendingOrder))
        sort_desc_action.triggered.connect(lambda: self.sortByColumn(index, Qt.DescendingOrder))
        filter_action.triggered.connect(lambda: FilterMenu(index, self.model(), self).popup(pos))

        menu.exec_(pos)


class DamnitTableModel(QtGui.QStandardItemModel):
    value_changed = QtCore.pyqtSignal(int, int, str, object)
    time_comment_changed = QtCore.pyqtSignal(int, str)
    run_visibility_changed = QtCore.pyqtSignal(int, bool)

    def __init__(self, db: DamnitDB, column_settings: dict, parent):
        self.column_ids, self.column_titles = self._load_columns(db, column_settings)
        n_run_rows = db.conn.execute("SELECT count(*) FROM run_info").fetchone()[0]
        log.info(f"Table will have {n_run_rows} runs")

        super().__init__(n_run_rows, len(self.column_ids), parent)
        self.setHorizontalHeaderLabels(self.column_titles)
        self._main_window = parent
        self.is_sorted_by = ""
        self.is_sorted_order = None
        self.db = db
        self.column_index = {c: i for (i, c) in enumerate(self.column_ids)}
        self.run_index = {}  # {(proposal, run): row}

        self.processing_jobs = QtExtractionJobTracker(self)
        self.processing_jobs.run_jobs_changed.connect(self.update_processing_status)

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

    def comment_item(self, text):
        item = self.text_item(text)
        item.setToolTip(text)
        item.setEditable(True)
        return item

    def error_item(self, attrs):
        item = self.itemPrototype().clone()
        msg = attrs['error']
        match attrs.get('error_cls', ''):
            case 'Skip':
                colour = 'lightgrey'
                msg = "Skipped: " + msg
            case 'SourceNameError':  # Typically an issue with data, not code
                colour = 'lightgrey'
            case cls:
                colour = 'orange'
                msg = f"{cls}: {msg} (see processing log for details)"
        item.setToolTip(msg)
        item.setData(QtGui.QColor(colour), Qt.ItemDataRole.DecorationRole)
        return item

    def new_item(self, value, column_id, max_diff, attrs):
        if is_png_bytes(value):
            return self.image_item(value)
        elif column_id == 'comment':
            return self.comment_item(value)
        elif column_id == 'start_time':
            return self.text_item(value, timestamp2str(value))
        elif 'error' in attrs:
            return self.error_item(attrs)
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
            SELECT proposal, run, name, value, max_diff, summary_type, attributes FROM run_variables
            ORDER BY proposal, run
        """).fetchall(), key=lambda r: r[:2]):  # Group by proposal & run
            row_ix = self.run_index[(prop, run)]
            for *_, name, value, max_diff, summary_type, attr_json in grp:
                col_ix = self.column_index[name]
                if name in self.user_variables:
                    value = self.user_variables[name].get_type_class().from_db_value(value)
                if summary_type == "complex":
                    value = blob2complex(value)
                attrs = json.loads(attr_json) if attr_json else {}
                self.setItem(row_ix, col_ix, self.new_item(value, name, max_diff, attrs))

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

    def precreate_runs(self, n_runs: int):
        proposal = self.db.metameta["proposal"]
        start_run = max(
            [r for (p, r) in self.run_index if p == proposal], default=0
        ) + 1

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
        return row_ix

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

    def handle_processing_state_set(self, info):
        self.processing_jobs.on_processing_state_set(info)

    def handle_processing_finished(self, info):
        self.processing_jobs.on_processing_finished(info)

    def update_processing_status(self, proposal, run, jobs_for_run):
        """Show/hide the processing indicator for the given run"""
        try:
            row_ix = self.find_row(proposal, run)
        except KeyError:
            if jobs_for_run:
                row_ix = self.insert_run_row(proposal, run, {}, {}, {})
            else:
                return

        running = [j for j in jobs_for_run if j['status'] == 'RUNNING']

        row_header_item = self.verticalHeaderItem(row_ix)
        if running:
            row_header_item.setData(f"{run} ⚙️", Qt.ItemDataRole.DisplayRole)
            if len(running) == 1:
                info = running[0]
                msg = f"Processing on {info['username']}@{info['hostname']}"
                if job_id := info['slurm_job_id']:
                    msg += f" (Slurm job {job_id})"
                row_header_item.setToolTip(msg)
            else:
                row_header_item.setToolTip(f"Processing in {len(running)} jobs")
        elif jobs_for_run:
            # Jobs in the list but not running must be pending
            row_header_item.setData(f"{run} ⋮", Qt.ItemDataRole.DisplayRole)
            row_header_item.setToolTip("Processing is queued")
        else:
            row_header_item.setData(f"{run}", Qt.ItemDataRole.DisplayRole)
            row_header_item.setToolTip("")

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
        if pixmap.width() > THUMBNAIL_SIZE[0] or pixmap.height() > THUMBNAIL_SIZE[1]:
            pixmap = pixmap.scaled(*THUMBNAIL_SIZE, Qt.KeepAspectRatio)
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

    # QStandardItemModel assumes empty cells (no item) can be edited, regardless
    # of itemPrototype. This override prevents editing empty cells.
    def flags(self, model_index):
        # Not using itemFromIndex() here, as it creates & inserts items
        if model_index.isValid() and model_index.model() is self:
            itm = self.item(model_index.row(), model_index.column())
            if itm is not None:
                return itm.flags()

        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled |Qt.ItemIsDropEnabled

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


class QtExtractionJobTracker(ExtractionJobTracker, QtCore.QObject):
    run_jobs_changed = QtCore.pyqtSignal(int, int, object) # prop, run, jobs

    def __init__(self, parent):
        super().__init__()
        QtCore.QObject.__init__(self, parent)

        # Check for crashed Slurm jobs every 2 minutes
        self.slurm_check_timer = QtCore.QTimer(self)
        self.slurm_check_timer.timeout.connect(self.check_slurm_jobs)
        self.slurm_check_timer.start(120_000)

    def squeue_check_jobs(self, cmd, jobs_to_check):
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ForwardedErrorChannel)

        def done():
            proc.deleteLater()
            if proc.exitStatus() != QProcess.NormalExit or proc.exitCode() != 0:
                log.warning("Error calling squeue")
                return
            stdout = bytes(proc.readAllStandardOutput()).decode()
            self.process_squeue_output(stdout, jobs_to_check)

        proc.finished.connect(done)
        proc.start(cmd[0], cmd[1:])

    def on_run_jobs_changed(self, proposal, run):
        jobs = [i for i in self.jobs.values()
                if i['proposal'] == proposal and i['run'] == run]
        self.run_jobs_changed.emit(proposal, run, jobs)


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
