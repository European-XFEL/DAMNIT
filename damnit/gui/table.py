import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from ..backend.api import delete_variable
from ..backend.db import ReducedData, BlobTypes, DamnitDB
from ..backend.user_variables import value_types_by_name
from ..util import StatusbarStylesheet, timestamp2str

log = logging.getLogger(__name__)

ROW_HEIGHT = 30
THUMBNAIL_SIZE = 35
# The actual threshold for long messages is around 6200
# not 10k, otherwise one gets a '414 URI Too Long' error

class TableView(QtWidgets.QTableView):
    settings_changed = QtCore.pyqtSignal()
    log_view_requested = QtCore.pyqtSignal(int, int)  # proposal, run

    def __init__(self) -> None:
        super().__init__()
        self.setAlternatingRowColors(False)

        self.setSortingEnabled(True)
        self.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )

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

    def setModel(self, model):
        """
        Overload of setModel() to make sure that we restyle the comment rows
        when the model is updated.
        """
        super().setModel(model)
        # When loading a new model, the saved column order is applied at the
        # model level (changing column logical indices). So we need to reset
        # any reordering from the view level, which maps logical indices to
        # different visual indices, to show the columns as in the model.
        self.setHorizontalHeader(QtWidgets.QHeaderView(Qt.Horizontal, self))
        self.horizontalHeader().sortIndicatorChanged.connect(self.style_comment_rows)
        self.horizontalHeader().setSectionsClickable(True)
        if model is not None:
            self.model().rowsInserted.connect(self.style_comment_rows)
            self.model().rowsInserted.connect(self.resize_new_rows)
            self.model().columnsInserted.connect(self.on_columns_inserted)
            self.model().columnsRemoved.connect(self.on_columns_removed)
            self.resizeRowsToContents()

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
        column_index = self.model().find_column(name, by_title=True)

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
        # Take account of the static columns, and the Status column
        col_offset = self._static_columns_widget.count() + 1

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
            name = self.model().column_title_to_id(item.text())
            self.confirm_delete_variable(name)

    def confirm_delete_variable(self, name):
        button = QMessageBox.warning(self, "Confirm deletion",
                                     f"You are about to permanently delete the variable <b>'{name}'</b> "
                                     "from the database and HDF5 files. This cannot be undone. "
                                     "Are you sure you want to continue?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     defaultButton=QMessageBox.No)
        if button == QMessageBox.Yes:
            model = self.model()
            delete_variable(model.db, name)
            model.removeColumn(model.find_column(name, by_title=False))

    def add_new_columns(self, columns, statuses, positions = None):
        if positions is None:
            rows_count = self._columns_widget.count()
            positions = [ii + rows_count for ii in range(len(columns))]
        for column, status, position in zip(columns, statuses, positions):
            if column in ["Status", "comment_id"]:
                continue

            item = QtWidgets.QListWidgetItem(column)
            self._columns_widget.insertItem(position, item)
            item.setCheckState(Qt.Checked if status else Qt.Unchecked)

    def on_columns_inserted(self, _parent, first, last):
        titles = [self.model().column_title(i) for i in range(first, last + 1)]
        self.add_new_columns(titles, [True for _ in list(titles)])

    def on_columns_removed(self, _parent, _first, _last):
        col_header_view = self.horizontalHeader()
        cols = []
        for logical_idx, title in enumerate(self.model().column_titles):
            visual_idx = col_header_view.visualIndex(logical_idx)
            visible = not col_header_view.isSectionHidden(logical_idx)
            cols.append((visual_idx, title, visible))

        # Put titles in display order (sort by visual indices)
        cols.sort()

        self.set_columns([c[1] for c in cols], [c[2] for c in cols])

    def set_columns(self, columns, statuses):
        self._columns_widget.clear()
        self._static_columns_widget.clear()

        static_columns = ["Proposal", "Run", "Timestamp", "Comment"]
        for column, status in zip(columns, statuses):
            if column in static_columns:
                item = QtWidgets.QListWidgetItem(column)
                self._static_columns_widget.addItem(item)
                item.setCheckState(Qt.Checked if status else Qt.Unchecked)
        self._static_columns_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
                                                  QtWidgets.QSizePolicy.Minimum)

        # Remove the static columns
        columns, statuses = map(list, zip(*[x for x in zip(columns, statuses)
                                            if x[0] not in static_columns]))
        self.add_new_columns(columns, statuses)

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
        model : DamnitTableModel = self.model()
        comment_col = model.find_column("Comment", by_title=True)
        timestamp_col = model.find_column("Timestamp", by_title=True)

        for row_ix in model.standalone_comment_rows():
            self.setSpan(row_ix, 0, 1, timestamp_col)
            self.setSpan(row_ix, comment_col, 1, 1000)

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
        row = self.selectionModel().selectedRows()[0].row()
        prop, run = self.model().row_to_proposal_run(row)
        self.log_view_requested.emit(prop, run)


class DamnitTableModel(QtCore.QAbstractTableModel):
    value_changed = QtCore.pyqtSignal(int, int, str, object)
    time_comment_changed = QtCore.pyqtSignal(int, str)
    run_visibility_changed = QtCore.pyqtSignal(int, bool)

    def __init__(self, db: DamnitDB, column_settings: dict, parent):
        super().__init__(parent)
        self._main_window = parent
        self.is_sorted_by = ""
        self.is_sorted_order = None

        self.db = db
        # self._data uses IDs ('xgm_intensity') as column names.
        # column_ids holds a matching list of titles ('XGM intensity')
        self._data, self.column_titles = self._load_from_db(db, column_settings)
        self.is_constant_df = self.load_max_diffs()
        self.user_variables = db.get_user_variables()
        self.editable_columns = {"comment"} | {
            vv.name for vv in self.user_variables.values()
        }

    @staticmethod
    def _load_from_db(db: DamnitDB, col_settings):
        df = pd.read_sql_query("SELECT * FROM runs", db.conn)
        df.insert(0, "Status", True)

        # Ensure that the comment column is in the right spot
        if "comment" not in df.columns:
            df.insert(4, "comment", pd.NA)
        else:
            comment_column = df.pop("comment")
            df.insert(4, "comment", comment_column)

        df.insert(len(df.columns), "comment_id", pd.NA)
        df.pop("added_at")

        # Read the comments and prepare them for merging with the main data
        comments_df = pd.read_sql_query(
            "SELECT rowid as comment_id, * FROM time_comments", db.conn
        )
        comments_df.insert(0, "run", pd.NA)
        comments_df.insert(1, "proposal", pd.NA)
        # Don't try to plot comments
        comments_df.insert(2, "Status", False)

        col_id_to_title = {
            "run": "Run",
            "proposal": "Proposal",
            "start_time": "Timestamp",
            "comment": "Comment",
        } | dict(
            db.conn.execute("""SELECT name, title FROM variables WHERE title NOT NULL""")
        )
        col_title_to_id = {t: n for (n, t) in col_id_to_title.items()}

        data = pd.concat([
            df, comments_df.rename(columns={"timestamp": "start_time"})
        ])

        for vv in db.get_user_variables().values():
            type_cls = vv.get_type_class()
            if vv.name in data:
                # Convert loaded data to the right pandas type
                data[vv.name] = type_cls.convert(data[vv.name])
            else:
                # Add an empty column (before restoring saved order)
                data[vv.name] = pd.Series(index=data.index, dtype=type_cls.type_instance)


        # Column settings store human friendly titles - convert to IDs
        saved_col_order = [col_title_to_id.get(c, c) for c in col_settings]
        df_cols = data.columns.tolist()

        # Strip missing columns
        saved_col_order = [col for col in saved_col_order if col in df_cols]

        # Sort columns such that all static columns (proposal, run, etc) are at
        # the beginning, followed by all the columns that have saved settings,
        # followed by all the other columns (i.e. comment_id and any new columns
        # added in between the last save and now).
        static_cols = df_cols[:5]
        non_static_cols = df_cols[5:]
        sorted_cols = static_cols
        # Static columns are saved too to store their visibility, but we filter
        # them out here because they've already been added to the list.
        sorted_cols.extend([col for col in saved_col_order if col not in sorted_cols])
        # Add all other unsaved columns
        sorted_cols.extend([col for col in non_static_cols if col not in saved_col_order])

        data = data[sorted_cols]
        column_titles = [col_id_to_title.get(c, c) for c in data.columns]
        return data, column_titles

    def has_column(self, name, by_title=False):
        if by_title:
            return name in self.column_titles
        else:
            return name in self._data.columns

    def find_column(self, name, by_title=False):
        if by_title:
            try:
                return self.column_titles.index(name)
            except ValueError:  # Convert to KeyError, matching .get_loc()
                raise KeyError(name)
        else:
            return self._data.columns.get_loc(name)

    def column_title(self, col_ix):
        return self.column_titles[col_ix]

    def column_id(self, col_ix):
        return self._data.columns[col_ix]

    def column_title_to_id(self, title):
        return self.column_id(self.find_column(title, by_title=True))

    def find_row(self, proposal, run):
        df = self._data
        row = df.loc[(df["proposal"] == proposal) & (df["run"] == run)]
        if row.size:
            return row.index[0]
        else:
            raise KeyError((proposal, run))

    def row_to_proposal_run(self, row_ix):
        return self._data.at[row_ix, "proposal"], self._data.at[row_ix, "run"]

    def standalone_comment_rows(self):
        return self._data["comment_id"].dropna().index.tolist()

    def precreate_runs(self, n_runs: int):
        proposal = self.db.metameta["proposal"]
        start_run = self._data["run"].max() + 1
        for run in range(start_run, start_run + n_runs):
            # To precreate the run we add it to the `run_info` table, and
            # the `run_variables` table with an empty comment. Adding it to
            # both ensures that the run will show up in the `runs` view.
            self.db.ensure_run(proposal, run)
            self.db.set_variable(proposal, run, "comment", ReducedData(None))

            self.insert_row({ "proposal": proposal, "run": run,
                              "comment": "", "Status": True })

    def insert_columns(self, before: int, titles, column_ids=None, type_cls=None, editable=False):
        if column_ids is None:
            column_ids = titles
        else:
            assert len(column_ids) == len(titles)
        dtype = 'object' if (type_cls is None) else type_cls.type_instance
        self.beginInsertColumns(QtCore.QModelIndex(), before, before + len(titles) - 1)
        for i, (title, column_id) in enumerate(zip(titles, column_ids)):
            self._data.insert(before + i, column_id, pd.Series(
                index=self._data.index, dtype=dtype
            ))
            self.column_titles.insert(before + i, title)
            if editable:
                self.add_editable_column(title)

        self.endInsertColumns()

    def removeColumns(self, column: int, count=1, parent=None):
        column_ids = self._data.columns[column:column+count]
        new_df = self._data.drop(columns=column_ids)
        new_is_constant_df = self.is_constant_df.drop(columns=column_ids)
        new_col_titles = self.column_titles[:column] + self.column_titles[column+count:]

        self.beginRemoveColumns(QtCore.QModelIndex(), column, column + count - 1)
        self._data = new_df
        self.is_constant_df = new_is_constant_df
        self.column_titles = new_col_titles
        self.endRemoveColumns()
        return True

    def insert_row(self, contents: dict):
        # Extract the high-rank arrays from the messages, because
        # DataFrames can only handle 1D cell elements by default. The
        # way around this is to create a column manually with a dtype of
        # 'object'.
        ndarray_cols = {k: v for (k, v) in contents.items()
                        if isinstance(v, np.ndarray) and v.ndim > 1}
        plain_cols = {k: v for (k, v) in contents.items() if k not in ndarray_cols}

        sort_col = self.is_sorted_by
        if sort_col and not pd.isna(plain_cols.get(sort_col)):
            newval = plain_cols[sort_col]
            if self.is_sorted_order == Qt.SortOrder.AscendingOrder:
                ix = self._data[sort_col].searchsorted(newval)
            else:
                ix_back = self._data[sort_col][::-1].searchsorted(newval)
                ix = len(self._data) - ix_back
        else:
            ix = len(self._data)
        log.debug("New row in table at index %d", ix)

        # Create a DataFrame with the new data to insert into the main table
        new_entries = pd.DataFrame(plain_cols, index=[0])

        # Insert columns with 'object' dtype for the special columns
        # with arrays that are >1D.
        for col_name, value in ndarray_cols.items():
            col = pd.Series([value], index=[0], dtype="object")
            new_entries.insert(len(new_entries.columns), col_name, col)

        new_df = pd.concat(
            [
                self._data.iloc[:ix],
                new_entries,
                self._data.iloc[ix:],
            ],
            ignore_index=True,
        )

        self.beginInsertRows(QtCore.QModelIndex(), ix, ix)
        self._data = new_df
        self.generateThumbnail.cache_clear()
        self.endInsertRows()

    def load_max_diffs(self):
        max_diff_df = pd.read_sql_query("SELECT * FROM max_diffs",
                                        self.db.conn,
                                        index_col=["proposal", "run"]).fillna(0)
        # 1e-9 is the default tolerance of np.isclose()
        return max_diff_df < 1e-9

    def handle_run_values_changed(self, proposal, run, values: dict):
        known_col_ids = set(self._data.columns)
        new_col_ids = [c for c in values if c not in known_col_ids]

        if new_col_ids:
            log.info("New columns for table: %s", new_col_ids)
            # TODO: retrieve titles for new columns
            self.insert_columns(self.columnCount(), new_col_ids)

        try:
            row_ix = self.find_row(proposal, run)
        except KeyError:
            row_ix = None

        self.is_constant_df = self.load_max_diffs()


        if row_ix is not None:
            log.debug("Update existing row %s for run %s", row_ix, run)
            for ki, vi in values.items():
                self._data.at[row_ix, ki] = vi

                index = self.index(row_ix, self._data.columns.get_loc(ki))
                self.dataChanged.emit(index, index)

        else:
            self.insert_row(values | {
                "proposal": proposal, "run": run, "comment": "", "Status": True
            })

    def handle_variable_set(self, var_info: dict):
        col_id = var_info['name']
        title = var_info['title']
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
                self.headerDataChanged.emit(Qt.Orientation.Horizontal, col_ix, col_ix)

    def add_editable_column(self, name):
        if name == "Status":
            return
        self.editable_columns.add(name)

    def remove_editable_column(self, name):
        if name == "comment":
            return
        self.editable_columns.remove(name)

    def rowCount(self, index=None) -> int:
        return self._data.shape[0]

    def columnCount(self, parent=None) -> int:
        return self._data.shape[1]

    def insertRows(self, row, rows=1, index=QtCore.QModelIndex()):
        self.beginInsertRows(QtCore.QModelIndex(), row, row + rows - 1)
        self.endInsertRows()

        return True

    def insertColumns(self, column, columns=1, index=QtCore.QModelIndex()):
        self.beginInsertColumns(QtCore.QModelIndex(), column, column + columns - 1)
        self.endInsertColumns()

        return True

    @lru_cache(maxsize=1000)
    def generateThumbnail(self, data: bytes) -> QtGui.QPixmap:
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

    def variable_is_constant(self, run, proposal, quantity):
        """
        Check if the variable at the given index is constant throughout the run.
        """
        # If the run/proposal is a pd.NA then the row belongs to a standalone comment
        if pd.isna(run) or pd.isna(proposal):
            return True

        try:
            return self.is_constant_df.loc[(proposal, run)][quantity].item()
        except KeyError:
            return True

    _supported_roles = (
        Qt.CheckStateRole,
        Qt.DecorationRole,
        Qt.DisplayRole,
        Qt.EditRole,
        Qt.FontRole,
        Qt.ToolTipRole
    )

    def column_series(self, name, by_title=False, filter_status=True):
        if by_title:
            col_id = self.column_title_to_id(name)
        else:
            col_id = name
        series =  self._data[col_id].values
        if filter_status:
            series = series[self._data["Status"]]
        return series


    def get_value_at(self, index):
        """Get the value for programmatic use, not for display"""
        return self._data.iloc[index.row(), index.column()]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role not in self._supported_roles:
            return  # Fast exit for unused roles

        if not index.isValid():
            return

        r, c = index.row(), index.column()
        value = self._data.iat[r, c]

        if role == Qt.FontRole:
            # If the variable is not constant, make it bold
            proposal, run = self.row_to_proposal_run(r)
            quantity = self.column_id(c)
            if not self.variable_is_constant(run, proposal, quantity):
                font = QtGui.QFont()
                font.setBold(True)
                return font

        elif role == Qt.DecorationRole:
            if isinstance(value, bytes) and BlobTypes.identify(value) is BlobTypes.png:
                return self.generateThumbnail(value)

        elif role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if isinstance(value, bytes):
                # The image preview for this is taken care of by the DecorationRole
                return None

            elif pd.isna(value) or index.column() == self._data.columns.get_loc("Status"):
                return None

            elif index.column() == self._data.columns.get_loc("start_time"):
                return timestamp2str(value)

            elif pd.api.types.is_float(value):
                return prettify_notation(value)

            else:
                return str(value)

        elif role == Qt.ItemDataRole.CheckStateRole \
             and index.column() == self._data.columns.get_loc("Status") \
             and not self.isCommentRow(index.row()):
            if self._data["Status"].iloc[index.row()]:
                return QtCore.Qt.Checked
            else:
                return QtCore.Qt.Unchecked

        elif role == Qt.ToolTipRole:
            if index.column() == self._data.columns.get_loc("comment"):
                return self.data(index)

    def isCommentRow(self, row):
        return row in self._data["comment_id"].dropna()

    def setData(self, index, value, role=None) -> bool:
        if not index.isValid():
            return False

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            # Change the value in the table
            changed_column = self.column_id(index.column())
            if changed_column == "comment":
                self._data.iloc[index.row(), index.column()] = value
            else:
                variable_type_class = self.user_variables[changed_column].get_type_class()

                try:
                    value = variable_type_class.convert(value, unwrap=True) if value != '' else None
                    self._data.iloc[index.row(), index.column()] = value
                except Exception:
                    self._main_window.show_status_message(
                        f'Value {value!r} is not valid for the {variable_type_class} column "{self._data.columns[index.column()]}"',
                        timeout=5000,
                        stylesheet=StatusbarStylesheet.ERROR
                    )
                    return False

            self.dataChanged.emit(index, index)

            # Send appropriate signals if we edited a standalone comment or an
            # editable column.
            prop, run = self.row_to_proposal_run(index.row())

            if pd.isna(prop) and pd.isna(run) and changed_column == "comment":
                comment_id = self._data.iloc[index.row()]["comment_id"]
                if not pd.isna(comment_id):
                    self.time_comment_changed.emit(comment_id, value)
            elif self._data.columns[index.column()] in self.editable_columns:
                if not (pd.isna(prop) or pd.isna(run)):
                    self.value_changed.emit(int(prop), int(run), changed_column, value)

        elif role == Qt.ItemDataRole.CheckStateRole:
            new_state = not self._data["Status"].iloc[index.row()]
            self._data["Status"].values[index.row()] = new_state
            self.run_visibility_changed.emit(index.row(), new_state)

        return True

    def headerData(self, ix, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
        ):
            return self.column_title(ix)
        elif(
            orientation == Qt.Orientation.Vertical
            and role == Qt.ItemDataRole.DisplayRole
        ):
            row = self._data.iloc[ix]['run']
            if pd.isna(row):
                row = ''
            return row

    def flags(self, index) -> Qt.ItemFlag:
        item_flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled

        if self._data.columns[index.column()] in self.editable_columns:
            item_flags |= Qt.ItemIsEditable
        elif index.column() == self._data.columns.get_loc("Status"):
            item_flags |= Qt.ItemIsUserCheckable

        return item_flags

    def sort(self, column, order):
        is_sorted_by = self._data.columns.tolist()[column]

        self.layoutAboutToBeChanged.emit()

        try:
            self._data.sort_values(
                is_sorted_by,
                ascending=order == Qt.SortOrder.AscendingOrder,
                inplace=True,
            )
        except ValueError:
            QtWidgets.QMessageBox.warning(self._main_window, "Sorting error",
                                          "This column cannot be sorted")
        else:
            self.is_sorted_by = is_sorted_by
            self.is_sorted_order = order
            self._data.reset_index(inplace=True, drop=True)
        finally:
            self.layoutChanged.emit()

    def dataframe_for_export(self, column_titles, rows=None, drop_image_cols=False):
        """Create a cleaned-up dataframe to be saved as a spreadsheet"""
        # Helper function to convert image blobs to a string tag.
        def image2str(value):
            if isinstance(value, bytes) and BlobTypes.identify(value) is BlobTypes.png:
                return "<image>"
            else:
                return value

        col_ids_to_titles = dict(zip(self._data.columns, self.column_titles))
        col_titles_to_ids = dict(zip(self.column_titles, self._data.columns))
        column_ids = [col_titles_to_ids[t] for t in column_titles]

        if drop_image_cols:
            image_cols = self._columns_with_thumbnails()
            column_ids = [c for c in column_ids if c not in image_cols]

        cleaned_df = self._data[column_ids].copy()
        if rows is not None:
            cleaned_df = cleaned_df.iloc[rows]

        # Use human-friendly column names
        cleaned_df = cleaned_df.rename(columns=col_ids_to_titles)

        # Format timestamps nicely
        if "Timestamp" in cleaned_df:
            cleaned_df["Timestamp"] = cleaned_df["Timestamp"].map(timestamp2str)

        # Format images nicely
        return cleaned_df.applymap(image2str)

    def _columns_with_thumbnails(self):
        obj_columns = self._data.dtypes == 'object'
        image_cols = set()
        for column in obj_columns.index:
            for item in self._data[column]:
                if isinstance(item, bytes) and BlobTypes.identify(item) is BlobTypes.png:
                    image_cols.add(column)
                    break

        return image_cols

def prettify_notation(value):
    if pd.api.types.is_float(value):
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


