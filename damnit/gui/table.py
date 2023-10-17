from functools import lru_cache

import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt

from ..util import StatusbarStylesheet, timestamp2str

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

        self.horizontalHeader().sortIndicatorChanged.connect(self.style_comment_rows)
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

        self._static_columns_widget.itemChanged.connect(self.item_changed)
        self._static_columns_widget.setStyleSheet("QListWidget {padding: 0px;} QListWidget::item { margin: 5px; }")
        self._columns_widget.setStyleSheet("QListWidget {padding: 0px;} QListWidget::item { margin: 5px; }")

        self.context_menu = QtWidgets.QMenu(self)
        self.zulip_action = QtWidgets.QAction('Export table to the Logbook', self)
        self.zulip_action.triggered.connect(self.export_selection_to_zulip)
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
        if model is not None:
            self.model().rowsInserted.connect(self.style_comment_rows)
            self.model().rowsInserted.connect(self.resize_new_rows)
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
        column_index = self.model()._data.columns.get_loc(name)

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
        data = self.model()._data

        comment_col = data.columns.get_loc("Comment")
        timestamp_col = data.columns.get_loc("Timestamp")

        for row in data["comment_id"].dropna().index:
            self.setSpan(row, 0, 1, timestamp_col)
            self.setSpan(row, comment_col, 1, 1000)

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

    def export_selection_to_zulip(self):
        zulip_ok = self.model()._main_window.check_zulip_messenger()
        if not zulip_ok:
            return
            
        selected_rows = [r.row() for r in 
                         self.selectionModel().selectedRows()]
        df = pd.DataFrame(self.model()._main_window.data)
        df = df.iloc[selected_rows]
        
        blacklist_columns = ['Proposal', 'Status']
        blacklist_columns = blacklist_columns + self.columns_with_thumbnails(self.model()._data) \
            + self.columns_invisible(df)
        blacklist_columns = list(dict.fromkeys(blacklist_columns))
        sorted_columns =['Run', 'Timestamp', 'Comment'] + \
            [self._columns_widget.item(i).text() for i in range(self._columns_widget.count())]
        
        columns = [column for column in sorted_columns if column not in blacklist_columns] 
        df = pd.DataFrame(df, columns=columns)
        df.sort_values('Run', axis=0, inplace=True)
        
        if 'Timestamp' in df.columns:
            df['Timestamp'] = df['Timestamp'].apply(timestamp2str)
        
        df = df.applymap(prettify_notation)
        df.replace(["None", '<NA>', 'nan'], '', inplace=True)
        self.model()._main_window.zulip_messenger.send_table(df)

    def show_run_logs(self):
        # Get first selected row
        row = self.selectionModel().selectedRows()[0].row()
        prop, run = self.model().row_to_proposal_run(row)
        self.log_view_requested.emit(prop, run)
        
    def columns_with_thumbnails(self, df):
        obj_columns = df.dtypes == 'object'
        blacklist_columns = []
        for column in obj_columns.index:
            for item in df[column]:
                if isinstance(item, np.ndarray):
                    blacklist_columns.append(column)
                    break
                   
        return blacklist_columns
    
    def columns_invisible(self, df):
        blacklist_columns = []
        for column in range(0,self.model().columnCount()-1):
            if self.isColumnHidden(column):
                blacklist_columns.append(df.columns[column])
        
        return blacklist_columns

    
class Table(QtCore.QAbstractTableModel):
    value_changed = QtCore.pyqtSignal(int, int, str, object)
    time_comment_changed = QtCore.pyqtSignal(int, str)
    run_visibility_changed = QtCore.pyqtSignal(int, bool)

    def __init__(self, main_window):
        super().__init__()
        self._main_window = main_window
        self.is_sorted_by = ""
        self.is_sorted_order = None
        self.editable_columns = {"Comment"}

    @property
    def _data(self):
        return self._main_window.data

    def add_editable_column(self, name):
        if name == "Status":
            return
        self.editable_columns.add(name)

    def remove_editable_column(self, name):
        if name == "Comment":
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
    def generateThumbnail(self, run, proposal, quantity) -> QtGui.QPixmap:
        """
        Helper function to generate a thumbnail for a 2D array.

        Note that we require an index for self._data as the argument instead of
        an ndarray. That's because this function is quite expensive and called
        very often by self.data(), so for performance we try to cache the
        thumbnails with @lru_cache. Unfortunately ndarrays are not hashable so
        we have to take an index instead.
        """
        df_row = self._data.loc[(self._data["Run"] == run) & (self._data["Proposal"] == proposal)]
        image = df_row[quantity].item()

        height, width = image.shape[:2]
        image = QtGui.QImage(image.data, width, height, QtGui.QImage.Format_ARGB32)
        return QtGui.QPixmap(image).scaled(QtCore.QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE),
                                           Qt.KeepAspectRatio)

    def variable_is_constant(self, run, proposal, quantity):
        """
        Check if the variable at the given index is constant throughout the run.
        """
        is_constant_df = self._main_window.is_constant_df
        if quantity in is_constant_df.columns:
            return is_constant_df.loc[(proposal, run)][quantity].item()
        else:
            return True

    _supported_roles = (
        Qt.CheckStateRole,
        Qt.DecorationRole,
        Qt.DisplayRole,
        Qt.EditRole,
        Qt.FontRole,
        Qt.ToolTipRole
    )

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role not in self._supported_roles:
            return  # Fast exit for unused roles

        if not index.isValid():
            return

        r, c = index.row(), index.column()
        value = self._data.iat[r, c]
        run = self._data.iat[r, self._data.columns.get_loc("Run")]
        proposal = self._data.iat[r, self._data.columns.get_loc("Proposal")]
        quantity_title = self._data.columns[index.column()]
        quantity = self._main_window.col_title_to_name(quantity_title)

        if role == Qt.FontRole:
            # If the variable is not constant, make it bold
            if not self.variable_is_constant(run, proposal, quantity):
                font = QtGui.QFont()
                font.setBold(True)
                return font

        elif role == Qt.DecorationRole:
            if isinstance(value, np.ndarray):
                return self.generateThumbnail(run, proposal, quantity_title)
        elif role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if isinstance(value, np.ndarray):
                # The image preview for this is taken care of by the DecorationRole
                return None

            elif pd.isna(value) or index.column() == self._data.columns.get_loc("Status"):
                return None

            elif index.column() == self._data.columns.get_loc("Timestamp"):
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
            if index.column() == self._data.columns.get_loc("Comment"):
                return self.data(index)

    def isCommentRow(self, row):
        return row in self._data["comment_id"].dropna()

    def setData(self, index, value, role=None) -> bool:
        if not index.isValid():
            return False

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            # Change the value in the table
            changed_column = self._main_window.col_title_to_name(self._data.columns[index.column()])
            if changed_column == "comment":
                self._data.iloc[index.row(), index.column()] = value
            else:
                variable_type_class = self._main_window.get_variable_from_name(changed_column).get_type_class()

                try:
                    value = variable_type_class.convert(value, unwrap=True) if value != '' else None
                    self._data.iloc[index.row(), index.column()] = value
                except Exception:
                    self._main_window.show_status_message(
                        f"Value \"{value}\" is not valid for the \"{self._data.columns[index.column()]}\" column of type \"{variable_type_class}\".",
                        timeout=5000,
                        stylesheet=StatusbarStylesheet.ERROR
                    )
                    return False

            self.dataChanged.emit(index, index)

            # Send appropriate signals if we edited a standalone comment or an
            # editable column.
            prop, run = self._data.iloc[index.row()][["Proposal", "Run"]]

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

    def headerData(self, col, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
        ):
            name = self._data.columns[col]
            return self._main_window.column_title(name)
        elif(
            orientation == Qt.Orientation.Vertical
            and role == Qt.ItemDataRole.DisplayRole
        ):
            row = self._data.iloc[col]['Run']
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

    def row_to_proposal_run(self, row_ix):
        r = self._data.iloc[row_ix]
        return r['Proposal'], r['Run']

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


