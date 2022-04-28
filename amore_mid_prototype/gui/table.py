import time
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


class TableView(QtWidgets.QTableView):
    def __init__(self) -> None:
        super().__init__()
        self.setAlternatingRowColors(False)

        self.setSortingEnabled(True)
        self.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )

        self.horizontalHeader().sortIndicatorChanged.connect(self.style_comment_rows)

        # these are fixed
        self.static_columns = ["Timestamp", "Status", "Run", "Comment"]

        # these are hidden
        self.hidden_column = []

    def setModel(self, model):
        """
        Overload of setModel() to make sure that we restyle the comment rows
        when the model is updated.
        """
        super().setModel(model)
        self.model().rowsInserted.connect(self.style_comment_rows)

    def item_changed(self, item):
        state = item.checkState()
        column_index = self.model()._data.columns.get_loc(item.text())

        if Qt.Checked == state:
            self.setColumnHidden(column_index, False)
        else:
            self.setColumnHidden(column_index, True)

    def item_moved(self, parent, start, end, destination, row):
        # Take account of the static columns
        col_offset = len(self.static_columns)

        col_from = start + col_offset
        col_to = self._columns_widget.currentIndex().row() + col_offset

        self.horizontalHeader().moveSection(col_from, col_to)

    def set_item_columns_visibility(self, columns, status):
        if "Status" in columns:
            columns.remove("Status")

        for i in range(len(columns)):
            item = QtWidgets.QListWidgetItem(columns[i])
            item.setCheckState(Qt.Checked if status[i] else Qt.Unchecked)

            self._columns_widget.addItem(item)

    def set_columns_visibility(self, columns, statuses):
        group = QtWidgets.QGroupBox("Column settings")

        layout = QtWidgets.QVBoxLayout()

        # Add the widget for static columns
        #self._static_columns_widget = QtWidgets.QListWidget()
        
        #for column, status in zip(columns, statuses):
        #    if column in static_columns:
        #        item = QtWidgets.QListWidgetItem(column)
        #        item.setCheckState(Qt.Checked if status else Qt.Unchecked)
        #        self._static_columns_widget.addItem(item)
        #self._static_columns_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
        #                                    QtWidgets.QSizePolicy.Minimum)
        #self._static_columns_widget.itemChanged.connect(self.item_changed)

        # Remove the static columns
        columns, statuses = map(list, zip(*[x for x in zip(columns, statuses)
                                            if (x[0] not in self.static_columns and not x[0].startswith("_"))]))

        self._columns_widget = QtWidgets.QListWidget()
        self._columns_widget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self._columns_widget.itemChanged.connect(self.item_changed)
        self._columns_widget.model().rowsMoved.connect(self.item_moved)

        #self._static_columns_widget.setStyleSheet("QListWidget {padding: 0px;} QListWidget::item { margin: 5px; }")
        self._columns_widget.setStyleSheet("QListWidget {padding: 0px;} QListWidget::item { margin: 5px; }")

        self.set_item_columns_visibility(columns, statuses)

        #layout.addWidget(QtWidgets.QLabel("These columns can be hidden but not reordered:"))
        #layout.addWidget(self._static_columns_widget)
        layout.addWidget(QtWidgets.QLabel("Drag these columns to reorder them:"))
        layout.addWidget(self._columns_widget)
        group.setLayout(layout)

        return group

    def style_comment_rows(self, *_):
        self.clearSpans()
        data = self.model()._data

        comment_col = data.columns.get_loc("Comment")

        for row in data["_comment_id"].dropna().index:
            self.setSpan(row, comment_col, 1, 100)

class Table(QtCore.QAbstractTableModel):
    comment_changed = QtCore.pyqtSignal(int, int, str)
    time_comment_changed = QtCore.pyqtSignal(int, str)
    run_visibility_changed = QtCore.pyqtSignal(int, bool)

    def __init__(self, main_window):
        super().__init__()
        self._main_window = main_window
        self.is_sorted_by = ""
        self.is_sorted_order = None

    @property
    def _data(self):
        return self._main_window.data

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

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            value = self._data.iloc[index.row(), index.column()]

            if pd.isna(value) or index.column() == self._data.columns.get_loc("Status"):
                return None

            elif index.column() == self._data.columns.get_loc("Timestamp"):
                return time.strftime("%H:%M:%S %d/%m/%Y", time.localtime(value))

            elif pd.api.types.is_float(value):
                if value % 1 == 0:
                    # If it has no decimal places, display it as an int
                    return f"{int(value)}"
                elif 0.1 < value < 10_000:
                    # If it's an easily-recognized range for numbers, display as a float
                    return f"{value:.3f}"
                else:
                    # Otherwise, display in scientific notation
                    return f"{value:.3e}"

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
        
        elif role == QtCore.Qt.BackgroundRole: # and len(self.comment_index):
            if index.row() in self._data["_comment_id"].dropna().index:
                return QtGui.QBrush(Qt.yellow)

    def isCommentRow(self, row):
        return row in self._data["_comment_id"].dropna()

    def setData(self, index, value, role=None) -> bool:
        if not index.isValid():
            return False

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            self._data.iloc[index.row(), index.column()] = value
            self.dataChanged.emit(index, index)

            # Only comment column is editable
            if index.column() == self._data.columns.get_loc("Comment"):
                prop, run = self._data.iloc[index.row()][["Proposal", "Run"]]
                if not (pd.isna(prop) or pd.isna(run)):
                    self.comment_changed.emit(int(prop), int(run), value)
                else:
                    comment_id = self._data.iloc[index.row()]["_comment_id"]
                    if not pd.isna(comment_id):
                        self.time_comment_changed.emit(comment_id, value)

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

    def flags(self, index) -> Qt.ItemFlag:
        item_flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled

        if index.column() == self._data.columns.get_loc("Comment"):
            item_flags |= Qt.ItemIsEditable
        elif index.column() == self._data.columns.get_loc("Status"):
            item_flags |= Qt.ItemIsUserCheckable

        return item_flags

    def sort(self, column, order):
        self.is_sorted_by = self._data.columns.tolist()[column]
        self.is_sorted_order = order

        self.layoutAboutToBeChanged.emit()

        self._data.sort_values(
            self.is_sorted_by,
            ascending=order == Qt.SortOrder.AscendingOrder,
            inplace=True,
        )
        self._data.reset_index(inplace=True, drop=True)

        self.layoutChanged.emit()