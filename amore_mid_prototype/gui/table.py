import time
import pandas as pd
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt


class TableView(QtWidgets.QTableView):
    def __init__(self) -> None:
        super().__init__()
        self.setAlternatingRowColors(False)

        self.setSortingEnabled(True)
        self.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        # movable columns
        self.verticalHeader().setSectionsMovable(True)
        self.horizontalHeader().setSectionsMovable(True)
        self.setDragDropOverwriteMode(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )

    def state_changed(self, state, column_index):
        if Qt.CheckState.Checked == state:
            self.setColumnHidden(column_index, False)
        else:
            self.setColumnHidden(column_index, True)

    def set_item_columns_visibility(self, columns, status):
        if "Status" in columns:
            columns.remove("Status")

        for i in range(len(columns)):
            item = QtWidgets.QCheckBox(columns[i])
            item.setCheckable(True)
            item.setCheckState(
                Qt.CheckState.Checked if status[i] else Qt.CheckState.Unchecked
            )
            item.stateChanged.connect(
                lambda state, column_index=self.model()._data.columns.get_loc(
                    columns[i]
                ): self.state_changed(state, column_index)
            )

            self._columns_visibility_layout.insertWidget(
                self._columns_visibility_layout.count() - 1, item
            )

    def set_columns_visibility(self, columns, status):
        group = QtWidgets.QGroupBox("Show column")
        self._columns_visibility_layout = QtWidgets.QVBoxLayout()

        self._columns_visibility_layout.addStretch()

        self.set_item_columns_visibility(columns, status)

        group.setLayout(self._columns_visibility_layout)

        return group


class Table(QtCore.QAbstractTableModel):
    comment_changed = QtCore.pyqtSignal(int, int, str)

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

            elif (
                index.column() == self._data.columns.get_loc("Added at")
                if "Added at" in self._data.columns
                else False
            ):
                return time.strftime("%H:%M:%S %d/%m/%Y", time.localtime(value))

            elif pd.api.types.is_float(value):
                return str("{:.6f}".format(value))

            else:
                return str(value)

        elif role == Qt.ItemDataRole.CheckStateRole and index.column() == self._data.columns.get_loc(
            "Status"
        ):
            if self._data["Status"].iloc[index.row()]:
                return QtCore.Qt.Checked
            else:
                return QtCore.Qt.Unchecked

    def setData(self, index, value, role=None) -> bool:
        if not index.isValid():
            return False

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            self._data.iloc[index.row(), index.column()] = value
            self.dataChanged.emit(index, index)

            print(value)

            # Only comment column is editable
            if index.column() == self._data.columns.get_loc("Comment"):
                prop, run = self._data.iloc[index.row()][["Proposal", "Run"]]
                self.comment_changed.emit(int(prop), int(run), value)

        elif role == Qt.ItemDataRole.CheckStateRole:
            if self._data["Status"].iloc[index.row()]:
                self._data["Status"].values[index.row()] = False
            else:
                self._data["Status"].values[index.row()] = True

        return True

    def headerData(self, col, orientation, role):
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
        ):
            name = self._data.columns[col]
            return self._main_window.column_title(name)

    def flags(self, index) -> Qt.ItemFlag:

        if index.column() == self._data.columns.get_loc("Comment"):
            return (
                Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsEditable
            )

        elif index.column() == self._data.columns.get_loc("Status"):
            return (
                Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsUserCheckable
            )

        else:
            return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled

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
