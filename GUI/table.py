import time
from PyQt6 import QtCore, QtWidgets

class TableView(QtWidgets.QTableView):
    def __init__(self) -> None:
        super().__init__()

        #self.setSortingEnabled(True)

        # movable columns
        self.verticalHeader().setSectionsMovable(True)
        self.horizontalHeader().setSectionsMovable(True)
        self.setDragDropOverwriteMode(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)


class Table(QtCore.QAbstractTableModel):
    def __init__(self, data) -> None:
        super().__init__()
        self._data = data

    def rowCount(self, index=None) -> None:
        return self._data.shape[0]

    def columnCount(self, parent=None) -> None:
        return self._data.shape[1]

    def insertRows(self, row, rows=1, index=QtCore.QModelIndex()):
        self.beginInsertRows(QtCore.QModelIndex(), row, row + rows - 1)
        self.endInsertRows()

        return True
    
    def insertColumns(self, column, columns=1, index=QtCore.QModelIndex()):
        self.beginInsertColumns(QtCore.QModelIndex(), column, column + columns - 1)
        self.endInsertColumns()

        return True

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole) -> None:
        if index.isValid():
            if (
                role == QtCore.Qt.ItemDataRole.DisplayRole
                or role == QtCore.Qt.ItemDataRole.EditRole
            ):
                value = self._data.iloc[index.row(), index.column()]

                return str(value) if index.column() != self._data.columns.get_loc('Timestamp') else time.strftime("%H:%M:%S %d/%m/%Y", time.localtime(value))

    def setData(self, index, value, role=None) -> None:
        if not index.isValid():
            return False
  
        self._data.iloc[index.row(), index.column()] = value
        self.dataChanged.emit(index, index)

        return True

    def headerData(self, col, orientation, role) -> None:
        if (
            orientation == QtCore.Qt.Orientation.Horizontal
            and role == QtCore.Qt.ItemDataRole.DisplayRole
        ):
            return self._data.columns[col]

    def flags(self, index) -> None:

        if index.column() == self._data.columns.get_loc('Comment'):
            return (
                QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsEditable
            )

        else:
            return (
                QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
            )