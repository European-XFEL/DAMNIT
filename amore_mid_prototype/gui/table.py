import time
from functools import lru_cache

import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt


ROW_HEIGHT = 30
THUMBNAIL_SIZE = 35


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
        self.verticalHeader().setMinimumSectionSize(ROW_HEIGHT)

        # menu
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        # these are fixed
        self.static_columns = [
            "Timestamp",
            "Comment",
        ]

        # these are hidden
        self.hidden_column = []

    def setModel(self, model):
        """
        Overload of setModel() to make sure that we restyle the comment rows
        when the model is updated.
        """
        super().setModel(model)
        self.model().rowsInserted.connect(self.style_comment_rows)
        self.model().rowsInserted.connect(self.resize_new_rows)
        self.resizeRowsToContents()

    def show_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        menu.addAction("Change color?")
        menu.exec_(self.mapToGlobal(pos))

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

        for i in range(len(columns)):
            item = QtWidgets.QListWidgetItem(columns[i])
            item.setCheckState(Qt.Checked if status[i] else Qt.Unchecked)

            self._columns_widget.addItem(item)

    def set_columns_visibility(self, columns, statuses):
        # group = QtWidgets.QGroupBox("Column settings")

        layout = QtWidgets.QVBoxLayout()

        # Remove the static columns
        columns, statuses = map(
            list,
            zip(
                *[
                    x
                    for x in zip(columns, statuses)
                    if (
                        x[0] not in self.static_columns
                    )  # and not x[0].startswith("_"))
                ]
            ),
        )

        # default hidden columns to unchecked
        for i, ci in enumerate(columns):
            if ci.startswith("_"):
                statuses[i] = False

        self._columns_widget = QtWidgets.QListWidget()
        self._columns_widget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self._columns_widget.itemChanged.connect(self.item_changed)
        self._columns_widget.model().rowsMoved.connect(self.item_moved)

        self._columns_widget.setStyleSheet(
            "QListWidget {padding: 0px;} QListWidget::item { margin: 5px; }"
        )

        self.set_item_columns_visibility(columns, statuses)

        layout.addWidget(QtWidgets.QLabel("Drag these columns to reorder them:"))
        layout.addWidget(self._columns_widget)
        # group.setLayout(layout)

        return layout

    def style_comment_rows(self, *_):
        self.clearSpans()
        data = self.model()._data

        comment_col = data.columns.get_loc("Comment")

        for row in data["_comment_id"].dropna().index:
            self.setSpan(row, comment_col, 1, 100)

    def resize_new_rows(self, parent, first, last):
        for row in range(first, last + 1):
            self.resizeRowToContents(row)


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

    @lru_cache(maxsize=500)
    def generateThumbnail(self, index) -> QtGui.QPixmap:
        """
        Helper function to generate a thumbnail for a 2D array.

        Note that we require an index for self._data as the argument instead of
        an ndarray. That's because this function is quite expensive and called
        very often by self.data(), so for performance we try to cache the
        thumbnails with @lru_cache. Unfortunately ndarrays are not hashable so
        we have to take an index instead.
        """
        image = self._data.iloc[index.row(), index.column()]

        fig = Figure(figsize=(1, 1))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        vmin = np.nanquantile(image, 0.01, interpolation="nearest")
        vmax = np.nanquantile(image, 0.99, interpolation="nearest")
        ax.imshow(image, vmin=vmin, vmax=vmax, extent=(0, 1, 1, 0))
        ax.axis("tight")
        ax.axis("off")
        ax.margins(0, 0)
        canvas.draw()

        width, height = int(fig.figbbox.width), int(fig.figbbox.height)
        image = QtGui.QImage(
            canvas.buffer_rgba(), width, height, QtGui.QImage.Format_ARGB32
        )
        return QtGui.QPixmap(image).scaled(
            QtCore.QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE), Qt.KeepAspectRatio
        )

    @lru_cache(maxsize=1000)
    def variable_is_constant(self, index):
        """
        Check if the variable at the given index is constant throughout the run.
        """
        is_constant = True
        run = self._data.iloc[index.row(), self._data.columns.get_loc("Run")]
        proposal = self._data.iloc[index.row(), self._data.columns.get_loc("Proposal")]
        quantity = self._main_window.ds_name(self._data.columns[index.column()])

        try:
            file_name, run_file = self._main_window.get_run_file(proposal, run, write_to_log=False)
        except:
            return is_constant

        if run_file is not None:
            if quantity in run_file and "trainId" in run_file[quantity]:
                ds = run_file[quantity]["data"]
                # If it's an array
                if len(ds.shape) == 1:
                    data = ds[:]
                    if not np.all(np.isclose(data, data[0])):
                        is_constant = False

            run_file.close()
        return is_constant

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return

        value = self._data.iloc[index.row(), index.column()]

        if role == Qt.FontRole:
            # If the variable is not constant, make it bold
            if not self.variable_is_constant(index):
                font = QtGui.QFont()
                font.setBold(True)
                return font

        elif role == Qt.DecorationRole:
            if isinstance(value, np.ndarray):
                return self.generateThumbnail(index)
        elif role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if isinstance(value, np.ndarray):
                # The image preview for this is taken care of by the DecorationRole
                return None

            elif pd.isna(value) or index.column() == self._data.columns.get_loc("Use"):
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

        elif (
            role == Qt.ItemDataRole.CheckStateRole
            and index.column() == self._data.columns.get_loc("Use")
            and not self.isCommentRow(index.row())
        ):
            if self._data["Use"].iloc[index.row()]:
                return QtCore.Qt.Checked
            else:
                return QtCore.Qt.Unchecked

        elif role == Qt.ToolTipRole:
            if index.column() == self._data.columns.get_loc("Comment"):
                return self.data(index)

        elif role == QtCore.Qt.BackgroundRole:
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
            new_state = not self._data["Use"].iloc[index.row()]
            self._data["Use"].values[index.row()] = new_state
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
        elif index.column() == self._data.columns.get_loc("Use"):
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
            QtWidgets.QMessageBox.warning(
                self._main_window, "Sorting error", "This column cannot be sorted"
            )
        else:
            self.is_sorted_by = is_sorted_by
            self.is_sorted_order = order
            self._data.reset_index(inplace=True, drop=True)
        finally:
            self.layoutChanged.emit()
