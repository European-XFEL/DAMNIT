from math import nan, inf, isnan

from PyQt5 import QtCore
from PyQt5.QtWidgets import QHeaderView, QMenu, QAction, QListWidgetItem, QWidgetAction, QPushButton, QHBoxLayout, QWidget, QVBoxLayout, QLineEdit, QCheckBox
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QColor, QDoubleValidator
from superqt import QSearchableListWidget
from fonticon_fa6 import FA6S
from superqt.fonticon import icon
from superqt.utils import qdebounced
from natsort import natsorted


class NumericRangeInput(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        self.min_input = QLineEdit()
        self.max_input = QLineEdit()
        self.min_input.setPlaceholderText("Min")
        self.max_input.setPlaceholderText("Max")
        self.include_nan = QCheckBox("include nan")

        # Create and set validator for numerical input
        # Allow both positive and negative numbers with decimals
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.min_input.setValidator(validator)
        self.max_input.setValidator(validator)

        self.parent().selectionChanged.connect(self._check_selection)

        layout.addWidget(self.min_input)
        layout.addWidget(self.max_input)
        layout.addWidget(self.include_nan)
        self.setLayout(layout)

    def _check_selection(self):
        if self.parent()._is_all_item(Qt.Checked) or self.parent()._is_all_item(Qt.Unchecked):
            self.min_input.setText(None)
            self.max_input.setText(None)

    def get_values(self):
        """Get the current values from the inputs"""
        min_val = self.min_input.text()
        max_val = self.max_input.text()
        include_nan = self.include_nan.isChecked()

        min_val = float(min_val) if min_val else -inf
        max_val = float(max_val) if max_val else inf

        return min_val, max_val, include_nan


class ToggleButtonsWidget(QWidget):
    toggled = QtCore.pyqtSignal(bool)  # all: True, none: False

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        self.all_button = QPushButton('All')
        self.none_button = QPushButton('None')
        # Set buttons to be checkable (toggleable)
        self.all_button.setCheckable(True)
        self.none_button.setCheckable(True)
        self._check_selection()

        self.all_button.clicked.connect(self.on_all_clicked)
        self.none_button.clicked.connect(self.on_none_clicked)
        self.parent().selectionChanged.connect(self._check_selection)

        layout.addWidget(self.all_button)
        layout.addWidget(self.none_button)
        self.setLayout(layout)

    def _check_selection(self):
        if self.parent()._is_all_item(Qt.Checked):
            self.all_button.setChecked(True)
        elif self.parent()._is_all_item(Qt.Unchecked):
            self.none_button.setChecked(True)
        else:
            self.all_button.setChecked(False)
            self.none_button.setChecked(False)

    def on_all_clicked(self):
        if self.all_button.isChecked():
            self.none_button.setChecked(False)
            self.toggled.emit(True)
        else:
            # don't change state when clicking on checked button
            self.all_button.setChecked(True)

    def on_none_clicked(self):
        if self.none_button.isChecked():
            self.all_button.setChecked(False)
            self.toggled.emit(False)
        else:
            # don't change state when clicking on checked button
            self.none_button.setChecked(True)


class FilterStatus(QPushButton):
    def __init__(self, table_view, parent=None):
        super().__init__(parent)
        self._actions = []
        self.table_view = table_view
        self._update_model()
        self.menu = QMenu(self)
        self.setMenu(self.menu)
        self._update_text()

        if self.model is not None:
            self.model.filterChanged.connect(self._update_text)
        self.menu.aboutToShow.connect(self._populate_menu)
        self.table_view.model_updated.connect(self._update_model)

    def _update_model(self):
        self.model = None
        model = self.table_view.model()
        if isinstance(model, FilterProxy):
            self.model = model
            self.model.filterChanged.connect(self._update_text)

        self._update_text()

    def _populate_menu(self):
        self.menu.clear()
        self._actions.clear()

        if self.model is None or len(self.model.filters) == 0:
            action = QAction("No active filter")
            self._actions.append(action)
            self.menu.addAction(action)
            return

        clear_all = QAction("Clear All Filters", self)
        clear_all.triggered.connect(lambda x: self.model.clear_filters())
        self._actions.append(clear_all)
        self.menu.addAction(clear_all)
        self.menu.addSeparator()

        for column in self.model.filters:
            title = self.model.sourceModel().column_title(column)
            action = QAction(icon(FA6S.trash), f"Clear filter on {title}")
            action.triggered.connect(lambda x, column=column: self.model.set_filter(column, None))
            self._actions.append(action)
            self.menu.addAction(action)

    def _clear_menu(self):
        self.menu.clear()
        self._actions.clear()

    def _update_text(self):
        if self.model is None:
            self.setText("Filters (0)")
        else:
            self.setText(f"Filters ({len(self.model.filters)})")


class FilterProxy(QtCore.QSortFilterProxyModel):
    filterChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filters = {}

    def set_filter(self, column, expression=None):
        if expression is not None:
            self.filters[column] = expression
        elif self.filters.get(column) is not None:
            del self.filters[column]
        self.invalidateFilter()
        self.filterChanged.emit()

    def clear_filters(self):
        self.filters = {}
        self.invalidateFilter()
        cid = self.parent().damnit_model.find_column("Timestamp", by_title=True)
        self.parent().sortByColumn(cid, Qt.AscendingOrder)
        self.filterChanged.emit()

    def filterAcceptsRow(self, source_row, source_parent):
        for col, expr in self.filters.items():
            data = self.sourceModel().index(source_row, col, source_parent).data(Qt.UserRole)
            # print(source_row, self.parent().damnit_model.column_title(col), col, data, type(data))
            if not expr(data):
                return False
        return True


class FilterMenu(QMenu):
    selectionChanged = QtCore.pyqtSignal()

    def __init__(self, column, model, parent=None):
        super().__init__(parent)
        self.column = column
        self.model = model

        # List of selectable unique values
        self.item_list = QSearchableListWidget()
        self.item_list.filter_widget.setPlaceholderText("Search")
        self.item_list.layout().setContentsMargins(0, 0, 0, 0)
        self.item_list.itemChanged.connect(self.selection_changed)

        self.all_numeric = True
        sel_values = self._unique_values(filtered=True, display=False)
        for disp, data in natsorted(self._unique_values()):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, data)
            item.setData(Qt.DisplayRole, disp)
            item.setCheckState(Qt.Checked if (data in sel_values) else Qt.Unchecked)
            self.item_list.addItem(item)

            if not isinstance(data, (int, float, type(None))):
                self.all_numeric = False

        list_action = QWidgetAction(self)
        list_action.setDefaultWidget(self.item_list)

        # select all / none values
        self.all_none = ToggleButtonsWidget(self)
        self.all_none.toggled.connect(self._set_all_item)
        action_all_none = QWidgetAction(self)
        action_all_none.setDefaultWidget(self.all_none)

        # all values in this column are numbers
        if self.all_numeric:
            self.min_max = NumericRangeInput(self)
            self.min_max.min_input.editingFinished.connect(self._new_value_range)
            self.min_max.max_input.editingFinished.connect(self._new_value_range)
            self.min_max.include_nan.toggled.connect(self._new_value_range)
            action_min_max = QWidgetAction(self)
            action_min_max.setDefaultWidget(self.min_max)
            self.addAction(action_min_max)
            self.addSeparator()

        self.addAction(action_all_none)
        self.addAction(list_action)

    def _unique_values(self, filtered=False, display=True):
        model = self.model if filtered else self.model.sourceModel()

        values = set()
        for row in range(model.rowCount()):
            item = model.index(row, self.column)
            if item is None:
                continue

            if display:
                value = (item.data(Qt.DisplayRole) or '', item.data(Qt.UserRole) or nan)
            else:
                value = item.data(Qt.UserRole) or nan
            values.add(value)

        return values

    def _list_items(self):
        for idx in range(self.item_list.count()):
            yield self.item_list.item(idx)

    def _is_all_item(self, state=Qt.Checked):
        """Return True if all items in self.item_list == state"""
        return all(e.checkState() == state for e in self._list_items())

    @QtCore.pyqtSlot(bool)
    def _set_all_item(self, checked):
        """Set state to all items in self.item_list"""
        for item in self._list_items():
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    @QtCore.pyqtSlot()
    def _new_value_range(self):
        import time
        t0 = time.perf_counter()
        vmin, vmax, inc_nan = self.min_max.get_values()

        item_state_changed = False
        for item in self._list_items():
            data = item.data(Qt.UserRole)
            state = item.checkState()

            if inc_nan and data in (None, nan):
                new_state = Qt.Checked
            elif data and vmin <= data <= vmax:
                new_state = Qt.Checked
            else:
                new_state = Qt.Unchecked
            
            if new_state != state:
                item.setCheckState(new_state)
                item_state_changed = True

        if item_state_changed:
            print(time.perf_counter() - t0)
            self.selection_changed()

    @qdebounced(timeout=20, leading=False)
    @QtCore.pyqtSlot()
    def selection_changed(self):
        print('sss')
        selection = set()
        all_selected = True
        none_selected = True

        for item in self._list_items():
            if item.checkState() == Qt.Checked:
                selection.add(item.data(Qt.UserRole))
                none_selected = False
            else:
                all_selected = False

        # print(selection, [type(s) for s in selection])
        if all_selected:
            self.model.set_filter(self.column, None)
        else:
             self.model.set_filter(self.column, lambda data: data in selection)
        self.selectionChanged.emit()
