from enum import Enum
from math import inf, isnan, nan
from typing import Any, Dict, Optional, Set

from fonticon_fa6 import FA6S
from natsort import natsorted
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from superqt import QSearchableListWidget
from superqt.fonticon import icon


class FilterType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class Filter:
    """Base class for all filters."""

    def __init__(self, column: int, filter_type: FilterType):
        self.column = column
        self.type = filter_type
        self.enabled = True

    def accepts(self, value: Any) -> bool:
        """Return True if the value passes the filter."""
        raise NotImplementedError


class NumericFilter(Filter):
    """Filter for numeric values with optional NaN handling."""

    def __init__(
        self,
        column: int,
        min_val: float = -inf,
        max_val: float = inf,
        include_nan: bool = False,
    ):
        super().__init__(column, FilterType.NUMERIC)
        self.min_val = min_val
        self.max_val = max_val
        self.include_nan = include_nan

    def accepts(self, value: Any) -> bool:
        if value in (None, nan) or (isinstance(value, float) and isnan(value)):
            return self.include_nan
        try:
            value = float(value)
            return self.min_val <= value <= self.max_val
        except (TypeError, ValueError):
            return False


class CategoricalFilter(Filter):
    """Filter for categorical values based on a set of selected values."""

    def __init__(self, column: int, selected_values: Optional[Set[Any]] = None):
        super().__init__(column, FilterType.CATEGORICAL)
        self.selected_values = set(selected_values) if selected_values else set()

    def accepts(self, value: Any) -> bool:
        # If no values are selected, reject all values
        if not self.selected_values:
            return False
        return value in self.selected_values


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
        if self.parent()._is_all_item(Qt.Checked) or self.parent()._is_all_item(
            Qt.Unchecked
        ):
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
        self.all_button = QPushButton("All")
        self.none_button = QPushButton("None")
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
            action.triggered.connect(
                lambda x, column=column: self.model.set_filter(column, None)
            )
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
    """Proxy model that applies filters to rows."""

    filterChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filters: Dict[int, Filter] = {}

    def set_filter(self, column: int, filter: Optional[Filter] = None):
        """Set or remove a filter for a column."""
        if filter is not None:
            self.filters[column] = filter
        elif column in self.filters:
            del self.filters[column]
        self.invalidateFilter()
        self.filterChanged.emit()

    def clear_filters(self):
        """Remove all filters."""
        self.filters.clear()
        self.invalidateFilter()
        if hasattr(self.parent(), "damnit_model"):
            cid = self.parent().damnit_model.find_column("Timestamp", by_title=True)
            self.parent().sortByColumn(cid, Qt.AscendingOrder)
        self.filterChanged.emit()

    def filterAcceptsRow(
        self, source_row: int, source_parent: QtCore.QModelIndex
    ) -> bool:
        """Return True if the row passes all active filters."""
        for col, filter in self.filters.items():
            if not filter.enabled:
                continue
            data = (
                self.sourceModel()
                .index(source_row, col, source_parent)
                .data(Qt.UserRole)
            )
            if not filter.accepts(data):
                return False
        return True


class FilterMenu(QMenu):
    """Menu for configuring filters on table columns."""

    def __init__(self, column: int, model: FilterProxy, parent=None):
        super().__init__(parent)
        self.column = column
        self.model = model

        # Determine if column is numeric
        self.is_numeric = True
        values = set()
        for row in range(model.sourceModel().rowCount()):
            value = model.sourceModel().index(row, column).data(Qt.UserRole)
            values.add(value)
            # Only check type for non-None, non-NaN values
            if value is not None and not (isinstance(value, float) and isnan(value)):
                if not isinstance(value, (int, float)):
                    self.is_numeric = False
                    # break

        # Create appropriate filter widget
        if self.is_numeric:
            self.filter_widget = NumericFilterWidget(column)
        else:
            self.filter_widget = CategoricalFilterWidget(column, values)

        # Connect filter widget to model
        self.filter_widget.filterChanged.connect(self._on_filter_changed)

        # Add widget to menu
        action = QWidgetAction(self)
        action.setDefaultWidget(self.filter_widget)
        self.addAction(action)

        # Set initial state if there's an existing filter
        existing_filter = model.filters.get(column)
        if existing_filter is not None:
            self.filter_widget.set_filter(existing_filter)

    def _on_filter_changed(self, filter: Filter):
        """Apply the new filter to the model."""
        self.model.set_filter(self.column, filter)


class NumericFilterWidget(QWidget):
    """Widget for configuring numeric range filters."""

    filterChanged = QtCore.pyqtSignal(NumericFilter)

    def __init__(self, column: int, parent=None):
        super().__init__(parent)
        self.column = column

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Range inputs
        self.min_input = QLineEdit()
        self.max_input = QLineEdit()
        self.min_input.setPlaceholderText("Min")
        self.max_input.setPlaceholderText("Max")

        # Create and set validator for numerical input
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.min_input.setValidator(validator)
        self.max_input.setValidator(validator)

        # NaN handling
        self.include_nan = QCheckBox("Include NaN/empty values")

        # Connect signals
        self.min_input.editingFinished.connect(self._on_value_changed)
        self.max_input.editingFinished.connect(self._on_value_changed)
        self.include_nan.toggled.connect(self._on_value_changed)

        # Layout
        layout.addWidget(self.min_input)
        layout.addWidget(self.max_input)
        layout.addWidget(self.include_nan)
        self.setLayout(layout)

    def _on_value_changed(self):
        """Create and emit a new NumericFilter based on current widget state."""
        min_val = self.min_input.text()
        max_val = self.max_input.text()
        include_nan = self.include_nan.isChecked()

        min_val = float(min_val) if min_val else -inf
        max_val = float(max_val) if max_val else inf

        self.filterChanged.emit(
            NumericFilter(self.column, min_val, max_val, include_nan)
        )

    def set_filter(self, filter: Optional[NumericFilter]):
        """Update widget state from an existing filter."""
        if filter is None:
            self.min_input.clear()
            self.max_input.clear()
            self.include_nan.setChecked(False)
            return

        if filter.min_val != -inf:
            self.min_input.setText(str(filter.min_val))
        if filter.max_val != inf:
            self.max_input.setText(str(filter.max_val))
        self.include_nan.setChecked(filter.include_nan)


class CategoricalFilterWidget(QWidget):
    """Widget for configuring categorical filters with a searchable list of values."""

    filterChanged = QtCore.pyqtSignal(CategoricalFilter)

    def __init__(self, column: int, values: Set[Any], parent=None):
        super().__init__(parent)
        self.column = column

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Searchable list of values
        self.list_widget = QSearchableListWidget()
        self.list_widget.filter_widget.setPlaceholderText("Search values...")
        self.list_widget.layout().setContentsMargins(0, 0, 0, 0)

        # Add values to list
        for value in natsorted(values):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, value)
            item.setData(
                Qt.DisplayRole, str(value) if value is not None else "NaN/empty"
            )
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)

        # All/None buttons
        button_layout = QHBoxLayout()
        self.all_button = QPushButton("Select All")
        self.none_button = QPushButton("Select None")
        self.all_button.clicked.connect(lambda: self._set_all_checked(True))
        self.none_button.clicked.connect(lambda: self._set_all_checked(False))
        button_layout.addWidget(self.all_button)
        button_layout.addWidget(self.none_button)

        # Connect signals
        self.list_widget.itemChanged.connect(self._on_selection_changed)

        # Layout
        layout.addLayout(button_layout)
        layout.addWidget(self.list_widget)
        self.setLayout(layout)

    def _set_all_checked(self, checked: bool):
        """Set all items to checked or unchecked state."""
        for idx in range(self.list_widget.count()):
            self.list_widget.item(idx).setCheckState(
                Qt.Checked if checked else Qt.Unchecked
            )
        # Emit signal after setting all items
        selected = (
            set()
            if not checked
            else {
                self.list_widget.item(idx).data(Qt.UserRole)
                for idx in range(self.list_widget.count())
            }
        )
        self.filterChanged.emit(CategoricalFilter(self.column, selected))

    def _on_selection_changed(self, item: QListWidgetItem):
        """Create and emit a new CategoricalFilter based on current selection."""
        selected = {
            self.list_widget.item(idx).data(Qt.UserRole)
            for idx in range(self.list_widget.count())
            if self.list_widget.item(idx).checkState() == Qt.Checked
        }
        self.filterChanged.emit(CategoricalFilter(self.column, selected))

    def set_filter(self, filter: Optional[CategoricalFilter]):
        """Update widget state from an existing filter."""
        if filter is None:
            self._set_all_checked(True)
            return

        selected_values = filter.selected_values
        for idx in range(self.list_widget.count()):
            item = self.list_widget.item(idx)
            item.setCheckState(
                Qt.Checked
                if item.data(Qt.UserRole) in selected_values
                else Qt.Unchecked
            )
