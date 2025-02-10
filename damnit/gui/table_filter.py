from enum import Enum
from math import inf, isnan
from typing import Any, Dict, Optional, Set

from fonticon_fa6 import FA6S
from natsort import natsorted
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QPixmap
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
    QGroupBox,
)
from superqt import QSearchableListWidget
from superqt.fonticon import icon
from superqt.utils import qthrottled


class FilterType(Enum):
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


class CategoricalFilter(Filter):
    """Filter for categorical values based on a set of selected values."""

    def __init__(self, column: int, selected_values: Optional[Set[Any]] = None, include_nan: bool = True):
        super().__init__(column, FilterType.CATEGORICAL)
        self.selected_values = set(selected_values) if selected_values else set()
        self.include_nan = include_nan

    def accepts(self, value: Any) -> bool:
        # Handle nan/empty values
        if value is None or (isinstance(value, float) and isnan(value)):
            return self.include_nan
        # If no values are selected, reject all values
        if not self.selected_values:
            return False
        return value in self.selected_values


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

            item = self.sourceModel().index(source_row, col, source_parent)
            data = item.data(Qt.UserRole)
            if not filter.accepts(data):
                return False
        return True


class FilterMenu(QMenu):
    """Menu for configuring filters on table columns."""

    def __init__(self, column: int, model: FilterProxy, parent=None):
        super().__init__(parent)
        self.column = column
        self.model = model

        self.filter_widget = self._create_filter_widget(column, model)

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

    def _create_filter_widget(self, column: int, model: FilterProxy):
        values = []
        for row in range(model.sourceModel().rowCount()):
            item = model.sourceModel().index(row, column)
            value = item.data(Qt.UserRole)
            values.add(value)

        return CategoricalFilterWidget(column, values)

    @qthrottled(timeout=20, leading=False)
    def _on_filter_changed(self, filter: Filter):
        """Apply the new filter to the model."""
        self.model.set_filter(self.column, filter)


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

        # All/None buttons
        button_layout = QHBoxLayout()
        self.all_button = QPushButton("Select All")
        self.none_button = QPushButton("Select None")
        button_layout.addWidget(self.all_button)
        button_layout.addWidget(self.none_button)

        # NaN handling
        self.include_nan = QCheckBox("Include NaN/empty values")
        self.include_nan.setChecked(True)

        # Connect signals
        self.list_widget.itemChanged.connect(self._on_selection_changed)
        self.all_button.clicked.connect(lambda: self._set_all_checked(True))
        self.none_button.clicked.connect(lambda: self._set_all_checked(False))
        self.include_nan.toggled.connect(self._on_selection_changed)

        # Layout
        layout.addLayout(button_layout)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.include_nan)
        self.setLayout(layout)

        # Add values to list (excluding nan/empty values)
        for value in natsorted(values):
            if value is not None and not (isinstance(value, float) and isnan(value)):
                item = QListWidgetItem()
                item.setData(Qt.UserRole, value)
                item.setData(Qt.DisplayRole, str(value))
                item.setCheckState(Qt.Checked)
                self.list_widget.addItem(item)

    def _set_all_checked(self, checked: bool):
        """Set all items to checked or unchecked state."""
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(
                Qt.Checked if checked else Qt.Unchecked
            )
        # Emit signal after setting all items
        selected = set()
        if checked:
            for i in range(self.list_widget.count()):
                selected.add(self.list_widget.item(i).data(Qt.UserRole))

        self.filterChanged.emit(
            CategoricalFilter(
                self.column,
                selected_values=selected,
                include_nan=self.include_nan.isChecked()
            )
        )

    def _on_selection_changed(self, item: QListWidgetItem = None):
        selected = set()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.add(item.data(Qt.UserRole))

        self.filterChanged.emit(
            CategoricalFilter(
                self.column,
                selected_values=selected,
                include_nan=self.include_nan.isChecked()
            )
        )

    def set_filter(self, filter: Optional[CategoricalFilter]):
        if filter is None:
            self._set_all_checked(True)
            self.include_nan.setChecked(True)
            return

        # Set nan checkbox state
        self.include_nan.setChecked(filter.include_nan)

        # Set item check states
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            value = item.data(Qt.UserRole)
            item.setCheckState(
                Qt.Checked if value in filter.selected_values else Qt.Unchecked
            )
