from enum import Enum
from math import inf, isnan
from typing import Any, Dict, Optional, Set

import numpy as np
from fonticon_fa6 import FA6S
from natsort import natsorted
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QHBoxLayout,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
    QGroupBox,
)
from superqt import QSearchableListWidget as SuperQListWidget
from superqt.fonticon import icon
from superqt.utils import qthrottled

from .widgets import ValueRangeWidget


class FilterType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    THUMBNAIL = "thumbnail"


class QSearchableListWidget(SuperQListWidget):
    def update_visible(self, text):
        # Change the original implementation from using a set instead of a list
        # for more efficient lookup on large tables
        items_text = {x.text() for x in self.list_widget.findItems(text, Qt.MatchContains)}

        for index in range(self.list_widget.count()):
            item = self.item(index)
            item.setHidden(item.text() not in items_text)


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
        include_nan: bool = True,
        selected_values: Optional[Set[Any]] = None,
    ):
        super().__init__(column, FilterType.NUMERIC)
        self.min_val = min_val
        self.max_val = max_val
        self.include_nan = include_nan
        self.selected_values = selected_values if selected_values else set()

    def accepts(self, value: Any) -> bool:
        if value is None or (isinstance(value, float) and isnan(value)):
            return self.include_nan
        try:
            value = float(value)
            if not (self.min_val <= value <= self.max_val):
                return False
            return value in self.selected_values
        except (TypeError, ValueError):
            return False


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


class ThumbnailFilter(Filter):
    """Filter for columns containing thumbnails, filtering based on presence/absence of thumbnails."""

    def __init__(self, column: int, show_with_thumbnail: bool = True, show_without_thumbnail: bool = True):
        super().__init__(column, FilterType.THUMBNAIL)
        self.show_with_thumbnail = show_with_thumbnail
        self.show_without_thumbnail = show_without_thumbnail

    def accepts(self, value: Any) -> bool:
        has_thumbnail = value is QPixmap
        return (has_thumbnail and self.show_with_thumbnail) or (not has_thumbnail and self.show_without_thumbnail)


class FilterStatus(QPushButton):
    def __init__(self, table_view, parent=None):
        super().__init__(parent)
        self._actions = []
        self.table_view = table_view
        self._update_model()
        self.menu = QMenu(self)
        self.setMenu(self.menu)
        self._update_text()

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

            item = self.sourceModel().index(source_row, col, source_parent)
            if isinstance(filter, ThumbnailFilter):
                data = type(item.data(Qt.DecorationRole))
            else:
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
        self.filter_widget.filterCleared.connect(self._on_filter_cleared)

        # Add widget to menu
        action = QWidgetAction(self)
        action.setDefaultWidget(self.filter_widget)
        self.addAction(action)

        # Set initial state if there's an existing filter
        existing_filter = model.filters.get(column)
        if existing_filter is not None:
            self.filter_widget.set_filter(existing_filter)

    def _create_filter_widget(self, column: int, model: FilterProxy):
        # Determine if column is numeric
        is_numeric = True
        values = []
        decos = set()

        for row in range(model.sourceModel().rowCount()):
            item = model.sourceModel().index(row, column)

            if thumb := item.data(Qt.DecorationRole):
                # QColor decoration is used for errors (no value)
                if not isinstance(thumb, QColor):
                    decos.add(type(thumb))

            if (value := item.data(Qt.UserRole)) is None:
                continue
            if isinstance(value, float) and isnan(value):
                continue

            if not isinstance(value, (int, float)):
                is_numeric = False

            values.append(value)

        unique_values = set(values)

        # Create appropriate filter widget
        if len(unique_values) == 0 and (len(decos) == 1) and (decos.pop() is QPixmap):
            filter_widget = ThumbnailFilterWidget(column)
        elif is_numeric:
            filter_widget = NumericFilterWidget(column, values)
        else:
            filter_widget = CategoricalFilterWidget(column, unique_values)
        return filter_widget

    @qthrottled(timeout=20, leading=False)
    def _on_filter_changed(self, filter: Filter):
        """Apply the new filter to the model."""
        self.model.set_filter(self.column, filter)

    @qthrottled(timeout=20, leading=False)
    def _on_filter_cleared(self):
        """Clear filter from the model."""
        self.model.set_filter(self.column, None)


class BaseFilterWidget(QWidget):
    """Base class for all filter widgets with common functionality."""

    filterCleared = QtCore.pyqtSignal()

    def __init__(self, column: int, parent=None):
        super().__init__(parent)
        self.column = column

        # Create main layout with consistent margins
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.main_layout)

    def hideEvent(self, event):
        """Emit current filter state when widget is hidden."""
        self._emit_filter()
        super().hideEvent(event)

    def _emit_filter(self):
        """Create and emit a new filter based on current widget state."""
        raise NotImplementedError

    def set_filter(self, filter: Optional[Filter]):
        """Update widget state from an existing filter."""
        raise NotImplementedError


class NumericFilterWidget(BaseFilterWidget):
    """Widget for configuring numeric filters with both range and value selection."""

    filterChanged = QtCore.pyqtSignal(NumericFilter)

    def __init__(self, column: int, values: list[Any], parent=None):
        super().__init__(column, parent)

        self.all_values = np.asarray(values, dtype=np.float64)
        self.all_values.sort()
        self.unique_values = np.unique(self.all_values)
        vmin, vmax = self.unique_values[[0, -1]]

        # Range inputs
        range_group = QGroupBox("Value Range")
        range_layout = QVBoxLayout()

        self.range_widget = ValueRangeWidget(self.all_values, vmin, vmax)

        range_layout.addWidget(self.range_widget)
        range_group.setLayout(range_layout)
        self.main_layout.addWidget(range_group)

        # Value selection list
        list_group = QGroupBox("Select Values")
        list_layout = QVBoxLayout()

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

        list_layout.addLayout(button_layout)
        list_layout.addWidget(self.list_widget)
        list_group.setLayout(list_layout)

        self.main_layout.addWidget(list_group)
        self.main_layout.addWidget(self.include_nan)

        # Populate the list initially
        self._populate_list()

        # Connect signals
        self.range_widget.rangeChanged.connect(self._on_range_changed)
        self.list_widget.itemChanged.connect(self._on_selection_changed)
        self.all_button.clicked.connect(lambda: self._set_all_checked(True))
        self.none_button.clicked.connect(lambda: self._set_all_checked(False))
        self.include_nan.toggled.connect(self._emit_filter)

    def _populate_list(self):
        """Populate the list widget with values that match the current range."""
        self.list_widget.clear()

        min_val, max_val = self.range_widget.sel

        # Add all values to list, but only check those in range
        for value in self.unique_values:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, value)
            item.setData(Qt.DisplayRole, str(value))
            # Check if value is in range
            try:
                float_val = float(value)
                item.setCheckState(Qt.Checked if min_val <= float_val <= max_val else Qt.Unchecked)
            except (TypeError, ValueError):
                item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

    def _on_range_changed(self):
        """Handle changes in the range inputs."""
        self._populate_list()  # Update list to match range
        self._emit_filter()

    def _set_all_checked(self, checked: bool):
        """Set all items to checked or unchecked state."""
        self.range_widget.update_values(-inf, inf)

        for idx in range(self.list_widget.count()):
            self.list_widget.item(idx).setCheckState(
                Qt.Checked if checked else Qt.Unchecked
            )
        self._emit_filter()

    def _on_selection_changed(self, item: QListWidgetItem = None):
        """Handle changes in value selection."""
        self._emit_filter()

    @qthrottled(timeout=100, leading=False)
    def _emit_filter(self):
        """Create and emit a new NumericFilter based on current widget state."""
        min_val, max_val = self.range_widget.sel
        include_nan = self.include_nan.isChecked()

        # Get selected values
        selected, count = set(), 0
        for idx in range(self.list_widget.count()):
            item = self.list_widget.item(idx)
            if item.checkState() == Qt.Checked:
                selected.add(item.data(Qt.UserRole))
                count += 1

        if (
            min_val == -inf
            and max_val == inf
            and include_nan
            and count == self.list_widget.count()
        ):
            self.filterCleared.emit()
        else:
            self.filterChanged.emit(
                NumericFilter(
                    self.column,
                    min_val=min_val,
                    max_val=max_val,
                    selected_values=selected,
                    include_nan=include_nan
                )
            )

    def set_filter(self, filter: Optional[NumericFilter]):
        """Update widget state from an existing filter."""
        if filter is None:
            return

        self.include_nan.setChecked(filter.include_nan)

        # Update list and selection
        self._populate_list()
        if hasattr(filter, 'selected_values'):
            for idx in range(self.list_widget.count()):
                item = self.list_widget.item(idx)
                item.setCheckState(
                    Qt.Checked
                    if item.data(Qt.UserRole) in filter.selected_values
                    else Qt.Unchecked
                )

        self.range_widget.update_values(filter.min_val, filter.max_val)


class ThumbnailFilterWidget(BaseFilterWidget):
    """Widget for configuring thumbnail filters."""

    filterChanged = QtCore.pyqtSignal(ThumbnailFilter)

    def __init__(self, column: int, parent=None):
        super().__init__(column, parent)

        # Checkboxes for filtering
        self.with_thumbnail = QCheckBox("Show thumbnails")
        self.without_thumbnail = QCheckBox("Show empty")

        # Set initial state
        self.with_thumbnail.setChecked(True)
        self.without_thumbnail.setChecked(True)

        # Connect signals
        self.with_thumbnail.toggled.connect(self._emit_filter)
        self.without_thumbnail.toggled.connect(self._emit_filter)

        # Layout
        self.main_layout.addWidget(self.with_thumbnail)
        self.main_layout.addWidget(self.without_thumbnail)

    def _emit_filter(self):
        show_thumb = self.with_thumbnail.isChecked()
        show_no_thumb = self.without_thumbnail.isChecked()

        if show_thumb and show_no_thumb:
            self.filterCleared.emit()
        else:
            self.filterChanged.emit(
                ThumbnailFilter(
                    self.column,
                    show_with_thumbnail=show_thumb,
                    show_without_thumbnail=show_no_thumb,
                )
            )

    def set_filter(self, filter: Optional[ThumbnailFilter]):
        if filter is None:
            self.with_thumbnail.setChecked(True)
            self.without_thumbnail.setChecked(True)
        else:
            self.with_thumbnail.setChecked(filter.show_with_thumbnail)
            self.without_thumbnail.setChecked(filter.show_without_thumbnail)


class CategoricalFilterWidget(BaseFilterWidget):
    """Widget for configuring categorical filters with a searchable list of values."""

    filterChanged = QtCore.pyqtSignal(CategoricalFilter)

    def __init__(self, column: int, values: Set[Any], parent=None):
        super().__init__(column, parent)

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
        self.main_layout.addLayout(button_layout)
        self.main_layout.addWidget(self.list_widget)
        self.main_layout.addWidget(self.include_nan)

        # Add values to list (excluding nan/empty values)
        for value in natsorted(values, key=str):
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

        self._emit_filter()

    def _on_selection_changed(self, item: QListWidgetItem = None):
        self._emit_filter()

    @qthrottled(timeout=100)
    def _emit_filter(self):
        # Get selected values
        selected, count = set(), 0
        for idx in range(self.list_widget.count()):
            item = self.list_widget.item(idx)
            if item.checkState() == Qt.Checked:
                selected.add(item.data(Qt.UserRole))
                count += 1

        if count == self.list_widget.count() and self.include_nan.isChecked():
            # no filter
            self.filterCleared.emit()
        else:
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
