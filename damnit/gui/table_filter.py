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
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    THUMBNAIL = "thumbnail"


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
        # Determine if column is numeric
        is_numeric = True
        values = set()
        decos = set()

        for row in range(model.sourceModel().rowCount()):
            item = model.sourceModel().index(row, column)
            value = item.data(Qt.UserRole)
            values.add(value)
            # Only check type for non-None, non-NaN values
            if value is not None and not (isinstance(value, float) and isnan(value)):
                if not isinstance(value, (int, float)):
                    is_numeric = False
                    # break

            if thumb := item.data(Qt.DecorationRole):
                decos.add(type(thumb))

        # Create appropriate filter widget
        if values == {None} and len(decos) == 1 and decos.pop() is QPixmap:
            filter_widget = ThumbnailFilterWidget(column)
        elif is_numeric:
            filter_widget = NumericFilterWidget(column, values)
        else:
            filter_widget = CategoricalFilterWidget(column, values)
        return filter_widget

    @qthrottled(timeout=20, leading=False)
    def _on_filter_changed(self, filter: Filter):
        """Apply the new filter to the model."""
        self.model.set_filter(self.column, filter)


class NumericFilterWidget(QWidget):
    """Widget for configuring numeric filters with both range and value selection."""

    filterChanged = QtCore.pyqtSignal(NumericFilter)

    def __init__(self, column: int, values: Set[Any], parent=None):
        super().__init__(parent)
        self.column = column
        self.all_values = values

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Range inputs
        range_group = QGroupBox("Value Range")
        range_layout = QVBoxLayout()

        self.min_input = QLineEdit()
        self.max_input = QLineEdit()
        self.min_input.setPlaceholderText("Min")
        self.max_input.setPlaceholderText("Max")

        # Create and set validator for numerical input
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.min_input.setValidator(validator)
        self.max_input.setValidator(validator)

        range_layout.addWidget(self.min_input)
        range_layout.addWidget(self.max_input)
        range_group.setLayout(range_layout)

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

        # Main layout
        layout.addWidget(range_group)
        layout.addWidget(list_group)
        layout.addWidget(self.include_nan)
        self.setLayout(layout)

        # Populate the list initially
        self._populate_list()

        # Connect signals
        self.min_input.editingFinished.connect(self._on_range_changed)
        self.max_input.editingFinished.connect(self._on_range_changed)
        self.list_widget.itemChanged.connect(self._on_selection_changed)
        self.all_button.clicked.connect(lambda: self._set_all_checked(True))
        self.none_button.clicked.connect(lambda: self._set_all_checked(False))
        self.include_nan.toggled.connect(self._emit_filter)

    def _populate_list(self):
        """Populate the list widget with values that match the current range."""
        self.list_widget.clear()

        min_val = float(self.min_input.text()) if self.min_input.text() else -inf
        max_val = float(self.max_input.text()) if self.max_input.text() else inf

        # Add all values to list, but only check those in range
        for value in natsorted(self.all_values):
            if value is not None and not (isinstance(value, float) and isnan(value)):
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
        self.min_input.clear()
        self.max_input.clear()
        for idx in range(self.list_widget.count()):
            self.list_widget.item(idx).setCheckState(
                Qt.Checked if checked else Qt.Unchecked
            )
        self._emit_filter()

    def _on_selection_changed(self, item: QListWidgetItem = None):
        """Handle changes in value selection."""
        self._emit_filter()

    def _emit_filter(self):
        """Create and emit a new NumericFilter based on current widget state."""
        min_val = self.min_input.text()
        max_val = self.max_input.text()
        include_nan = self.include_nan.isChecked()

        # Get range values
        min_val = float(min_val) if min_val else -inf
        max_val = float(max_val) if max_val else inf

        # Get selected values
        selected_values = {
            self.list_widget.item(idx).data(Qt.UserRole)
            for idx in range(self.list_widget.count())
            if self.list_widget.item(idx).checkState() == Qt.Checked
        }

        self.filterChanged.emit(
            NumericFilter(
                self.column,
                min_val=min_val,
                max_val=max_val,
                selected_values=selected_values,
                include_nan=include_nan
            )
        )

    def set_filter(self, filter: Optional[NumericFilter]):
        """Update widget state from an existing filter."""
        if filter is None:
            self.min_input.clear()
            self.max_input.clear()
            self.include_nan.setChecked(True)
            self._populate_list()
            return

        if filter.min_val != -inf:
            self.min_input.setText(str(filter.min_val))
        if filter.max_val != inf:
            self.max_input.setText(str(filter.max_val))

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


class ThumbnailFilterWidget(QWidget):
    """Widget for configuring thumbnail filters."""

    filterChanged = QtCore.pyqtSignal(ThumbnailFilter)

    def __init__(self, column: int, parent=None):
        super().__init__(parent)
        self.column = column

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

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
        layout.addWidget(self.with_thumbnail)
        layout.addWidget(self.without_thumbnail)
        self.setLayout(layout)

    def _emit_filter(self):
        filter = ThumbnailFilter(
            self.column,
            show_with_thumbnail=self.with_thumbnail.isChecked(),
            show_without_thumbnail=self.without_thumbnail.isChecked(),
        )
        self.filterChanged.emit(filter)

    def set_filter(self, filter: Optional[ThumbnailFilter]):
        if filter is None:
            self.with_thumbnail.setChecked(True)
            self.without_thumbnail.setChecked(True)
        else:
            self.with_thumbnail.setChecked(filter.show_with_thumbnail)
            self.without_thumbnail.setChecked(filter.show_without_thumbnail)


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
