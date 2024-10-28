from PyQt5 import QtCore
from PyQt5.QtWidgets import QHeaderView, QMenu, QAction, QListWidgetItem, QWidgetAction, QPushButton, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QColor
from superqt import QSearchableListWidget
from fonticon_fa6 import FA6S
from superqt.fonticon import icon


class ToggleButtonsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        self.all_button = QPushButton('All')
        self.none_button = QPushButton('None')
        # Set buttons to be checkable (toggleable)
        self.all_button.setCheckable(True)
        self.none_button.setCheckable(True)

        if all(item.checkState() == Qt.Checked for item in parent._list_items()):
            self.all_button.setChecked(True)
        elif all(item.checkState() == Qt.Unchecked for item in parent._list_items()):
            self.on_none_clicked.setChecked(True)

        # Set style sheets for different states
        self.style_sheet = """
            QPushButton {
                padding: 5px;
                border: 2px solid #8f8f91;
                border-radius: 6px;
                background-color: #f0f0f0;
                min-width: 80px;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                border: 2px solid #45a049;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:checked:hover {
                background-color: #45a049;
            }
        """
        self.setStyleSheet(self.style_sheet)

        self.all_button.clicked.connect(self.on_all_clicked)
        self.none_button.clicked.connect(self.on_none_clicked)

        layout.addWidget(self.all_button)
        layout.addWidget(self.none_button)
        self.setLayout(layout)

    def on_all_clicked(self):
        if self.all_button.isChecked():
            self.none_button.setChecked(False)

            for item in self.parent()._list_items():
                item.setCheckState(Qt.Checked)
        else:
            # don't change state when clicking on checked button
            self.all_button.setChecked(True)

    def on_none_clicked(self):
        if self.none_button.isChecked():
            self.all_button.setChecked(False)

            for item in self.parent()._list_items():
                item.setCheckState(Qt.Unchecked)
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
        print('Yay!')

        self.model = None
        model = self.table_view.model()
        if isinstance(model, FilterProxy):
            self.model = model
            self.model.filterChanged.connect(self._update_text)

        self._update_text()

    def _populate_menu(self):
        print('yo >> _populate_menu')
        print(self.model.filters)
        self.menu.clear()
        self._actions.clear()

        if self.model is None or len(self.model.filters) == 0:
            print('??? nothingness')
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
            print('???', column)
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


def check_type_consistency(model, column):
    """
    Check if all items in a specific column have the same data type.
    
    Args:
        model: QStandardItemModel
        column: int, column index to check
    
    Returns:
        tuple: (bool, type or None) - (is_consistent, common_type)
    """
    if model.rowCount() == 0:
        return True, None

    types = set()
    for row in range(model.rowCount()):
        item = model.index(row, column)
        item_data = item.data(Qt.ItemDataRole.UserRole)
        if item and item_data is not None:
            types.add(type(item_data))

    if len(types) == 1:
        return True, types.pop()
    return False, None


def check_type_numeric(model, column):
    """Check if all non-null values in column can be converted to float"""
    for row in range(model.rowCount()):
        item = model.item(row, column)
        item_data = item.data(Qt.ItemDataRole.UserRole)
        if item and item_data:
            try:
                float(item_data)
            except ValueError:
                return False
    return True


class FilterProxy(QtCore.QSortFilterProxyModel):
    filterChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filters = {}

    def set_filter(self, column, expression=None):
        print('set_filter', column, expression)
        if expression is not None:
            self.filters[column] = expression
        elif self.filters.get(column) is not None:
            print('clear filter:', column)
            del self.filters[column]
        self.invalidateFilter()
        self.filterChanged.emit()

    def clear_filters(self):
        self.filters = {}
        self.invalidateFilter()
        self.filterChanged.emit()

    def filterAcceptsRow(self, source_row, source_parent):
        for col, expr in self.filters.items():
            data = self.sourceModel().index(source_row, col, source_parent).data(Qt.UserRole)
            if not expr(data):
                return False
        return True


class SearchMenu(QMenu):
    def __init__(self, column, model, parent=None):
        super().__init__(parent)
        self.column = column
        self.model = model

        self.item_list = QSearchableListWidget()
        self.item_list.filter_widget.setPlaceholderText("Search")
        self.item_list.layout().setContentsMargins(0, 0, 0, 0)
        self.item_list.itemChanged.connect(self.selectionChanged)

        sel_values = self._unique_values(filtered=True, display=False)
        for disp, data in sorted(self._unique_values()):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, data)
            item.setData(Qt.DisplayRole, disp)
            item.setCheckState(Qt.Checked if (data in sel_values) else Qt.Unchecked)
            self.item_list.addItem(item)

        list_action = QWidgetAction(self)
        list_action.setDefaultWidget(self.item_list)

        self.all_none = ToggleButtonsWidget(self)
        self.all_none.layout().setContentsMargins(0, 0, 0, 0)
        action_all_none = QWidgetAction(self)
        action_all_none.setDefaultWidget(self.all_none)

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
                value = (item.data(Qt.DisplayRole) or '', item.data(Qt.UserRole))
            else:
                value = item.data(Qt.UserRole)
            values.add(value)
        return values

    def _list_items(self):
        for idx in range(self.item_list.count()):
            yield self.item_list.item(idx)

    @QtCore.pyqtSlot()
    def selectionChanged(self):
        selection = set()
        all_selected = True

        for item in self._list_items():
            if item.checkState() == Qt.Checked:
                selection.add(item.data(Qt.UserRole))
            else:
                all_selected = False
        print(selection, [type(s) for s in selection])
        if all_selected:
            self.model.set_filter(self.column, None)
        else:
            self.model.set_filter(self.column, lambda data: data in selection)


class HeaderView(QHeaderView):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        # self.setSectionsMovable(True)  # TODO to activate that we need to update variable order in the table
        self.setSectionResizeMode(QHeaderView.Interactive)
        self.setSectionsClickable(True)
        self.hovered_section = -1
        self.menu_hovered = False

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        hovered_section = self.logicalIndexAt(event.pos())
        old_hovered = self.hovered_section
        old_menu_hovered = self.menu_hovered

        self.hovered_section = hovered_section
        self.menu_hovered = self.is_menu_hovered(event.pos())

        if old_hovered != self.hovered_section or old_menu_hovered != self.menu_hovered:
            self.updateSection(old_hovered)
            self.updateSection(self.hovered_section)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        old_hovered = self.hovered_section
        self.hovered_section = -1
        self.menu_hovered = False
        self.updateSection(old_hovered)

    def paintSection(self, painter, rect, logicalIndex):
        painter.save()
        super().paintSection(painter, rect, logicalIndex)
        painter.restore()

        # If this section is being hovered, draw the menu
        if logicalIndex == self.hovered_section:
            # Set up the painter for the menu
            painter.save()
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            painter.setPen(QColor(60, 60, 60) if self.menu_hovered else QColor(120, 120, 120))
            painter.drawText(self.menu_box(rect), Qt.AlignCenter, "⋮")
            painter.restore()

    def mousePressEvent(self, event):
        index = self.logicalIndexAt(event.pos())
        if index == self.hovered_section and self.is_menu_hovered(event.pos()):
            self.parent().show_horizontal_header_menu(index, event.globalPos())
        else:
            super().mousePressEvent(event)

    def menu_box(self, rect):
        """Return the menu interration area"""
        return QRect(rect.left() + 2, rect.top() + 2, 18, 18)

    def is_menu_hovered(self, pos):
        index = self.logicalIndexAt(pos)
        if index != -1:
            rect = self.rect()
            rect.setLeft(self.sectionViewportPosition(index))
            rect.setRight(self.sectionViewportPosition(index) + self.sectionSize(index) - 1)
            return self.menu_box(rect).contains(pos)
        return False
