from PyQt5 import QtCore
from PyQt5.QtWidgets import QHeaderView, QMenu, QAction, QListWidgetItem, QWidgetAction, QPushButton
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QColor
from superqt import QSearchableListWidget
from fonticon_fa6 import FA6S
from superqt.fonticon import icon


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
            data = self.sourceModel().index(source_row, col, source_parent).data()
            if not expr(data):
                return False
        return True


class SearchMenu(QMenu):
    def __init__(self, column, model, parent=None):
        super().__init__(parent)
        self.column = column
        self.model = model

        action_all = QAction("All", self)
        action_all.triggered.connect(self.onActionAllTriggered)
        self.addAction(action_all)

        self.item_list = QSearchableListWidget()
        self.item_list.filter_widget.setPlaceholderText("Search")
        self.item_list.layout().setContentsMargins(0, 0, 0, 0)
        self.item_list.itemChanged.connect(self.selectionChanged)

        unique_data = set()
        for row in range(model.rowCount()):
            _item = model.index(row, column)
            if _item is None:
                continue
            data = _item.data()
            # print(f'row:{row}, col:{self.column}, data:{data}')
            unique_data.add(data)

        for data in sorted(unique_data, key=lambda item: str(item) if item is not None else ''):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, data)
            item.setData(Qt.DisplayRole, str(data or ''))
            item.setCheckState(Qt.Checked)
            self.item_list.addItem(item)

        list_action = QWidgetAction(self)
        list_action.setDefaultWidget(self.item_list)
        self.addAction(list_action)

    @QtCore.pyqtSlot()
    def onActionAllTriggered(self):
        self.model.set_filter(self.column, None)

    @QtCore.pyqtSlot()
    def selectionChanged(self):
        selection = set()
        for idx in range(self.item_list.count()):
            item = self.item_list.item(idx)
            if item.checkState() == Qt.Checked:
                data = item.data(Qt.DisplayRole)
                selection.add(item.data(Qt.DisplayRole))

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
