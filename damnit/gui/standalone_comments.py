import logging

from PyQt5.QtCore import (QAbstractTableModel, QDateTime, QModelIndex, Qt,
                          QVariant)
from PyQt5.QtWidgets import (QDateTimeEdit, QDialog, QHBoxLayout, QLineEdit,
                             QPushButton, QTableView, QVBoxLayout, QHeaderView)

log = logging.getLogger(__name__)


class CommentModel(QAbstractTableModel):
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self._data = []
        self._headers = ['#', 'Timestamp', 'Comment']
        self._sort_column = 1  # Default sort by timestamp
        self._sort_order = Qt.DescendingOrder
        self.load_comments()

    def load_comments(self):
        """Load comments from the database, sorted by timestamp in descending order"""
        self._data = self.db.conn.execute("""
            SELECT rowid, timestamp, comment FROM time_comments
            ORDER BY timestamp DESC
        """).fetchall()

        self.layoutChanged.emit()

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            row = index.row()
            col = index.column()

            if col == 1:
                return QDateTime.fromSecsSinceEpoch(
                    int(self._data[row][col])
                ).toString("yyyy-MM-dd HH:mm:ss")
            else:
                return self._data[row][col]
        return QVariant()

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]
        return QVariant()

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def addComment(self, timestamp, comment):
        """Add a comment to the database"""
        if self.db is None:
            log.warning("No SQLite database in use, comment not saved")
            return

        cid = self.db.add_standalone_comment(timestamp, comment)
        log.debug("Saving time-based id %d", cid)
        # Reload comments to reflect the latest state
        self.load_comments()

    def sort(self, column, order):
        """Sort table by given column number."""
        self._sort_column = column
        self._sort_order = order
        
        self.layoutAboutToBeChanged.emit()
        self._data = sorted(self._data,
                            key=lambda x: x[column],
                            reverse=(order == Qt.DescendingOrder))
        self.layoutChanged.emit()


class TimeComment(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        # Table View
        self.tableView = QTableView()
        self.tableView.setSortingEnabled(True)
        self.model = CommentModel(self.parent().db, self)
        self.tableView.setModel(self.model)

        # Configure column widths
        header = self.tableView.horizontalHeader()
        for ix in range(self.model.columnCount() - 1):
            header.setSectionResizeMode(ix, header.ResizeToContents)
        header.setStretchLastSection(True)
        # Set word wrap for the comment column
        self.tableView.setWordWrap(True)
        # Ensure rows resize properly
        self.tableView.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        # Dialog layout
        layout.addWidget(self.tableView)

        inputLayout = QHBoxLayout()

        self.timestampInput = QDateTimeEdit()
        self.timestampInput.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.timestampInput.setDateTime(QDateTime.currentDateTime())

        self.commentInput = QLineEdit()
        self.commentInput.setPlaceholderText('Comment:')

        publishButton = QPushButton("Publish")
        publishButton.clicked.connect(self.publishComment)

        inputLayout.addWidget(self.timestampInput)
        inputLayout.addWidget(self.commentInput)
        inputLayout.addWidget(publishButton)

        layout.addLayout(inputLayout)

        self.setLayout(layout)
        self.setWindowTitle('Standalone comments')
        self.resize(600, 400)

    def publishComment(self):
        timestamp = self.timestampInput.dateTime().toSecsSinceEpoch()
        comment = self.commentInput.text()
        if comment:
            self.model.addComment(timestamp, comment)
            self.commentInput.clear()      
            self.timestampInput.setDateTime(QDateTime.currentDateTime())
            self.tableView.resizeRowsToContents()
