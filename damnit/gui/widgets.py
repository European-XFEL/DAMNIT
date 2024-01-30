from pathlib import Path

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, Qt

from ..util import icon_path


class Arrow(QLabel):

    is_open = pyqtSignal(bool)

    def __init__(self):
        super().__init__()

        self.setMaximumHeight(20)
        self.setAlignment(Qt.AlignCenter)

        self.open = True
        self.update()

        self.mouseReleaseEvent = self.toggle
        self.enterEvent = self.update
        self.leaveEvent = self.update

    def toggle(self, event):
        self.open = not self.open
        self.update()
        self.is_open.emit(self.open)

    def update(self, event=None):
        if self.open:
            if self.underMouse():
                img_path = icon_path('open-hover.png')
            else:
                img_path = icon_path('open.png')
        else:
            if self.underMouse():
                img_path = icon_path('closed-hover.png')
            else:
                img_path = icon_path('closed.png')

        pixmap = QPixmap(img_path)
        self.setPixmap(
            pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


class CollapsibleWidget(QWidget):
    """Container who can hide or show any component it contains
    """

    def __init__(self):
        super().__init__()

        self.header = Arrow()
        self.header.is_open.connect(self.collapse)

        self.content = QWidget()
        self.content_layout = QVBoxLayout()
        self.content.setLayout(self.content_layout)
        self.content.setVisible(self.header.open)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.header)
        self.layout.addWidget(self.content)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)

    def collapse(self, is_open):
        self.content.setVisible(is_open)
