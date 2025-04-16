import math
from itertools import pairwise

import numpy as np
from PyQt5.QtCore import QRect, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QDoubleValidator
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QLineEdit, QHBoxLayout
from superqt import QDoubleRangeSlider
from superqt.utils import signals_blocked

from .util import icon_path, kde


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


class QtWaitingSpinner(QWidget):
    # from https://github.com/z3ntu/QtWaitingSpinner/blob/master/waitingspinnerwidget.py
    def __init__(self, parent, centerOnParent=True, disableParentWhenSpinning=False, modality=Qt.WindowModality.NonModal):
        super().__init__(parent)

        self._centerOnParent = centerOnParent
        self._disableParentWhenSpinning = disableParentWhenSpinning

        # WAS IN initialize()
        self._color = QColor(Qt.GlobalColor.black)
        self._roundness = 100.0
        self._minimumTrailOpacity = 3.14159265358979323846
        self._trailFadePercentage = 80.0
        self._revolutionsPerSecond = 1.57079632679489661923
        self._numberOfLines = 20
        self._lineLength = 10
        self._lineWidth = 2
        self._innerRadius = 10
        self._currentCounter = 0
        self._isSpinning = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.rotate)
        self.updateSize()
        self.updateTimer()
        self.hide()
        # END initialize()

        self.setWindowModality(modality)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, QPaintEvent):
        self.updatePosition()
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self._currentCounter >= self._numberOfLines:
            self._currentCounter = 0

        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(0, self._numberOfLines):
            painter.save()
            painter.translate(self._innerRadius + self._lineLength, self._innerRadius + self._lineLength)
            rotateAngle = float(360 * i) / float(self._numberOfLines)
            painter.rotate(rotateAngle)
            painter.translate(self._innerRadius, 0)
            distance = self.lineCountDistanceFromPrimary(i, self._currentCounter, self._numberOfLines)
            color = self.currentLineColor(distance, self._numberOfLines, self._trailFadePercentage,
                                          self._minimumTrailOpacity, self._color)
            painter.setBrush(color)
            rect = QRect(0, int(-self._lineWidth / 2), int(self._lineLength), int(self._lineWidth))
            painter.drawRoundedRect(rect, self._roundness, self._roundness, Qt.SizeMode.RelativeSize)
            painter.restore()

    def start(self):
        self.updatePosition()
        self._isSpinning = True
        self.show()

        if self.parentWidget and self._disableParentWhenSpinning:
            self.parentWidget().setEnabled(False)

        if not self._timer.isActive():
            self._timer.start()
            self._currentCounter = 0

    def stop(self):
        self._isSpinning = False
        self.hide()

        if self.parentWidget() and self._disableParentWhenSpinning:
            self.parentWidget().setEnabled(True)

        if self._timer.isActive():
            self._timer.stop()
            self._currentCounter = 0

    def setNumberOfLines(self, lines):
        self._numberOfLines = lines
        self._currentCounter = 0
        self.updateTimer()

    def setLineLength(self, length):
        self._lineLength = length
        self.updateSize()

    def setLineWidth(self, width):
        self._lineWidth = width
        self.updateSize()

    def setInnerRadius(self, radius):
        self._innerRadius = radius
        self.updateSize()

    def color(self):
        return self._color

    def roundness(self):
        return self._roundness

    def minimumTrailOpacity(self):
        return self._minimumTrailOpacity

    def trailFadePercentage(self):
        return self._trailFadePercentage

    def revolutionsPersSecond(self):
        return self._revolutionsPerSecond

    def numberOfLines(self):
        return self._numberOfLines

    def lineLength(self):
        return self._lineLength

    def lineWidth(self):
        return self._lineWidth

    def innerRadius(self):
        return self._innerRadius

    def isSpinning(self):
        return self._isSpinning

    def setRoundness(self, roundness):
        self._roundness = max(0.0, min(100.0, roundness))

    def setColor(self, color=Qt.GlobalColor.black):
        self._color = QColor(color)

    def setRevolutionsPerSecond(self, revolutionsPerSecond):
        self._revolutionsPerSecond = revolutionsPerSecond
        self.updateTimer()

    def setTrailFadePercentage(self, trail):
        self._trailFadePercentage = trail

    def setMinimumTrailOpacity(self, minimumTrailOpacity):
        self._minimumTrailOpacity = minimumTrailOpacity

    def rotate(self):
        self._currentCounter += 1
        if self._currentCounter >= self._numberOfLines:
            self._currentCounter = 0
        self.update()

    def updateSize(self):
        size = int((self._innerRadius + self._lineLength) * 2)
        self.setFixedSize(size, size)

    def updateTimer(self):
        self._timer.setInterval(int(1000 / (self._numberOfLines * self._revolutionsPerSecond)))

    def updatePosition(self):
        if self.parentWidget() and self._centerOnParent:
            self.move(int(self.parentWidget().width() / 2 - self.width() / 2),
                      int(self.parentWidget().height() / 2 - self.height() / 2))

    def lineCountDistanceFromPrimary(self, current, primary, totalNrOfLines):
        distance = primary - current
        if distance < 0:
            distance += totalNrOfLines
        return distance

    def currentLineColor(self, countDistance, totalNrOfLines, trailFadePerc, minOpacity, colorinput):
        color = QColor(colorinput)
        if countDistance == 0:
            return color
        minAlphaF = minOpacity / 100.0
        distanceThreshold = int(math.ceil((totalNrOfLines - 1) * trailFadePerc / 100.0))
        if countDistance > distanceThreshold:
            color.setAlphaF(minAlphaF)
        else:
            alphaDiff = color.alphaF() - minAlphaF
            gradient = alphaDiff / float(distanceThreshold + 1)
            resultAlpha = color.alphaF() - gradient * countDistance
            # If alpha is out of bounds, clip it.
            resultAlpha = min(1.0, max(0.0, resultAlpha))
            color.setAlphaF(resultAlpha)
        return color


class PlotLineWidget(QWidget):
    def __init__(self, x_data, y_data):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.slider_position = (x_data[0], x_data[-1])  # (left, right)
        self.setMinimumSize(100, 100)

    def set_slider_position(self, position: tuple[float, float]):
        self.slider_position = position
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Get widget dimensions
        w = self.width()
        h = self.height()

        # Calculate scaling factors
        x_min = self.x_data[0]
        x_max = self.x_data[-1]
        x_range = x_max - x_min
        x_scale = w / x_range if x_range != 0 else 1.0

        y_min, y_max = np.min(self.y_data), np.max(self.y_data)
        y_range = y_max - y_min
        padding = y_range * 0.1 if y_range != 0 else 1.0
        y_scale = (h - 20) / ((y_range) + 2 * padding)

        # Draw the main curve
        pen = QPen(QColor(0, 120, 255))
        pen.setWidth(2)
        painter.setPen(pen)

        # Convert data points to screen coordinates
        def _scale_x(value):
            return (value - x_min) * x_scale

        def _scale_y(value):
            return h - ((value - y_min + padding) * y_scale)

        x_pos = _scale_x(self.x_data).astype(int)
        y_pos = _scale_y(self.y_data).astype(int)

        for (x0, x1), (y0, y1) in zip(pairwise(x_pos), pairwise(y_pos)):
            painter.drawLine(x0, y0, x1, y1)

        left_x = int(_scale_x(self.slider_position[0]))
        right_x = int(_scale_x(self.slider_position[1]))

        pen = QPen(QColor(255, 0, 0, 150))  # Semi-transparent red
        pen.setWidth(2)
        painter.setPen(pen)

        # draw selection boundaries and gray out outer areas
        if right_x < w:
            painter.drawLine(right_x, 0, right_x, h)
            painter.fillRect(
                right_x, 0, 
                w - right_x, h,
                QBrush(QColor(128, 128, 128, 100))  # Semi-transparent gray
            )
        if left_x > 0:
            painter.drawLine(left_x, 0, left_x, h)
            painter.fillRect(
                0, 0, 
                left_x, h,
                QBrush(QColor(128, 128, 128, 100))  # Semi-transparent gray
            )


class ValueRangeWidget(QWidget):

    rangeChanged = pyqtSignal(float, float)

    def __init__(self, values: list[float], vmin: float, vmax: float, parent=None):
        super().__init__(parent)

        self.values = values
        self.vmin = vmin
        self.vmax = vmax
        self.sel = (float('-inf'), float('inf'))

        # line plot widget
        if vmin != vmax:
            x, y = kde(self.values)
            self.plot = PlotLineWidget(x, y)

            # Slider
            self.slider = QDoubleRangeSlider(Qt.Horizontal)
            self.slider.setRange(vmin, vmax)
            self.slider.setValue((vmin, vmax))
        else:
            self.plot = None
            self.slider = None

        # Min and max inputs
        self.min_input = QLineEdit()
        self.max_input = QLineEdit()
        self.min_input.setPlaceholderText("-inf")
        self.max_input.setPlaceholderText("inf")

        # Create and set validator for numerical input
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.min_input.setValidator(validator)
        self.max_input.setValidator(validator)

        min_max_layout = QHBoxLayout()
        min_max_layout.addWidget(self.min_input)
        min_max_layout.addWidget(self.max_input)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)
        layout.addWidget(self.slider)
        layout.addLayout(min_max_layout)

        # connect signals
        self.min_input.editingFinished.connect(self._on_input_changed)
        self.max_input.editingFinished.connect(self._on_input_changed)
        if self.slider is not None:
            self.slider.valueChanged.connect(self._on_range_changed)

    def update_values(self, vmin, vmax):
        self.sel = (vmin, vmax)

        # set inputs
        txt_min = "" if vmin == -math.inf else str(vmin)
        txt_max = "" if vmax == math.inf else str(vmax)

        with signals_blocked(self.min_input):
            self.min_input.setText(txt_min)
            self.min_input.setCursorPosition(0)
        with signals_blocked(self.max_input):
            self.max_input.setText(txt_max)
            self.max_input.setCursorPosition(0)

        # set plot/slider
        if self.plot is not None:
            slider_min = self.vmin if vmin == -math.inf else max(self.vmin, vmin)
            slider_max = self.vmax if vmax == math.inf else min(self.vmax, vmax)

            with signals_blocked(self.plot):
                self.plot.set_slider_position((slider_min, slider_max))
            with signals_blocked(self.slider):
                self.slider.setValue((slider_min, slider_max))

    def _on_input_changed(self):
        vmin = float(self.min_input.text() or '-inf')
        vmax = float(self.max_input.text() or 'inf')
        self.update_values(vmin, vmax)
        self.rangeChanged.emit(*self.sel)

    def _on_range_changed(self, value):
        vmin, vmax = value
        if vmin == self.vmin:
            vmin = -math.inf
        if vmax == self.vmax:
            vmax = math.inf

        self.update_values(vmin, vmax)
        self.rangeChanged.emit(*self.sel)
