import logging
import pandas as pd
import numpy as np

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

log = logging.getLogger(__name__)


class Canvas(QtWidgets.QDialog):
    def __init__(
        self, parent=None, x=[], y=[], xlabel="", ylabel="", fmt="o", autoscale=True
    ):
        super().__init__()
        self.setStyleSheet("background-color: white")

        layout = QtWidgets.QVBoxLayout(self)

        self.figure = Figure(figsize=(5, 3))
        self._canvas = FigureCanvas(self.figure)
        self._axis = self._canvas.figure.subplots()
        self._axis.set_xlabel(xlabel)
        self._axis.set_ylabel(ylabel)
        self._axis.set_position([0.15, 0.15, 0.85, 0.85])
        self._axis.grid()

        (self._line,) = self._axis.plot(x, y, fmt)

        self._navigation_toolbar = NavigationToolbar(self._canvas, self)
        self._navigation_toolbar.setIconSize(QtCore.QSize(20, 20))
        self._navigation_toolbar.layout().setSpacing(1)

        if autoscale:
            self._autoscale_checkbox = QtWidgets.QCheckBox("Autoscale", self)
            self._autoscale_checkbox.setCheckState(QtCore.Qt.CheckState.Checked)
            self._autoscale_checkbox.setLayoutDirection(
                QtCore.Qt.LayoutDirection.RightToLeft
            )

        layout.addWidget(self._canvas)
        if autoscale:
            layout.addWidget(self._autoscale_checkbox)
        layout.addWidget(self._navigation_toolbar)

    def update_canvas(self, x, y):
        self._line.set_data(x, y)

        if self._autoscale_checkbox.isChecked():
            self._axis.set_xlim((x.min(), x.max()))
            self._axis.set_ylim((y.min(), y.max()))

        self._line.figure.canvas.draw()


class Plot:
    def __init__(self, main_window) -> None:
        self._main_window = main_window
        keys = list(main_window.data.columns)

        for ki in ["Comment", "Status"]:
            keys.remove(ki)

        self._button_plot = QtWidgets.QPushButton(main_window)
        self._button_plot.setEnabled(True)
        self._button_plot.setText("Plot")
        self._button_plot.clicked.connect(self._button_plot_clicked)

        self.create_combo_box()
        self.update_combo_box(keys)
        self._combo_box_x_axis.setCurrentText("Run")

        self._canvas = {"key.x": [], "key.y": [], "canvas": []}

    @property
    def _data(self):
        return self._main_window.data

    def create_combo_box(self):
        self._combo_box_x_axis = QtWidgets.QComboBox(self._main_window)
        self._combo_box_y_axis = QtWidgets.QComboBox(self._main_window)

    def update_combo_box(self, keys):
        for ki in keys:
            self._combo_box_x_axis.addItem(ki)
            self._combo_box_y_axis.addItem(ki)

    def _button_plot_clicked(self):
        xlabel = self._combo_box_x_axis.currentText()
        ylabel = self._combo_box_y_axis.currentText()

        # multiple rows can be selected
        # we could even merge multiple runs here
        for index in self._main_window.table_view.selectedIndexes():
            log.info("Selected row %d", index.row())

        log.info("New plot for x=%r, y=%r", xlabel, ylabel)
        canvas = Canvas(self._main_window, xlabel=xlabel, ylabel=ylabel)

        self._canvas["key.x"].append(xlabel)
        self._canvas["key.y"].append(ylabel)
        self._canvas["canvas"].append(canvas)

        self.update()

        canvas.show()

    def update(self):
        for xi, yi, ci in zip(
            self._canvas["key.x"], self._canvas["key.y"], self._canvas["canvas"]
        ):
            log.debug("Updating plot for x=%s, y=%s", xi, yi)

            # not nice to replace NAs with nans, but better solutions require more coding
            ci.update_canvas(
                self._data[xi].replace(pd.NA, np.nan)[self._data["Status"]],
                self._data[yi].replace(pd.NA, np.nan)[self._data["Status"]],
            )
