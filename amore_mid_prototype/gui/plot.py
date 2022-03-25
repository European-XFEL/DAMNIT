import logging
import numpy as np

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox

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

        self._fmt = fmt
        self._lines = { }

        self._navigation_toolbar = NavigationToolbar(self._canvas, self)
        self._navigation_toolbar.setIconSize(QtCore.QSize(20, 20))
        self._navigation_toolbar.layout().setSpacing(1)

        self._autoscale_checkbox = QtWidgets.QCheckBox("Autoscale", self)
        if autoscale:
            self._autoscale_checkbox.setCheckState(QtCore.Qt.CheckState.Checked)
            self._autoscale_checkbox.setLayoutDirection(
                QtCore.Qt.LayoutDirection.RightToLeft
            )

        layout.addWidget(self._canvas)
        if autoscale:
            layout.addWidget(self._autoscale_checkbox)
        layout.addWidget(self._navigation_toolbar)

        self.update_canvas(x, y)

    @property
    def has_data(self):
        return len(self._lines) > 0

    def update_canvas(self, xs, ys, series_names=["default"]):
        for x, y, series in zip(xs, ys, series_names):
            fmt = self._fmt if len(xs) == 1 else "o"

            plot_exists = series in self._lines
            if not plot_exists:
                self._lines[series] = self._axis.plot([], [], fmt)[0]

            line = self._lines[series]
            line.set_data(x, y)

            if self._autoscale_checkbox.isChecked() or not plot_exists:
                self._axis.set_xlim((x.min(), x.max()))
                self._axis.set_ylim((y.min(), y.max()))

            line.figure.canvas.draw()


class Plot:
    def __init__(self, main_window) -> None:
        self._main_window = main_window
        keys = list(main_window.data.columns)

        for ki in ["Comment", "Status"]:
            keys.remove(ki)

        self._button_plot = QtWidgets.QPushButton(main_window)
        self._button_plot.setEnabled(True)
        self._button_plot.setText("Plot summary for all runs")
        self._button_plot.clicked.connect(lambda: self._button_plot_clicked(False))

        self._button_plot_runs = QtWidgets.QPushButton("Plot for selected runs", main_window)
        self._button_plot_runs.clicked.connect(lambda: self._button_plot_clicked(True))

        self.create_combo_box()
        self.update_combo_box(keys)
        self._combo_box_x_axis.setCurrentText("Run")

        self._canvas = {
            "key.x": [],
            "key.y": [],
            "canvas": [],
            "runs_as_series": []
        }

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

    def _button_plot_clicked(self, runs_as_series):
        selected_rows = self._main_window.table_view.selectionModel().selectedRows()
        xlabel = self._combo_box_x_axis.currentText()
        ylabel = self._combo_box_y_axis.currentText()

        # multiple rows can be selected
        # we could even merge multiple runs here
        for index in selected_rows:
            log.info("Selected row %d", index.row())

        if runs_as_series:
            if len(selected_rows) == 0:
                QMessageBox.warning(self._main_window, "No runs selected",
                                    "When plotting runs as series, you must select some runs in the table.")
                return

            # Find the proposals of currently selected runs
            proposals = [index.siblingAtColumn(1).data() for index in selected_rows]
            if len(set(proposals)) > 1:
                QMessageBox.warning(self._main_window, "Multiple proposals selected",
                                    "Cannot plot data for runs from different proposals")
                return

            try:
                index = selected_rows[0]
                proposal = proposals[0]
                run = index.siblingAtColumn(2).data()
                self.get_run_series_data(proposal, run, xlabel, ylabel)
            except:
                QMessageBox.warning(self._main_window, "Plotting failed",
                                    f"Cannot plot {ylabel} against {xlabel}, some data is missing for the run")
                return

        log.info("New plot for x=%r, y=%r", xlabel, ylabel)
        canvas = Canvas(self._main_window, xlabel=xlabel, ylabel=ylabel)
        if runs_as_series:
            canvas.setWindowTitle(f"Run {run}")

        self._canvas["key.x"].append(xlabel)
        self._canvas["key.y"].append(ylabel)
        self._canvas["canvas"].append(canvas)
        self._canvas["runs_as_series"].append(selected_rows if runs_as_series else None)

        self.update()

        canvas.show()

    def update(self):
        for index, (xi, yi, ci, runs_as_series) in enumerate(zip(
                self._canvas["key.x"].copy(), self._canvas["key.y"].copy(),
                self._canvas["canvas"].copy(), self._canvas["runs_as_series"].copy()
        )):
            xs = []
            ys = []

            if runs_as_series:
                # Plots with runs as series don't need to be updated (unless the
                # variables have been changed by re-running the backend on a
                # modified context file, but luckily this hasn't been
                # implemented yet).
                if ci.has_data:
                    continue

                # Find the proposals of currently selected runs
                index = runs_as_series[0]
                proposal = index.siblingAtColumn(1).data()
                runs = [index.siblingAtColumn(2).data() for index in runs_as_series]

                x, y = self.get_run_series_data(proposal, runs[0], xi, yi)
                xs.append(self._main_window.make_finite(x))
                ys.append(self._main_window.make_finite(y))
            else:
                # not nice to replace NAs/infs with nans, but better solutions require more coding
                xs.append(self._main_window.make_finite(self._data[xi])[self._data["Status"]])
                ys.append(self._main_window.make_finite(self._data[yi])[self._data["Status"]])

            log.debug("Updating plot for x=%s, y=%s", xi, yi)
            ci.update_canvas(xs, ys)

    def get_run_series_data(self, proposal, run, xlabel, ylabel):
        try:
            file_name, dataset = self._main_window.get_run_file(proposal, run)
        except:
            return

        x_quantity = self._main_window.ds_name(xlabel)
        y_quantity = self._main_window.ds_name(ylabel)

        try:
            x_ds, y_ds = dataset[x_quantity], dataset[y_quantity]
            x_tids, y_tids = x_ds["trainId"][:], y_ds["trainId"][:]
            tids, x_idxs, y_idxs = np.intersect1d(x_tids, y_tids, return_indices=True)

            x = x_ds["data"][x_idxs]
            y = y_ds["data"][y_idxs]
        except KeyError as e:
            log.warning(f"{xlabel} or {ylabel} could not be found in {file_name}")
            raise e
        finally:
            dataset.close()

        return x, y
