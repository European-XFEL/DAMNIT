from cProfile import label
import logging
from unittest.mock import patch
import pandas as pd
import numpy as np

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from .PyQt_elements import CheckableComboBox

import mplcursors
from matplotlib.backends.backend_qtagg import (
    FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib import cm as mpl_cm
from mpl_interactions import zoom_factory, panhandler

log = logging.getLogger(__name__)


class Canvas(QtWidgets.QDialog):
    def __init__(
        self,
        parent=None,
        x=[],
        y=[],
        y1=None,
        xlabel="",
        ylabel="",
        ylabel1=None,
        fmt="o",
        color=None,
        legend=None,
        plot_type="default",
        autoscale=True,
    ):
        super().__init__()
        self.setStyleSheet("background-color: white")

        self.data_x, self.data_y = None, None
        self.histogram1D_bins = 10

        layout = QtWidgets.QVBoxLayout(self)

        self.plot_type = plot_type
        is_histogram = self.plot_type != "default"

        self.figure = Figure(figsize=(6, 4))
        self._canvas = FigureCanvas(self.figure)
        self._axis = self._canvas.figure.subplots()
        self._axis.set_xlabel(xlabel)
        self._axis.set_ylabel(ylabel if not is_histogram else "Probability density")
        if y1 is not None:
            self._axis1 = self._axis.twinx()
            self._axis1.set_ylabel(ylabel1)
        # self._axis.set_title(
        #    f"{ylabel} vs. {xlabel}"
        #    if not is_histogram
        #    else f"Probability density of {xlabel}"
        # )
        self._axis.grid()

        self._fmt = fmt
        self._color = color
        self._legend = legend
        self._lines = {}
        self._lines1 = {}
        self._kwargs = {}

        self._navigation_toolbar = NavigationToolbar(self._canvas, self)
        self._navigation_toolbar.setIconSize(QtCore.QSize(20, 20))
        self._navigation_toolbar.layout().setSpacing(1)

        # This is a filthy hack to stop the navigation bars box-zoom feature and
        # the panhandler interfering with each other. If both of these are
        # enabled at the same time then the panhandler will move the canvas
        # while the user draws a box, which doesn't work very well. This way,
        # the panhandler is only enabled when box zoom is disabled.
        #
        # Ideally the panhandler would support matplotlibs widgetLock, see:
        # https://github.com/ianhi/mpl-interactions/pull/243#issuecomment-1101523740
        self._navigation_toolbar._actions["zoom"].triggered.connect(
            lambda checked: self.toggle_panhandler(not checked)
        )

        layout.addWidget(self._canvas)

        self._autoscale_checkbox = QtWidgets.QCheckBox("Autoscale", self)
        self._autoscale_checkbox.setCheckState(QtCore.Qt.CheckState.Checked)
        self._autoscale_checkbox.setLayoutDirection(
            QtCore.Qt.LayoutDirection.RightToLeft
        )

        self._display_annotations_checkbox = QtWidgets.QCheckBox(
            "Display hover annotations", self
        )
        self._display_annotations_checkbox.stateChanged.connect(self.toggle_annotations)
        self._display_annotations_checkbox.setLayoutDirection(QtCore.Qt.RightToLeft)

        if self.plot_type == "histogram1D":
            self._probability_density_bins = QtWidgets.QSpinBox(self)
            self._probability_density_bins.setMinimum(5)
            self._probability_density_bins.setMaximum(100000)
            self._probability_density_bins.setSingleStep(25)
            self._probability_density_bins.setValue(self.histogram1D_bins)
            self._probability_density_bins.valueChanged.connect(
                self.probability_density_bins_changed
            )

        h1_layout = QtWidgets.QHBoxLayout()
        h1_layout.addStretch()
        h1_layout.addWidget(self._autoscale_checkbox)

        h2_layout = QtWidgets.QHBoxLayout()
        if self.plot_type == "histogram1D":
            h2_layout.addWidget(QtWidgets.QLabel("Number of bins"))
            h2_layout.addWidget(self._probability_density_bins)

        h2_layout.addStretch()
        h2_layout.addWidget(self._display_annotations_checkbox)

        layout.addLayout(h1_layout)
        layout.addLayout(h2_layout)

        layout.addWidget(self._navigation_toolbar)

        self._cursors = []
        self._zoom_factory = None
        self._panhandler = panhandler(self.figure, button=1)

        self.update_canvas(x, y, y1s=y1)
        self.figure.tight_layout()

    def toggle_panhandler(self, enabled):
        if enabled:
            self._panhandler.enable()
        else:
            self._panhandler.disable()

    def toggle_annotations(self, state):
        if state == QtCore.Qt.Checked:
            for line in list(self._lines.values()) + list(self._lines1.values()):
                self._cursors.append(mplcursors.cursor(line, hover=True))
        else:
            for cursor in self._cursors:
                cursor.remove()

            self._cursors.clear()

    @property
    def has_data(self):
        return len(self._lines) > 0

    def probability_density_bins_changed(self):
        self.histogram1D_bins = self._probability_density_bins.value()
        self.update_canvas()

    def autoscale(self, x_min, x_max, y_min, y_max, margin=0.05):
        x_range = np.abs(x_max - x_min)
        x_min = x_min - x_range * margin
        x_max = x_max + x_range * margin

        y_range = np.abs(y_max - y_min)
        y_min = y_min - y_range * margin
        y_max = y_max + y_range * margin

        return x_min, x_max, y_min, y_max

        # self._axis.set_xlim((x_min, x_max))
        # self._axis.set_ylim((y_min, y_max))

    def update_canvas(self, xs=None, ys=None, y1s=None, series_names=["default"]):
        # this should be a "generate" (see "plot_exists") and "update" function
        cmap = mpl_cm.get_cmap("tab20")

        # ("qqq", xs, ys, y1s)

        if xs is None and ys is None:  # and self.plot_type == "histogram1D":
            xs, ys, y1s = [], [], []

            for series in self._lines.keys():
                for i in range(len(self.data_x)):
                    x, y = self.data_x[i], self.data_y[i]

                    if self.plot_type == "default":
                        self._lines[series][i].set_data(x, y)
                    if self.plot_type == "histogram1D":
                        _ = [b.remove() for b in self._lines[series][i]]
                        y, x, patches = self._axis.hist(
                            x, bins=self.histogram1D_bins, **self._kwargs[series][i]
                        )
                        self._lines[series][i] = patches

                    xs.append(x)
                    ys.append(y)
            self.figure.canvas.draw()

            xs_min, ys_min = np.asarray(xs[0]).min(), 0
            xs_max, ys_max = np.asarray(xs[0]).max(), 1

            if self._autoscale_checkbox.isChecked():
                # print("-----")
                # self._axis.legend().set_visible(False)
                # self._axis.legend().remove()
                # self.figure.canvas.draw()
                # print("-----")
                for x, y in zip(xs, ys):
                    xs_min = min(xs_min, x.min())
                    xs_max = max(xs_max, x.max())
                    ys_max = min(ys_max, y.max())

                x_min, x_max, y_min, y_max = self.autoscale(
                    xs_min, xs_max, 0, ys_max, margin=0.05
                )
                self._axis.set_xlim((x_min, x_max))
                self._axis.set_ylim((y_min, y_max))

            return

        self.data_x = xs
        self.data_y = ys

        if len(xs):
            xs_min, ys_min = np.asarray(xs[0]).min(), 0
            xs_max, ys_max = np.asarray(xs[0]).max(), 1

        if y1s is None:
            y1s = []

        self._lines[series_names[0]] = []
        self._lines1[series_names[0]] = []
        self._kwargs[series_names[0]] = []
        for i, x, y, series, label in zip(
            range(len(xs)),
            xs,
            ys,
            len(xs) * series_names,
            self._legend if self._legend is not None else len(xs) * [None],
        ):
            fmt = self._fmt if len(xs) == 1 else "o"
            fmt1 = "s"
            color = (
                cmap(i / len(xs))
                if len(ys) == 0
                else cmap(i / len(xs) / 2)
                if self._color is None
                else self._color
            )
            color1 = cmap((i + len(xs)) / len(xs) / 2)

            plot_exists = len(self._lines[series_names[0]]) == len(xs)
            if not plot_exists:
                if self.plot_type == "default":
                    self._lines[series].append(
                        self._axis.plot(
                            [], [], fmt, color=color, label=label, alpha=0.5
                        )[0]
                    )
                    if len(y1s) > 0:
                        self._lines1[series].append(
                            self._axis1.plot(
                                [],
                                [],
                                fmt1,
                                color=color1,
                                label=self._legend[len(xs) + 1]
                                if len(self._legend) > len(xs) + 1
                                else self._legend[1],
                                alpha=0.5,
                            )[0]
                        )

            if self.plot_type == "default":
                self._lines[series][-1].set_data(x, y)
                if len(y1s) > 0:
                    self._lines1[series][-1].set_data(x, y1s[i])
            if self.plot_type == "histogram1D":
                self._kwargs[series].append(
                    {
                        "color": color,
                        "density": True,
                        "align": "mid",
                        "label": label,
                        "alpha": 0.5,
                    }
                )
                y, x, patches = self._axis.hist(
                    x, bins=self.histogram1D_bins, **self._kwargs[series][-1]
                )
                self._lines[series].append(patches)

                xs_min = min(xs_min, x.min())
                xs_max = max(xs_max, x.max())
                ys_max = min(ys_max, y.max())

        if self._legend is not None:
            self._axis.legend(loc=0)
            if "_axis1" in self.__dict__:
                self._axis1.legend(loc=1)

        self.figure.canvas.draw()

        if len(xs):
            if self._autoscale_checkbox.isChecked() or not plot_exists:

                if self.plot_type != "histogram1D":
                    xs_min = min([np.asarray(xi).min() for xi in xs])
                    ys_min = min([np.asarray(yi).min() for yi in ys])
                    xs_max = max([np.asarray(xi).max() for xi in xs])
                    ys_max = max([np.asarray(yi).max() for yi in ys])

                x_min, x_max, y_min, y_max = self.autoscale(
                    xs_min,
                    xs_max,
                    ys_min if not self.plot_type == "histogram1D" else 0,
                    ys_max,
                    margin=0.05,
                )
                self._axis.set_xlim((x_min, x_max))
                self._axis.set_ylim((y_min, y_max))

                if len(y1s) > 0:
                    y1s_min = min([np.asarray(yi).min() for yi in y1s])
                    y1s_max = max([np.asarray(yi).max() for yi in y1s])
                    x_min, x_max, y_min, y_max = self.autoscale(
                        xs_min,
                        xs_max,
                        y1s_min if not self.plot_type == "histogram1D" else 0,
                        y1s_max,
                        margin=0.05,
                    )
                    self._axis1.set_xlim((x_min, x_max))
                    self._axis1.set_ylim((y_min, y_max))

        if self._zoom_factory is not None:
            self._zoom_factory()
        self._zoom_factory = zoom_factory(self._axis, base_scale=1.07)

        # Update the toolbar history so that clicking the home button resets the
        # plot limits properly.
        self._canvas.toolbar.update()

        # If the Run is one of the axes, enable annotations
        if self._axis.get_xlabel() == "Run" or self._axis.get_ylabel() == "Run":
            # The cursors that display the annotations do not update their
            # internal state when the data of a plot changes. So when updating
            # the data, we first disable annotations to clear existing cursors
            # and then reenable annotations to create new cursors for the
            # current data.
            self._display_annotations_checkbox.setCheckState(QtCore.Qt.Unchecked)
            self._display_annotations_checkbox.setCheckState(QtCore.Qt.Checked)


class Plot:
    def __init__(self, main_window) -> None:
        self._main_window = main_window
        keys = list(main_window.data.columns)

        for ki in ["Comment", "Status"] + [xi for xi in keys if xi.startswith("_")]:
            keys.remove(ki)

        self.plot_type = "default"

        self._button_plot = QtWidgets.QPushButton(main_window)
        self._button_plot.setEnabled(True)
        self._button_plot.setText("All runs")
        self._button_plot.setToolTip("Plot a summary for all runs.")
        self._button_plot.clicked.connect(lambda: self._button_plot_clicked(False))

        self._button_plot_runs = QtWidgets.QPushButton(
            "Data for selected runs", main_window
        )
        self._button_plot_runs.setToolTip("Plot only for selected runs.")
        self._button_plot_runs.clicked.connect(lambda: self._button_plot_clicked(True))

        self._toggle_probability_density = QtWidgets.QCheckBox("Histogram", main_window)
        self._toggle_probability_density.setCheckable(True)
        self._toggle_probability_density.setChecked(True)
        self._toggle_probability_density.toggle()
        self._toggle_probability_density.clicked.connect(
            self._toggle_probability_density_clicked
        )

        self._combo_box_x_axis = QtWidgets.QComboBox(self._main_window)
        # self._combo_box_x_axis.setToolTips("x-axis.")

        self._combo_box_y_axis = CheckableComboBox(self._main_window, maximum_items=2)
        # self._combo_box_y_axis.setToolTips("y-axis: Select up to two variables.")

        self.vs_label = QtWidgets.QLabel("versus")

        self.update_combo_box(keys)
        self._combo_box_x_axis.setCurrentText("Run")

        self._canvas = {
            "key.x": [],
            "key.y": [],
            "canvas": [],
            "type": [],
            "legend": [],
            "runs_as_series": [],
        }

    def swap_plot_axes(self):
        new_x = self._combo_box_y_axis.currentText()
        self._combo_box_y_axis.setCurrentText(self._combo_box_x_axis.currentText())
        self._combo_box_x_axis.setCurrentText(new_x)

    @property
    def _data(self):
        return self._main_window.data

    def update_combo_box(self, keys):
        for ki in keys:
            self._combo_box_x_axis.addItem(ki)
            self._combo_box_y_axis.addItem(ki)

    def _button_plot_clicked(self, runs_as_series):
        non_data_field = {"Timestamp": None, "Proposal": None, "Run": None}

        selected_rows = self._main_window.table_view.selectionModel().selectedRows()
        xlabel = self._combo_box_x_axis.currentText()
        ylabel = self._combo_box_y_axis.currentData()

        if len(ylabel) == 0 and self.plot_type == "default":
            QMessageBox.warning(
                self._main_window,
                "No quantitity selected",
                "Select the quantities you want to plot.",
            )
            return

        # multiple rows can be selected
        run = [self._data.iloc[index.row()]["Run"] for index in selected_rows]
        status = [self._data.iloc[index.row()]["Status"] for index in selected_rows]
        selected_rows = [
            selected_rows[i]
            for i, ri, si in zip(range(len(run)), run, status)
            if (ri is not pd.NA and si == True)
        ]
        non_data_field["Run"] = [
            self._data.iloc[index.row()]["Run"] for index in selected_rows
        ]

        if runs_as_series:
            if len(non_data_field["Run"]) == 0:
                QMessageBox.warning(
                    self._main_window,
                    "No runs selected",
                    "When plotting runs as series, you must select some runs in the table.",
                )
                return

        for ri in non_data_field["Run"]:
            log.info("Selected run %d", ri)

        # Find the proposals of currently selected runs
        non_data_field["Proposal"] = [
            self._data.iloc[index.row()]["Proposal"] for index in selected_rows
        ]

        if len(set(non_data_field["Proposal"])) > 1:
            QMessageBox.warning(
                self._main_window,
                "Multiple proposals selected",
                "Cannot plot data for runs from different proposals",
            )
            return

        try:
            if self.plot_type == "histogram1D":
                # lazy workaround, one of the two should be None
                ylabel = xlabel

            non_data_field["Timestamp"] = [
                self._data.iloc[index.row()]["Timestamp"] for index in selected_rows
            ]

            x, y, y1 = [], [], []
            if runs_as_series:

                # it is annoying that run series can not be a function of Run or Timestamp
                # here's an ugly workaround:
                x_fake, y_fake = None, None
                if xlabel in non_data_field.keys():
                    x_fake = xlabel
                    xlabel = ylabel[0] if self.plot_type == "default" else ylabel
                if any([yi in non_data_field.keys() for yi in ylabel]):
                    y_fake = ylabel
                    ylabel = [xlabel] if self.plot_type == "default" else xlabel

                if x_fake is None or y_fake is None:
                    for p, r in zip(non_data_field["Proposal"], non_data_field["Run"]):
                        xi, yi = self.get_run_series_data(
                            p,
                            r,
                            xlabel,
                            ylabel[0] if self.plot_type == "default" else ylabel,
                        )
                        x.append(xi)
                        y.append(yi)

                        if len(ylabel) > 1:
                            _, yi = self.get_run_series_data(
                                p,
                                r,
                                xlabel,
                                ylabel[1] if self.plot_type == "default" else ylabel,
                            )
                            y1.append(yi)
                else:
                    x, y, y1 = [0], [0], [0]

                if x_fake is not None:
                    for ki, vi in non_data_field.items():
                        if x_fake == ki:
                            if xlabel not in non_data_field.keys():
                                x = vi * len(x)
                            else:
                                x = [[i] for i in vi] * len(x)
                    xlabel = x_fake

                if y_fake is not None:
                    ylabel = y_fake

                    for ki, vi in non_data_field.items():
                        if y_fake[0] == ki:
                            y = [[i] for i in vi] * len(y)
                    if len(y_fake) > 1:
                        if y_fake[1] in non_data_field.keys():
                            for ki, vi in non_data_field.items():
                                if y_fake[1] == ki:
                                    y1 = [[i] for i in vi] * len(y)
                        else:
                            log.warning(
                                "Heterogeneous plot not implemented", exc_info=True
                            )
                            y1 = None
                            ylabel = [y_fake[0]]

        except Exception:
            log.warning("Error getting data for plot", exc_info=True)
            QMessageBox.warning(
                self._main_window,
                "Plotting failed",
                f"Cannot plot {ylabel} against {xlabel}, some data is missing for the run",
            )
            return

        log.info("New plot for x=%r, y=%r", xlabel, ylabel)
        legend = None
        if not runs_as_series:
            legend = ylabel
        else:
            if len(ylabel) == 1 or self.plot_type == "histogram1D":
                legend = ["Run {}".format(ri) for ri in non_data_field["Run"]]
            else:
                legend = [
                    "Run {}. {}".format(ri, yi)
                    for yi in ylabel
                    for ri in non_data_field["Run"]
                ]

        canvas = Canvas(
            self._main_window,
            x=x,
            y=y,
            y1=None if (len(ylabel) == 1 or self.plot_type == "histogram1D") else y1,
            xlabel=xlabel,
            ylabel=ylabel[0],
            ylabel1=None
            if (len(ylabel) == 1 or self.plot_type == "histogram1D")
            else ylabel[1],
            legend=legend,
            plot_type=self.plot_type,
        )

        self._canvas["key.x"].append(xlabel)
        self._canvas["key.y"].append(ylabel)
        self._canvas["canvas"].append(canvas)
        self._canvas["type"].append(self.plot_type)
        self._canvas["runs_as_series"].append(selected_rows if runs_as_series else None)

        self.update()

        canvas.show()

    def _toggle_probability_density_clicked(self):
        if self._toggle_probability_density.isChecked():
            self._combo_box_y_axis.setEnabled(False)
            self.plot_type = "histogram1D"
        else:
            self._combo_box_y_axis.setEnabled(True)
            self.plot_type = "default"

    def update(self):
        for index, (xi, yi, ci, plot_type, runs_as_series) in enumerate(
            zip(
                self._canvas["key.x"].copy(),
                self._canvas["key.y"].copy(),
                self._canvas["canvas"].copy(),
                self._canvas["type"].copy(),
                self._canvas["runs_as_series"].copy(),
            )
        ):
            xs = []
            ys = []
            y1s = []

            if runs_as_series:
                # Plots with runs as series don't need to be updated (unless the
                # variables have been changed by re-running the backend on a
                # modified context file, but luckily this hasn't been
                # implemented yet).
                if ci.has_data:
                    continue

                # Find the proposals of currently selected runs
                # proposal = [index.siblingAtColumn(1).data() for index in runs_as_series]
                # run = [index.siblingAtColumn(2).data() for index in runs_as_series]
                proposal = [
                    self._data.iloc[index.row()]["Proposal"] for index in runs_as_series
                ]
                run = [self._data.iloc[index.row()]["Run"] for index in runs_as_series]

                for pi, ri in zip(proposal, run):
                    x, y = self.get_run_series_data(pi, ri, xi, yi)
                    xs.append(self._main_window.make_finite(x))
                    ys.append(self._main_window.make_finite(y))
            else:
                # not nice to replace NAs/infs with nans, but better solutions require more coding
                xs.append(
                    self._main_window.make_finite(self._data[xi])[self._data["Status"]]
                )
                ys.append(
                    self._main_window.make_finite(
                        self._data[yi[0] if plot_type == "default" else yi]
                    )[self._data["Status"]]
                )
                if len(yi) > 1:
                    y1s.append(
                        self._main_window.make_finite(
                            self._data[yi[1] if plot_type == "default" else yi]
                        )[self._data["Status"]]
                    )

            log.debug("Updating plot for x=%s, y=%s", xi, yi)
            ci.update_canvas(xs, ys, y1s=y1s)

    def get_run_series_data(self, proposal, run, xlabel, ylabel):
        file_name, dataset = self._main_window.get_run_file(proposal, run)

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
