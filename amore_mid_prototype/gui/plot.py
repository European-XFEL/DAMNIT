from cProfile import label
import logging
from unittest.mock import patch
import pandas as pd
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox

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
        image=None,
        xlabel="",
        ylabel="",
        fmt="o",
        color=None,
        legend=None,
        show_legend=False,
        plot_type="default",
    ):
        super().__init__()
        self.setStyleSheet("background-color: white")

        self._is_open = True

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
        self._axis.set_title(
            f"{ylabel} vs. {xlabel}" if not is_histogram else f"{xlabel}"
        )
        self._axis.grid(linewidth=0.25)

        self._fmt = fmt
        self._color = color
        self._legend = legend
        self._show_legend = show_legend
        self._lines = {}
        self._image = None
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

        self._legend_checkbox = QtWidgets.QCheckBox("Legend", self)
        self._legend_checkbox.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self._legend_checkbox.stateChanged.connect(self.show_legend)

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
        h1_layout.addWidget(self._legend_checkbox)
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

        self.update_canvas(x, y, image, legend=legend, plot_type=self.plot_type)
        self.figure.tight_layout()

    def closeEvent(self, event):
        self._is_open = False

        event.accept()

    def toggle_panhandler(self, enabled):
        if enabled:
            self._panhandler.enable()
        else:
            self._panhandler.disable()

    def toggle_annotations(self, state):
        if state == QtCore.Qt.Checked:
            for line in self._lines.values():
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
        self.update_canvas(plot_type="histogram1D")

    def autoscale(self, x_min, x_max, y_min, y_max, margin=0.05):
        x_range = np.abs(x_max - x_min)
        x_min = x_min - x_range * margin
        x_max = x_max + x_range * margin

        y_range = np.abs(y_max - y_min)
        y_min = y_min - y_range * margin
        y_max = y_max + y_range * margin

        return x_min, x_max, y_min, y_max

    def show_legend(self):
        self._axis.legend().set_visible(self._legend_checkbox.isChecked())
        self.figure.canvas.draw()

    def update_canvas(
        self,
        xs=None,
        ys=None,
        image=None,
        legend=None,
        plot_type="default",
        series_names=["default"],
    ):
        # this should be a "generate" (see "plot_exists") and "update" function
        cmap = mpl_cm.get_cmap("tab20")

        if xs is None and ys is None:
            xs, ys = [], []

            for series in self._lines.keys():
                for i in range(len(self.data_x)):
                    x, y = self.data_x[i], self.data_y[i]

                    if plot_type == "default":
                        self._lines[series][i].set_data(x, y)
                    if plot_type == "histogram1D":
                        _ = [b.remove() for b in self._lines[series][i]]
                        y, x, patches = self._axis.hist(
                            x, bins=self.histogram1D_bins, **self._kwargs[series][i]
                        )
                        self._lines[series][i] = patches

                    xs.append(x)
                    ys.append(y)
            self._axis.legend().set_visible(self._show_legend)
            self.figure.canvas.draw()

            xs_min, ys_min = np.asarray(xs[0]).min(), 0
            xs_max, ys_max = np.asarray(xs[0]).max(), 1

            if self._autoscale_checkbox.isChecked():
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

        elif image is not None:
            if self._image is None:
                self._image = self._axis.imshow(image, interpolation="nearest")
                self.figure.colorbar(self._image, ax=self._axis)
            else:
                self._image.set_array(image)

            vmin = np.nanquantile(image, 0.01, interpolation="nearest")
            vmax = np.nanquantile(image, 0.99, interpolation="nearest")
            self._image.set_clim(vmin, vmax)
        else:
            self._axis.grid(visible=True)
            self.data_x = xs
            self.data_y = ys

        if len(xs):
            if hasattr(xs[0], "__len__"):
                xs_min, xs_max = (
                    min([min(xi) for xi in xs]),
                    max([max(xi) for xi in xs]),
                )
            else:
                xs_min, xs_max = min(xs), max(xs)
            if hasattr(ys[0], "__len__"):
                ys_min, ys_max = (
                    min([min(yi) for yi in ys]),
                    max([max(yi) for yi in ys]),
                )
            else:
                ys_min, ys_max = min(ys), max(ys)

            if plot_type == "histogram1D":
                ys_min, ys_max = 0, 1

        self._lines[series_names[0]] = []
        self._kwargs[series_names[0]] = []

        for i, x, y, series, label in zip(
            range(len(xs)),
            xs,
            ys,
            len(xs) * series_names,
            self._legend
            if not isinstance(self._legend, str) and self._legend is not None
            else len(xs) * [self._legend],
        ):
            fmt = self._fmt if len(xs) == 1 else "o"
            color = (
                cmap(i / len(xs))
                if len(ys) == 0
                else cmap(i / len(xs) / 2)
                if self._color is None
                else self._color
            )

            plot_exists = len(self._lines[series_names[0]]) == len(xs)
            if not plot_exists:
                if plot_type == "default":
                    self._lines[series].append(
                        self._axis.plot(
                            [], [], fmt, ms=4, color=color, label=label, alpha=0.5
                        )[0]
                    )

            if plot_type == "default":
                self._lines[series][-1].set_data(x, y)
            elif plot_type == "histogram1D":
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

        self._axis.legend().set_visible(self._show_legend)
        self.figure.canvas.draw()

        if len(xs):
            if self._autoscale_checkbox.isChecked() or not plot_exists:

                if plot_type != "histogram1D":
                    xs_min = min([np.asarray(xi).min() for xi in xs])
                    ys_min = min([np.asarray(yi).min() for yi in ys])
                    xs_max = max([np.asarray(xi).max() for xi in xs])
                    ys_max = max([np.asarray(yi).max() for yi in ys])

                x_min, x_max, y_min, y_max = self.autoscale(
                    xs_min,
                    xs_max,
                    ys_min if not plot_type == "histogram1D" else 0,
                    ys_max,
                    margin=0.05,
                )
                self._axis.set_xlim((x_min, x_max))
                self._axis.set_ylim((y_min, y_max))

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

        for ki in ["Comment", "Use"] + [xi for xi in keys if xi.startswith("_")]:
            keys.remove(ki)

        self.plot_type = "default"

        self._button_plot_runs = QtWidgets.QPushButton("Plot", main_window)
        self._button_plot_runs.setToolTip("Plot data.")
        self._button_plot_runs.clicked.connect(
            lambda: self._button_plot_clicked(
                not self._toggle_plot_summary_table.isChecked(),
                self._toggle_plot_select_all_entries.isChecked(),
            )
        )

        self._toggle_plot_summary_table = QtWidgets.QCheckBox(
            "Use summary values", main_window
        )
        self._toggle_plot_summary_table.setCheckable(True)
        self._toggle_plot_summary_table.setChecked(True)

        self._toggle_plot_select_all_entries = QtWidgets.QCheckBox(
            "Use all entries", main_window
        )
        self._toggle_plot_select_all_entries.setCheckable(True)
        self._toggle_plot_select_all_entries.setChecked(False)

        self._toggle_probability_density = QtWidgets.QCheckBox("Histogram", main_window)
        self._toggle_probability_density.setCheckable(True)
        self._toggle_probability_density.setChecked(False)
        self._toggle_probability_density.clicked.connect(
            self._toggle_probability_density_clicked
        )

        self._combo_box_x_axis = QtWidgets.QComboBox(self._main_window)
        self._combo_box_y_axis = QtWidgets.QComboBox(self._main_window)

        self.vs_label = QtWidgets.QLabel("versus")

        self.update_combo_box(keys)
        self._combo_box_x_axis.setCurrentText("Run")

        self._canvas = {
            "key.x": [],
            "key.y": [],
            "canvas": [],
            "type": [],
            "indices": [],
            "non_data_field": [],
            "legend": [],
            "runs_as_series": [],
            "select_all": [],
            "updatable": [],
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

    def _plot_runs_as_series(self, xlabel, ylabel, non_data_field):
        x, y = [], []

        # it is annoying that run series can not be a function of Run or Timestamp
        # here's an ugly workaround:
        x_fake, y_fake = None, None
        if xlabel in non_data_field.keys():
            x_fake = xlabel
            xlabel = ylabel
        if ylabel in non_data_field.keys():
            y_fake = ylabel
            ylabel = xlabel

        if x_fake is None or y_fake is None:
            for pi, ri in zip(non_data_field["Proposal"], non_data_field["Run"]):
                xi, yi = self.get_run_series_data(pi, ri, xlabel, ylabel,)
                # x = self._main_window.make_finite(xi)
                # y = self._main_window.make_finite(yi)
                x.append(self._main_window.make_finite(xi))
                y.append(self._main_window.make_finite(yi))

        if x_fake is not None:
            for ki, vi in non_data_field.items():
                if x_fake == ki:
                    if xlabel not in non_data_field.keys():
                        _x = []
                        for sx in range(len(x)):
                            _x.append([vi[sx]] * len(x[sx]))
                        x = _x
                        continue
                    else:
                        x = vi
            xlabel = x_fake

        if y_fake is not None:
            ylabel = y_fake

            for ki, vi in non_data_field.items():
                if y_fake == ki:
                    if ylabel not in non_data_field.keys():
                        _y = []
                        for sy in range(len(y)):
                            _y.append([vi[sy]] * len(x[sy]))
                        y = _y
                        continue
                    else:
                        y = vi

        return x, y

    def _button_plot_clicked(self, runs_as_series, select_all):
        non_data_field = {
            "Use": None,
            "Timestamp": None,
            "Proposal": None,
            "Run": None,
        }

        indices = self._main_window.table_view.selectionModel().selectedRows()
        xlabel = self._combo_box_x_axis.currentText()
        ylabel = self._combo_box_y_axis.currentText()

        if not select_all:
            indices = [index.row() for index in indices]
        else:
            indices = [i for i in range(self._data["Run"].size)]
        # Don't try to plot columns with non-numeric types, which might include
        # e.g. images or strings. Note that the run and proposal columns are
        # special cases, since we definitely might want to plot against those
        # columns but they may have pd.NA's from comment rows (which are only
        # given a timestamp).
        dtype_warn = lambda col: QtWidgets.QMessageBox.warning(
            self._main_window,
            "Plotting failed",
            f"'{col}' could not be plotted, its column has non-numeric data.",
        )
        safe_cols = ["Proposal", "Run"]
        for label in [xlabel, ylabel]:
            if not label in safe_cols and not is_numeric_dtype(self._data[label].dtype):
                dtype_warn(label)
                return

        # multiple rows can be selected
        # we could even merge multiple runs here
        for index in selected_rows:
            log.info("Selected row %d", index.row())

        status = [self._data.iloc[index]["Use"] for index in indices]
        run = [self._data.iloc[index]["Run"] for index in indices]

        indices = [
            indices[i]
            for i, ri, si in zip(range(len(run)), run, status)
            if (ri is not pd.NA and si == True)
        ]

        print(indices)

        for ki in non_data_field.keys():
            non_data_field[ki] = [self._data.iloc[index][ki] for index in indices]

        if not select_all:
            if len(non_data_field["Run"]) == 0:
                QMessageBox.warning(
                    self._main_window,
                    "No runs selected",
                    "When plotting runs as series, you must select some runs in the table.",
                )
                return

        formatted_array = [
            "{}...{}".format(i, j) if i != j else "{}".format(i)
            for i, j in zip(
                np.array(non_data_field["Run"])[
                    [True] + list(np.diff(non_data_field["Run"]) != 1)
                ],
                np.array(non_data_field["Run"])[
                    list(np.diff(non_data_field["Run"]) != 1) + [True]
                ],
            )
        ]
        log.info("Selected runs {}".format(formatted_array))

        if len(set(non_data_field["Proposal"])) > 1:
            QMessageBox.warning(
                self._main_window,
                "Multiple proposals selected",
                "Cannot plot data for runs from different proposals",
            )
            return

        if self.plot_type == "histogram1D":
            # lazy workaround, one of the two should be None
            # reason: get_run_series_data
            xlabel = ylabel

            log.info("New histogram plot for x=%r", xlabel)

        else:
            log.info("New plot for x=%r, y=%r", xlabel, ylabel)

        if not runs_as_series:
            legend = ylabel
        else:
            legend = ["Run {}".format(ri) for ri in non_data_field["Run"]]

        canvas = Canvas(
            self._main_window,
            x=[],
            y=[],
            xlabel=xlabel,
            ylabel=ylabel,
            legend=legend,
            show_legend=False,
            plot_type=self.plot_type,
        )

        self._canvas["key.x"].append(xlabel)
        self._canvas["key.y"].append(ylabel)
        self._canvas["canvas"].append(canvas)
        self._canvas["indices"].append(indices)
        self._canvas["non_data_field"].append(non_data_field)
        self._canvas["legend"].append(legend)
        self._canvas["type"].append(self.plot_type)
        self._canvas["runs_as_series"].append(indices if runs_as_series else [])
        self._canvas["select_all"].append(select_all)
        self._canvas["updatable"].append(True)

        self.update()

        canvas.show()

    def _toggle_probability_density_clicked(self):
        if self._toggle_probability_density.isChecked():
            self._combo_box_x_axis.setEnabled(False)
            self.plot_type = "histogram1D"
        else:
            self._combo_box_x_axis.setEnabled(True)
            self.plot_type = "default"

    def update(self):
        for i in range(len(self._canvas["canvas"])):
            if not self._canvas["canvas"][i]._is_open:
                continue

            indices = self._canvas["indices"][i]

            if self._canvas["select_all"][i]:
                indices = [i for i in range(self._data["Run"].size)]
                for ki in self._canvas["non_data_field"][i].keys():
                    self._canvas["non_data_field"][i][ki] = [
                        self._data.iloc[index][ki] for index in indices
                    ]

                indices = [
                    indices[j]
                    for j, ri, si in zip(
                        range(len(self._canvas["non_data_field"][i]["Run"])),
                        self._canvas["non_data_field"][i]["Run"],
                        self._canvas["non_data_field"][i]["Use"],
                    )
                    if (ri is not pd.NA and si == True)
                ]

            # Find the proposals of currently selected runs
            for ki in self._canvas["non_data_field"][i].keys():
                self._canvas["non_data_field"][i][ki] = [
                    self._data.iloc[index][ki] for index in indices
                ]

            if len(self._canvas["runs_as_series"][i]):
                if self._canvas["updatable"]:
                    xi, yi = self._plot_runs_as_series(
                        self._canvas["key.x"][i],
                        self._canvas["key.y"][i],
                        self._canvas["non_data_field"][i],
                    )

            else:
                # not nice to replace NAs/infs with nans, but better solutions require more coding
                xi = [
                    self._main_window.make_finite(self._data[self._canvas["key.x"][i]])[
                        indices
                    ]
                ]
                yi = [
                    self._main_window.make_finite(self._data[self._canvas["key.y"][i]])[
                        indices
                    ]
                ]

            if not self._canvas["select_all"][i]:
                self._canvas["updatable"][i] = False

            log.debug(
                "Updating plot for x=%s, y=%s",
                self._canvas["key.x"],
                self._canvas["key.y"],
            )
            self._canvas["canvas"][i].update_canvas(
                xi, yi, plot_type=self._canvas["type"][i]
            )

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
