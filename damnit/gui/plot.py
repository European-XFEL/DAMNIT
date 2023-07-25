import logging
import numpy as np
import pandas as pd
import tempfile
from pandas.api.types import is_numeric_dtype

from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMessageBox

import mplcursors
from matplotlib.backends.backend_qtagg import (
    FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib import cm as mpl_cm
from mpl_pan_zoom import zoom_factory, PanManager, MouseButton

from .zulip_messenger import ZulipMessenger

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
        legend=None,
        plot_type="default",
        strongly_correlated=False,
        autoscale=True,
    ):
        super().__init__()
        self.setStyleSheet("background-color: white")
        self.setStyleSheet("::selection{ background-color: blue }")

        self.main_window = parent

        self.data_x, self.data_y = None, None
        self.histogram1D_bins = 5

        layout = QtWidgets.QVBoxLayout(self)

        self.plot_type = plot_type
        is_histogram = self.plot_type != "default"

        self.figure = Figure(figsize=(8, 5))
        self._canvas = FigureCanvas(self.figure)
        self._axis = self._canvas.figure.subplots()
        self._axis.set_xlabel(xlabel)
        self._axis.set_ylabel(ylabel if not is_histogram else "Probability density")
        self._axis.set_title(
            f"{ylabel} vs. {xlabel}"
            if not is_histogram
            else f"Probability density of {xlabel}"
        )

        self._fmt = fmt
        self._lines = {}
        self._image = None
        self._kwargs = {}

        self._navigation_toolbar = NavigationToolbar(self._canvas, self)
        self._navigation_toolbar.setIconSize(QtCore.QSize(20, 20))
        self._navigation_toolbar.layout().setSpacing(1)

        layout.addWidget(self._canvas)

        if not strongly_correlated:
            correlation_warning = "Note: the variables being plotted are not strongly correlated. " \
                "AMORE currently expects that all arrays are train-resolved; and when plotting " \
                "two arrays against each other that have train ID information, AMORE will use the " \
                "train IDs to properly correlate the values in the arrays." \
                "\n\n" \
                "If train ID information is not stored, then the arrays will be plotted directly " \
                "against each other. If your data is not train-resolved that's fine and you can " \
                "probably ignore this warning, otherwise make sure you use .xarray() to load data " \
                "in your context file with train IDs."

            warning_label = QtWidgets.QLabel(correlation_warning)
            warning_label.setWordWrap(True)
            layout.addWidget(warning_label)

        self._nan_warning_label = QtWidgets.QLabel("Warning: at least one of the variables is all NaNs, " \
                                                   "it may not be plotted.")
        self._nan_warning_label.setWordWrap(True)
        self._nan_warning_label.hide()
        layout.addWidget(self._nan_warning_label)

        self._autoscale_checkbox = QtWidgets.QCheckBox("Autoscale", self)
        self._autoscale_checkbox.setCheckState(QtCore.Qt.CheckState.Checked)
        self._autoscale_checkbox.setLayoutDirection(
            QtCore.Qt.LayoutDirection.RightToLeft
        )

        self._dynamic_aspect_checkbox = QtWidgets.QCheckBox("Dynamic aspect ratio")
        self._dynamic_aspect_checkbox.setCheckState(Qt.Unchecked)
        self._dynamic_aspect_checkbox.setLayoutDirection(Qt.RightToLeft)
        self._dynamic_aspect_checkbox.stateChanged.connect(
            lambda state: self.set_dynamic_aspect(state == Qt.Checked)
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
        if image is not None:
            layout.addWidget(self._dynamic_aspect_checkbox)

        layout.addWidget(self._navigation_toolbar)

        self._cursors = []
        self._zoom_factory = None
        self._panmanager = PanManager(self.figure, MouseButton.LEFT)

        self.update_canvas(x, y, image, legend=legend)

        # Take a guess at a good aspect ratio if it's an image
        if image is not None:
            aspect_ratio = max(image.shape[:2]) / min(image.shape[:2])
            if aspect_ratio > 4:
                self._dynamic_aspect_checkbox.setCheckState(Qt.Checked)

        self.figure.tight_layout()

    def toggle_annotations(self, state):
        if state == QtCore.Qt.Checked:
            for line in self._lines.values():
                self._cursors.append(mplcursors.cursor(line, hover=True))
        else:
            for cursor in self._cursors:
                cursor.remove()

            self._cursors.clear()
            
    def contextMenuEvent(self, event):
        self.menu = QtWidgets.QMenu(self)
        self.zulip_action = QtWidgets.QAction('Send plot to Zulip', self)
        self.zulip_action.triggered.connect(self.export_plot_to_zulip)
        self.menu.addAction(self.zulip_action)
        self.menu.popup(QtGui.QCursor.pos())
    
    def export_plot_to_zulip(self):
        if not isinstance(self.main_window.zulip_messenger, ZulipMessenger):
            self.main_window.zulip_messenger = ZulipMessenger(self.main_window)
            
        _, path_name = tempfile.mkstemp()
        file_name = path_name + '.png'
        self.figure.savefig(file_name, dpi=300, bbox_inches = "tight")
        
        with open(file_name, 'rb') as fn:
            self.main_window.zulip_messenger.send_figure(fn)

    @property
    def has_data(self):
        return len(self._lines) > 0

    def probability_density_bins_changed(self):
        self.histogram1D_bins = self._probability_density_bins.value()
        self.update_canvas()

    def autoscale(self, x_min, x_max, y_min, y_max, margin=0.05):
        if not np.any(np.isnan([x_min, x_max])):
            x_range = np.abs(x_max - x_min)
            x_min = x_min - x_range * margin
            x_max = x_max + x_range * margin
            self._axis.set_xlim((x_min, x_max))

        if not np.any(np.isnan([y_min, y_max])):
            y_range = np.abs(y_max - y_min)
            y_min = y_min - y_range * margin
            y_max = y_max + y_range * margin
            self._axis.set_ylim((y_min, y_max))

    def set_dynamic_aspect(self, is_dynamic):
        aspect = "auto" if is_dynamic else "equal"
        self._axis.set_aspect(aspect)
        self.figure.canvas.draw()

    def update_canvas(self, xs=None, ys=None, image=None, legend=None, series_names=["default"]):
        cmap = mpl_cm.get_cmap("tab20")
        self._nan_warning_label.hide()

        if (xs is None and ys is None) and self.plot_type == "histogram1D":
            xs, ys = [], []

            for series in self._lines.keys():
                for i in range(len(self.data_x)):
                    x, y = self.data_x[i], self.data_y[i]

                    if self.plot_type == "default":
                        self._lines[series][i].set_data(x, y)
                    if self.plot_type == "histogram1D":
                        # Don't try to update histograms of NaN arrays
                        if np.all(np.isnan(x)):
                            self._nan_warning_label.show()
                            continue

                        _ = [b.remove() for b in self._lines[series][i]]
                        y, x, patches = self._axis.hist(
                            x, bins=self.histogram1D_bins, **self._kwargs[series][i]
                        )
                        self._lines[series][i] = patches

                    xs.append(x)
                    ys.append(y)
            self.figure.canvas.draw()

            if self._autoscale_checkbox.isChecked() and len(xs) > 0:
                xs_min, ys_min = xs[0].min(), 0
                xs_max, ys_max = xs[0].max(), 1

                for x, y in zip(xs, ys):
                    xs_min = min(xs_min, x.min())
                    xs_max = max(xs_max, x.max())
                    ys_max = min(ys_max, y.max())

                self.autoscale(xs_min, xs_max, 0, ys_max, margin=0.05)

            return
        elif image is not None:
            if np.all(np.isnan(image)):
                self._nan_warning_label.show()

            if self._image is None:
                self._image = self._axis.imshow(image, interpolation="nearest")
                self.figure.colorbar(self._image, ax=self._axis)
            else:
                self._image.set_array(image)

            vmin = np.nanquantile(image, 0.01, interpolation='nearest')
            vmax = np.nanquantile(image, 0.99, interpolation='nearest')
            self._image.set_clim(vmin, vmax)
        else:
            self._axis.grid(visible=True)
            self.data_x = xs
            self.data_y = ys

            # Check for data that's all NaNs
            for data in [*xs, *ys]:
                if np.all(np.isnan(data)):
                    self._nan_warning_label.show()
                    break

            if len(xs):
                xs_min, ys_min = np.asarray(xs[0]).min(), 0
                xs_max, ys_max = np.asarray(xs[0]).max(), 1

            self._lines[series_names[0]] = []
            self._kwargs[series_names[0]] = []
            for i, x, y, series, label in zip(
                range(len(xs)),
                xs,
                ys,
                len(xs) * series_names,
                legend if legend is not None else len(xs) * [None],
            ):
                fmt = self._fmt if len(xs) == 1 else "o"
                color = cmap(i / len(xs))

                plot_exists = len(self._lines[series_names[0]]) == len(xs)
                if not plot_exists:
                    if self.plot_type == "default":
                        self._lines[series].append(
                            self._axis.plot(
                                [], [], fmt, color=color, label=label, alpha=0.5
                            )[0]
                        )

                if self.plot_type == "default":
                    self._lines[series][-1].set_data(x, y)
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

                    # Don't try to histogram NaNs
                    if np.all(np.isnan(x)):
                        self._lines[series].append([])
                        continue

                    y, x, patches = self._axis.hist(
                        x, bins=self.histogram1D_bins, **self._kwargs[series][-1]
                    )
                    self._lines[series].append(patches)

                    xs_min = min(xs_min, x.min())
                    xs_max = max(xs_max, x.max())
                    ys_max = min(ys_max, y.max())

                if len(xs) > 1:
                    self._axis.legend()
            self.figure.canvas.draw()

            if len(xs):
                if self._autoscale_checkbox.isChecked() or not plot_exists:

                    if self.plot_type != "histogram1D":
                        xs_min = np.nanmin([xi.min() for xi in xs])
                        ys_min = np.nanmin([yi.min() for yi in ys])
                        xs_max = np.nanmax([xi.max() for xi in xs])
                        ys_max = np.nanmax([yi.max() for yi in ys])

                    self.autoscale(
                        xs_min,
                        xs_max,
                        ys_min if not self.plot_type == "histogram1D" else 0,
                        ys_max,
                        margin=0.05,
                    )

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


class SearchableComboBox(QtWidgets.QComboBox):

    def __init__(self, parent = None):
        super().__init__(parent)

        self.setEditable(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)

        self.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.completer().setFilterMode(QtCore.Qt.MatchContains)

        self.currentTextChanged.connect(self.on_filter_text_changed)
        self.lineEdit().editingFinished.connect(self.on_filter_editing_finished)

    def on_filter_text_changed(self, txt):
        if self.findText(txt) != -1:
            self._text_to_select = txt
            self._last_valid = True
        else:
            self._last_valid = False

    def on_filter_editing_finished(self):
        cur_completion = self.completer().currentCompletion()
        text = self._text_to_select

        if not self._last_valid and cur_completion != "":
            text = cur_completion

        self.setCurrentText(text)

    def focusInEvent(self, event):
        r = event.reason()
        if r == QtCore.Qt.MouseFocusReason or \
           r == QtCore.Qt.TabFocusReason or \
           r == QtCore.Qt.BacktabFocusReason:
            QtCore.QTimer.singleShot(0, self.lineEdit().selectAll)
        else:
            super().focusInEvent(event)


class Plot:
    def __init__(self, main_window) -> None:
        self._main_window = main_window

        self.plot_type = "default"

        self._button_plot = QtWidgets.QPushButton(main_window)
        self._button_plot.setEnabled(True)
        self._button_plot.setText("Plot summary for all runs")
        self._button_plot.clicked.connect(lambda: self._button_plot_clicked(False))

        self._button_plot_runs = QtWidgets.QPushButton(
            "Plot for selected runs", main_window
        )
        self._button_plot_runs.clicked.connect(lambda: self._button_plot_clicked(True))

        self._toggle_probability_density = QtWidgets.QPushButton(
            "Histogram", main_window
        )
        self._toggle_probability_density.setCheckable(True)
        self._toggle_probability_density.setChecked(True)
        self._toggle_probability_density.toggle()
        self._toggle_probability_density.clicked.connect(
            self._toggle_probability_density_clicked
        )

        self._combo_box_x_axis = SearchableComboBox(self._main_window)
        self._combo_box_y_axis = SearchableComboBox(self._main_window)

        self.vs_button = QtWidgets.QToolButton()
        self.vs_button.setText("vs.")
        self.vs_button.setToolTip("Click to swap axes")
        self.vs_button.clicked.connect(self.swap_plot_axes)

        self._combo_box_x_axis.setCurrentText("Run")

        self._canvas = {
            "key.x": [],
            "key.y": [],
            "canvas": [],
            "type": [],
            "legend": [],
            "runs_as_series": [],
        }

    def update_columns(self):
        keys = list(self._main_window.data.columns)

        current_x = self._combo_box_x_axis.currentText()
        current_y = self._combo_box_y_axis.currentText()
        self._combo_box_x_axis.clear()
        self._combo_box_y_axis.clear()

        for ki in keys:
            # We don't want to plot these columns
            if ki in ["Comment", "Status", "comment_id"]:
                continue

            self._combo_box_x_axis.addItem(ki)
            self._combo_box_y_axis.addItem(ki)

        # Restore previous selection
        if current_x in keys:
            self._combo_box_x_axis.setCurrentText(current_x)
        if current_y in keys:
            self._combo_box_y_axis.setCurrentText(current_y)

    def swap_plot_axes(self):
        new_x = self._combo_box_y_axis.currentText()
        self._combo_box_y_axis.setCurrentText(self._combo_box_x_axis.currentText())
        self._combo_box_x_axis.setCurrentText(new_x)

    @property
    def _data(self):
        return self._main_window.data

    def _button_plot_clicked(self, runs_as_series):
        selected_rows = self._main_window.table_view.selectionModel().selectedRows()
        xlabel = self._combo_box_x_axis.currentText()
        ylabel = self._combo_box_y_axis.currentText()

        # Don't try to plot columns with non-numeric types, which might include
        # e.g. images or strings. Note that the run and proposal columns are
        # special cases, since we definitely might want to plot against those
        # columns but they may have pd.NA's from comment rows (which are only
        # given a timestamp).
        safe_cols = ["Proposal", "Run"]
        for label in [xlabel, ylabel]:
            finite_data = self._main_window.fix_data_for_plotting(self._data[label])
            if not label in safe_cols and not is_numeric_dtype(finite_data.dtype):
                QMessageBox.warning(self._main_window,
                                    "Plotting failed",
                                    f"'{label}' could not be plotted, its column has non-numeric data.")
                return

        # multiple rows can be selected
        # we could even merge multiple runs here
        for index in selected_rows:
            log.info("Selected row %d", index.row())

        if runs_as_series:
            if len(selected_rows) == 0:
                QMessageBox.warning(
                    self._main_window,
                    "No runs selected",
                    "When plotting runs as series, you must select some runs in the table.",
                )
                return

        # Find the proposals of currently selected runs
        proposals = [self._data.iloc[index.row()]["Proposal"] for index in selected_rows]
        proposals = [pi for pi in proposals if pi is not pd.NA]

        if len(set(proposals)) > 1:
            QMessageBox.warning(
                self._main_window,
                "Multiple proposals selected",
                "Cannot plot data for runs from different proposals",
            )
            return

        if self.plot_type == "histogram1D":
            ylabel = xlabel

        selected_runs = [self._data.iloc[index.row()]["Run"] for index in selected_rows]

        runs = []
        xs, ys = [], []
        strongly_correlated = True
        if runs_as_series:
            for p, r in zip(proposals, selected_runs):
                try:
                    correlated, xi, yi = self.get_run_series_data(p, r, xlabel, ylabel)
                    strongly_correlated = strongly_correlated and correlated
                except Exception:
                    pass
                else:
                    runs.append(r)
                    xs.append(xi)
                    ys.append(yi)

        if runs_as_series and (len(xs) == 0 or len(ys) == 0):
            log.warning("Error getting data for plot", exc_info=True)
            QMessageBox.warning(
                self._main_window,
                "Plotting failed",
                f"Cannot plot {ylabel} against {xlabel}, some data is missing for the run",
            )
            return

        if runs_as_series:
            # Check that each X/Y pair has the same length
            for run, x, y in zip(runs, xs, ys):
                if len(x) != len(y):
                    QMessageBox.warning(self._main_window,
                                        "X and Y have different lengths",
                                        f"'{xlabel}' and '{ylabel}' have different lengths for run {run}, they " \
                                        "must be the same to be plotted.\n\n" \
                                        f"'{xlabel}' has length {len(x)} and '{ylabel}' has length {len(y)}.")
                    return

        log.info("New plot for x=%r, y=%r", xlabel, ylabel)
        canvas = Canvas(
            self._main_window,
            x=xs,
            y=ys,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=runs,
            plot_type=self.plot_type,
            strongly_correlated=strongly_correlated
        )
        if runs_as_series:
            canvas.setWindowTitle(f"Runs: {runs}")

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
        for index, (xi, yi, ci, runs_as_series) in enumerate(
            zip(
                self._canvas["key.x"].copy(),
                self._canvas["key.y"].copy(),
                self._canvas["canvas"].copy(),
                self._canvas["runs_as_series"].copy(),
            )
        ):
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
                proposal = [index.siblingAtColumn(1).data() for index in runs_as_series]
                run = [index.siblingAtColumn(2).data() for index in runs_as_series]

                for pi, ri in zip(proposal, run):
                    strongly_correlated, x, y = self.get_run_series_data(pi, ri, xi, yi)
                    xs.append(self._main_window.fix_data_for_plotting(x))
                    ys.append(self._main_window.fix_data_for_plotting(y))
            else:
                # not nice to replace NAs/infs with nans, but better solutions require more coding
                xs.append(
                    self._main_window.fix_data_for_plotting(self._data[xi])[self._data["Status"]]
                )
                ys.append(
                    self._main_window.fix_data_for_plotting(self._data[yi])[self._data["Status"]]
                )

            log.debug("Updating plot for x=%s, y=%s", xi, yi)
            ci.update_canvas(xs, ys)

    def get_run_series_data(self, proposal, run, xlabel, ylabel):
        file_name, dataset = self._main_window.get_run_file(proposal, run)

        x_quantity = self._main_window.col_title_to_name(xlabel)
        y_quantity = self._main_window.col_title_to_name(ylabel)

        strongly_correlated = False
        try:
            x_ds, y_ds = dataset[x_quantity], dataset[y_quantity]

            # If these variables are train-resolved, correlate by train ID
            if "trainId" in x_ds and "trainId" in y_ds:
                x_tids, y_tids = x_ds["trainId"][:], y_ds["trainId"][:]
                tids, x_idxs, y_idxs = np.intersect1d(x_tids, y_tids, return_indices=True)

                x = x_ds["data"][x_idxs]
                y = y_ds["data"][y_idxs]
                strongly_correlated = True
            else:
                x = x_ds["data"][:]
                y = y_ds["data"][:]
        except KeyError as e:
            log.warning(f"{xlabel} or {ylabel} could not be found in {file_name}")
            raise e
        finally:
            dataset.close()

        return strongly_correlated, x, y
