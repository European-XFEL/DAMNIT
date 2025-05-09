import logging
import tempfile

import matplotlib
import mplcursors
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from mpl_pan_zoom import MouseButton, PanManager, zoom_factory
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtGui import QColor, QIcon, QPainter
from PyQt5.QtWidgets import QMessageBox

from ..api import RunVariables
from .theme import Theme

log = logging.getLogger(__name__)

CORRELATION_MSG = """\
Note: the variables being plotted are not strongly correlated.
AMORE currently expects that all arrays are train-resolved; and when plotting  
two arrays against each other that have train ID information, AMORE will use the  
train IDs to properly correlate the values in the arrays. 


If train ID information is not stored, then the arrays will be plotted directly  
against each other. If your data is not train-resolved that's fine and you can  
probably ignore this warning, otherwise make sure you use .xarray() to load data  
in your context file with train IDs.
"""

class PlotWindow(QtWidgets.QDialog):
    show_autoscale = False

    def __init__(
            self, parent=None, xlabel="", ylabel="", title=None, summary_values=False
    ):
        super().__init__()
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        
        self.main_window = parent
        self.layout = QtWidgets.QVBoxLayout(self)

        # Get current theme from main window
        self.current_theme = self.main_window.current_theme if self.main_window else Theme.LIGHT

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.summary_values = summary_values

        self.figure = Figure(figsize=(8, 5))
        self._canvas = FigureCanvas(self.figure)
        self._axis = self._canvas.figure.subplots()

        self._axis.set_xlabel(xlabel)
        self._axis.set_ylabel(ylabel)
        if title is not None:
            self.setWindowTitle(title)
            self._axis.set_title(title)

        self._navigation_toolbar = NavigationToolbar(self._canvas, self)
        self._navigation_toolbar.setIconSize(QtCore.QSize(20, 20))
        self._navigation_toolbar.layout().setSpacing(1)

        self.layout.addWidget(self._canvas)

        self._corr_warning_label = QtWidgets.QLabel(CORRELATION_MSG)
        self._corr_warning_label.setWordWrap(True)
        self._corr_warning_label.hide()
        self.layout.addWidget(self._corr_warning_label)

        self._nan_warning_label = QtWidgets.QLabel("Warning: at least one of the variables is all NaNs, "
                                                   "it may not be plotted.")
        self._nan_warning_label.setWordWrap(True)
        self._nan_warning_label.hide()
        self.layout.addWidget(self._nan_warning_label)

        if self.show_autoscale:
            self._autoscale_checkbox = QtWidgets.QCheckBox("Autoscale", self)
            self._autoscale_checkbox.setCheckState(QtCore.Qt.CheckState.Checked)
            self._autoscale_checkbox.setLayoutDirection(
                QtCore.Qt.LayoutDirection.RightToLeft
            )
            self.controls_row1 = QtWidgets.QHBoxLayout()
            self.controls_row1.addStretch()
            self.controls_row1.addWidget(self._autoscale_checkbox)
            self.layout.addLayout(self.controls_row1)

        self._display_annotations_checkbox = QtWidgets.QCheckBox(
            "Display hover annotations", self
        )
        self._display_annotations_checkbox.stateChanged.connect(self.toggle_annotations)
        self._display_annotations_checkbox.setLayoutDirection(QtCore.Qt.RightToLeft)

        # controls_row2 also holds the bins spinner for HistogramPlotWindow
        self.controls_row2 = QtWidgets.QHBoxLayout()
        self.controls_row2.addStretch()
        self.controls_row2.addWidget(self._display_annotations_checkbox)
        self.layout.addLayout(self.controls_row2)

        self.layout.addWidget(self._navigation_toolbar)

        self._cursors = []
        self._scroll_zoom = None
        self._panmanager = PanManager(self.figure, MouseButton.LEFT)

        self.figure.tight_layout()

        # Apply theme to matplotlib figure
        self._update_plot_theme()

    _autoscale_checkbox = None

    def _autoscale_enabled(self):
        return self._autoscale_checkbox and self._autoscale_checkbox.isChecked()

    def _make_cursors(self):
        return []  # Overridden in subclasses

    def _setup_scroll_zoom(self):
        # This needs to be redone when plotting changes the axes limits, so it
        # zooms on the correct position.
        if self._scroll_zoom is not None:
            self._scroll_zoom()  # Call to disconnect
        self._scroll_zoom = zoom_factory(self._axis, base_scale=1.07)

    def toggle_annotations(self, state):
        if state == QtCore.Qt.Checked:
            self._cursors.extend(self._make_cursors())
        else:
            for cursor in self._cursors:
                cursor.remove()

            self._cursors.clear()
            
    def contextMenuEvent(self, event):
        self.menu = QtWidgets.QMenu(self)
        self.zulip_action = QtWidgets.QAction('Send plot to the Logbook', self)
        self.zulip_action.triggered.connect(self.export_plot_to_zulip)
        self.menu.addAction(self.zulip_action)
        self.menu.popup(QtGui.QCursor.pos())
    
    def export_plot_to_zulip(self):
        zulip_ok = self.main_window.check_zulip_messenger()
        if not zulip_ok:
            return
            
        _, path_name = tempfile.mkstemp()
        file_name = path_name + '.png'
        self.figure.savefig(file_name, dpi=150, bbox_inches = "tight")
        
        with open(file_name, 'rb') as fn:
            self.main_window.zulip_messenger.send_figure(img = fn)

    def autoscale(self, x_min, x_max, y_min, y_max, margin=0.05):
        # Always convert the inputs to floats in case they're booleans or
        # something, which would otherwise fail later when subtracting the
        # min/max values.
        x_min = float(x_min)
        x_max = float(x_max)
        y_min = float(y_min)
        y_max = float(y_max)

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

    def update(self):
        pass  # Overridden in subclasses

    def _update_plot_toolbar_theme(self):
        if self.current_theme == Theme.DARK:
            enabled_color = QColor('white')
            disabled_color = QColor('grey')
        else:
            enabled_color = QColor('black')
            disabled_color = QColor('grey')

        for action in self._navigation_toolbar.actions():
            if action.icon() and not action.icon().isNull():
                pixmap = action.icon().pixmap(24, 24)
                painter = QPainter(pixmap)
                painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                painter.fillRect(pixmap.rect(), enabled_color if action.isEnabled() else disabled_color)
                painter.end()
                action.setIcon(QIcon(pixmap))

    def _update_plot_theme(self):
        """Update matplotlib figure colors based on current theme."""
        dark = self.current_theme == Theme.DARK

        self.figure.patch.set_facecolor('#232323' if dark else 'white')
        self._axis.set_facecolor('#232323' if dark else 'white')
        self._axis.tick_params(colors='white' if dark else 'black')
        self._axis.xaxis.label.set_color('white' if dark else 'black')
        self._axis.yaxis.label.set_color('white' if dark else 'black')
        self._axis.title.set_color('white' if dark else 'black')
        for spine in self._axis.spines.values():
            spine.set_color('white' if dark else 'black')

        self._canvas.draw()

    def update_theme(self, theme: Theme):
        """Update the window theme."""
        self.current_theme = theme
        self._update_plot_theme()
        self._update_plot_toolbar_theme()


class HistogramPlotWindow(PlotWindow):
    show_autoscale = True

    def __init__(self, parent, data, *, xlabel='', title=None, legend=None, **kwargs):
        if title is None:
            title = f"Probability density of {xlabel}"
        super().__init__(
            parent, xlabel=xlabel, ylabel="Probability density", title=title, **kwargs
        )

        self._hist_objects = []
        self._hist_kwargs = []

        self.n_bins = 5
        self._probability_density_bins = QtWidgets.QSpinBox(self)
        self._probability_density_bins.setMinimum(5)
        self._probability_density_bins.setMaximum(100000)
        self._probability_density_bins.setSingleStep(25)
        self._probability_density_bins.setValue(self.n_bins)
        self._probability_density_bins.valueChanged.connect(
            self.probability_density_bins_changed
        )
        self.controls_row2.addWidget(QtWidgets.QLabel("Number of bins"))
        self.controls_row2.addWidget(self._probability_density_bins)

        self.update_canvas(data, legend=legend)

    def _clear_data(self):
        for o in self._hist_objects:
            o.remove()
        self._hist_objects = []

    def probability_density_bins_changed(self):
        self.n_bins = self._probability_density_bins.value()

        # Regenerate the histograms
        self._clear_data()
        self._nan_warning_label.hide()

        xs, ys = [], []
        for i, data in enumerate(self.data_x):
            # Don't try to update histograms of NaN arrays
            if np.all(np.isnan(data)):
                self._nan_warning_label.show()
                continue

            y, x, patches = self._axis.hist(
                data, bins=self.n_bins, **self._hist_kwargs[i]
            )
            self._hist_objects.append(patches)

            xs.append(x)
            ys.append(y)
        self.figure.canvas.draw()

        if self._autoscale_enabled() and len(xs) > 0:
            xs_min, ys_min = xs[0].min(), 0
            xs_max, ys_max = xs[0].max(), 1

            for x, y in zip(xs, ys):
                xs_min = min(xs_min, x.min())
                xs_max = max(xs_max, x.max())
                ys_max = min(ys_max, y.max())

            self.autoscale(xs_min, xs_max, 0, ys_max, margin=0.05)

        # Update the toolbar history so that clicking the home button resets the
        # plot limits properly.
        self._canvas.toolbar.update()

    def _make_cursors(self):
        return [mplcursors.cursor(self._hist_objects, hover=True)]

    def update(self):
        if self.summary_values:
            x = self.main_window.table.numbers_for_plotting(self.xlabel)
            self.update_canvas([x])

    def update_canvas(self, xs, legend=None):
        plot_exists = bool(self._hist_objects)
        cmap = matplotlib.colormaps["tab20"]
        self._nan_warning_label.hide()
        self._clear_data()

        self._axis.grid(visible=True)
        self.data_x = xs

        self._hist_kwargs = []
        x_all, y_all = [], []
        for i, data, label in zip(
            range(len(xs)),
            xs,
            legend if legend is not None else len(xs) * [None],
        ):
            color = cmap(i / len(xs))

            plot_exists = len(self._hist_objects) == len(xs)

            self._hist_kwargs.append({
                "color": color,
                "density": True,
                "align": "mid",
                "label": label,
                "alpha": 0.5,
            })

            # Don't try to histogram NaNs
            if np.all(np.isnan(data)):
                self._nan_warning_label.show()
                self._hist_objects.append([])
                continue

            y, x, patches = self._axis.hist(
                data, bins=self.n_bins, **self._hist_kwargs[-1]
            )
            self._hist_objects.append(patches)

            x_all.append(x)
            y_all.append(y)

        if len(xs) > 1:
            self._axis.legend()
        self.figure.canvas.draw()

        if sum(a.size for a in x_all) and (self._autoscale_enabled() or not plot_exists):
            x_all = np.concatenate(x_all)
            y_all = np.concatenate(y_all)
            self.autoscale(
                np.nanmin(x_all), np.nanmax(x_all), 0, np.nanmax(y_all), margin=0.05,
            )

        self._setup_scroll_zoom()

        # Update the toolbar history so that clicking the home button resets the
        # plot limits properly.
        self._canvas.toolbar.update()


class ScatterPlotWindow(PlotWindow):
    def __init__(
        self,
        parent,
        x,
        y,
        fmt="o",
        legend=None,
        strongly_correlated=True,
        autoscale=False,
        **kwargs
    ):
        self.show_autoscale = autoscale

        super().__init__(parent, **kwargs)

        self._fmt = fmt
        self._lines = []

        if not strongly_correlated:
            self._corr_warning_label.show()

        self.update_canvas(x, y, legend=legend)

    def _make_cursors(self):
        return [mplcursors.cursor(self._lines, hover=True)]

    def update(self):
        if self.summary_values:
            x, y = self.main_window.table.numbers_for_plotting(
                self.xlabel, self.ylabel
            )
            self.update_canvas([np.array(x)], [np.array(y)])

    def update_canvas(self, xs=None, ys=None, legend=None):
        cmap = matplotlib.colormaps["tab20"]
        self._nan_warning_label.hide()

        self._axis.grid(visible=True)
        self.data_x = xs
        self.data_y = ys

        # Check for data that's all NaNs
        for data in [*xs, *ys]:
            if np.all(np.isnan(data)):
                self._nan_warning_label.show()
                break

        plot_exists = len(self._lines) == len(xs)

        self._lines = []
        for i, x, y, label in zip(
            range(len(xs)),
            xs,
            ys,
            legend if legend is not None else len(xs) * [None],
        ):
            fmt = self._fmt if len(xs) == 1 else "o"
            color = cmap(i / len(xs))

            self._lines.append(
                self._axis.plot(
                    x, y, fmt, color=color, label=label, alpha=0.5
                )[0]
            )

        if len(xs) > 1:
            self._axis.legend()
        self.figure.canvas.draw()

        if len(xs) and (self._autoscale_enabled() or not plot_exists):
            xs_min = np.nanmin([xi.min() for xi in xs])
            ys_min = np.nanmin([yi.min() for yi in ys])
            xs_max = np.nanmax([xi.max() for xi in xs])
            ys_max = np.nanmax([yi.max() for yi in ys])

            self.autoscale(xs_min, xs_max, ys_min, ys_max, margin=0.05)

        self._setup_scroll_zoom()

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

class Xarray1DPlotWindow(PlotWindow):
    def __init__(self, parent, data, **kwargs):
        super().__init__(parent, **kwargs)

        self.data = data
        if np.all(np.isnan(data)):
            self._nan_warning_label.show()

        data.plot(ax=self._axis)
        self._axis.grid()
        self._setup_scroll_zoom()
        # The plot call above can add axis labels, so we need to do this again
        self.figure.tight_layout()


class ImagePlotWindow(PlotWindow):
    _image_artist = None
    _colorbar = None
    _data_source = None

    def __init__(self, parent, image, **kwargs):
        super().__init__(parent, **kwargs)
        self._data_source = image

        self._dynamic_aspect_checkbox = QtWidgets.QCheckBox("Dynamic aspect ratio")
        self._dynamic_aspect_checkbox.setCheckState(Qt.Unchecked)
        self._dynamic_aspect_checkbox.setLayoutDirection(Qt.RightToLeft)
        self._dynamic_aspect_checkbox.stateChanged.connect(
            lambda state: self.set_dynamic_aspect(state == Qt.Checked)
        )

        self._plot_as_lines_checkbox = QtWidgets.QCheckBox("Plot as lines")
        self._plot_as_lines_checkbox.setCheckState(Qt.Unchecked)
        self._plot_as_lines_checkbox.setLayoutDirection(Qt.RightToLeft)
        self._plot_as_lines_checkbox.stateChanged.connect(self._on_plot_mode_changed)

        before_mpl_nav = self.layout.indexOf(self._navigation_toolbar)
        self.layout.insertWidget(before_mpl_nav, self._dynamic_aspect_checkbox)
        self.layout.insertWidget(before_mpl_nav + 1, self._plot_as_lines_checkbox)

        aspect_ratio = max(image.shape[:2]) / min(image.shape[:2])
        if aspect_ratio > 4:
            self._dynamic_aspect_checkbox.setCheckState(Qt.Checked)

        self.update_canvas(self._data_source)

    def _on_plot_mode_changed(self, state):
        self.update_canvas()

    def set_dynamic_aspect(self, is_dynamic):
        if self._plot_as_lines_checkbox.isChecked():
            aspect = "auto"
        else:
            aspect = "auto" if is_dynamic else "equal"
        self._axis.set_aspect(aspect)
        self.figure.canvas.draw_idle()

    def _draw_lines(self, current_data):
        """Draw data as 1D lines."""
        self._dynamic_aspect_checkbox.setEnabled(False)
        self._axis.set_aspect("auto")

        if isinstance(current_data, xr.DataArray):
            current_data.plot.line(
                hue=current_data.dims[0],
                ax=self._axis,
                add_legend=current_data.shape[0] <= 10,
            )
        else:
            # Numpy array
            self._axis.set_xlabel("Index")
            self._axis.set_ylabel("Value")
            self._axis.set_title("Line plot")
            for i, row in enumerate(current_data):
                self._axis.plot(row, label=f"Line {i}")
            if current_data.shape[0] <= 10:
                self._axis.legend()

        self.figure.tight_layout()

    def _draw_image(self, image):
        """Draw data as a 2D image."""
        self._dynamic_aspect_checkbox.setEnabled(True)
        self.set_dynamic_aspect(self._dynamic_aspect_checkbox.isChecked())

        if isinstance(image, xr.DataArray):
            if image.ndim == 2:
                vmin = np.nanquantile(image, 0.01, method='nearest')
                vmax = np.nanquantile(image, 0.99, method='nearest')
                plot_result = image.plot.imshow(
                    ax=self._axis, interpolation='antialiased', vmin=vmin, vmax=vmax,
                    origin='lower', add_colorbar=True
                )
                if isinstance(plot_result, dict):
                    self._image_artist = plot_result.get('artist')
                    self._colorbar = plot_result.get('color_bar')
                else:
                    self._image_artist = plot_result
                    self._colorbar = None
            else:  # RGB(A) colour image
                self._image_artist = image.plot.imshow(ax=self._axis, interpolation='antialiased')
            self.figure.tight_layout()
        else:
            # Numpy array
            is_color_image = image.ndim == 3
            self._image_artist = self._axis.imshow(image, interpolation="antialiased")
            if not is_color_image:
                self._colorbar = self.figure.colorbar(self._image_artist, ax=self._axis)
                vmin = np.nanquantile(image, 0.01, method='nearest')
                vmax = np.nanquantile(image, 0.99, method='nearest')
                self._image_artist.set_clim(vmin, vmax)
            else:
                self._axis.tick_params(left=False, bottom=False,
                                       labelleft=False, labelbottom=False)
                self._axis.set_xlabel("")
                self._axis.set_ylabel("")

    def update_canvas(self, data=None):
        if data is not None:
            self._data_source = data

        if self._data_source is None:
            return 

        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                log.error("Error removing colorbar", exc_info=True)
            self._colorbar = None

        self.figure.clear()
        self._axis = self.figure.add_subplot(111)
        self._update_plot_theme()
        self._image_artist = None

        if np.all(np.isnan(self._data_source)):
            self._nan_warning_label.show()
        else:
            self._nan_warning_label.hide()

        if self._plot_as_lines_checkbox.isChecked() and self._data_source.ndim == 2:
            self._draw_lines(self._data_source)
        else:
            self._draw_image(self._data_source)

        self._setup_scroll_zoom()
        self._canvas.toolbar.update()
        self.figure.canvas.draw_idle()


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


class PlottingControls:
    def __init__(self, main_window) -> None:
        self._main_window = main_window

        self.dialog = QtWidgets.QDialog(main_window)
        self.dialog.setWindowTitle("Plot Controls")
        
        plot_vertical_layout = QtWidgets.QVBoxLayout()
        plot_horizontal_layout = QtWidgets.QHBoxLayout()
        plot_parameters_horizontal_layout = QtWidgets.QHBoxLayout()

        self._button_plot = QtWidgets.QPushButton(self.dialog)
        self._button_plot.setEnabled(True)
        self._button_plot.setText("Plot summary for all runs")
        self._button_plot.clicked.connect(self._plot_summaries_clicked)

        self._button_plot_runs = QtWidgets.QPushButton("Plot for selected runs", self.dialog)
        self._button_plot_runs.clicked.connect(self._plot_run_data_clicked)

        plot_horizontal_layout.addWidget(self._button_plot)
        self._button_plot_runs.setMinimumWidth(200)
        plot_horizontal_layout.addStretch()

        self._combo_box_x_axis = SearchableComboBox(self.dialog)
        self._combo_box_y_axis = SearchableComboBox(self.dialog)

        self.vs_button = QtWidgets.QToolButton()
        self.vs_button.setText("vs.")
        self.vs_button.setToolTip("Click to swap axes")
        self.vs_button.clicked.connect(self.swap_plot_axes)

        plot_horizontal_layout.addWidget(QtWidgets.QLabel("Y:"))
        plot_horizontal_layout.addWidget(self._combo_box_y_axis)
        plot_horizontal_layout.addWidget(self.vs_button)
        plot_horizontal_layout.addWidget(QtWidgets.QLabel("X:"))
        plot_horizontal_layout.addWidget(self._combo_box_x_axis)

        self._combo_box_x_axis.setCurrentText("Run")

        plot_vertical_layout.addLayout(plot_horizontal_layout)

        plot_parameters_horizontal_layout.addWidget(self._button_plot_runs)
        self._button_plot.setMinimumWidth(200)
        plot_parameters_horizontal_layout.addStretch()

        self._toggle_probability_density = QtWidgets.QPushButton("Histogram", self.dialog)
        self._toggle_probability_density.setCheckable(True)
        self._toggle_probability_density.setChecked(False)
        self._toggle_probability_density.toggled.connect(self._combo_box_y_axis.setDisabled)

        plot_parameters_horizontal_layout.addWidget(self._toggle_probability_density)

        plot_vertical_layout.addLayout(plot_parameters_horizontal_layout)
        
        self.dialog.setLayout(plot_vertical_layout)
        self.dialog.setVisible(False)

        self._plot_windows = []

    def show_dialog(self):
        self.dialog.setVisible(True)

    def update_columns(self):
        keys = self.table.column_titles

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
    def table(self):
        return self._main_window.table

    def _plot_run_data_clicked(self):
        selected_rows = self._main_window.table_view.selected_rows()
        log.debug("Selected rows %r", [ix.row() for ix in selected_rows])

        if len(selected_rows) == 0:
            QMessageBox.warning(
                self._main_window,
                "No runs selected",
                "When plotting runs as series, you must select some runs in the table.",
            )
            return

        # Find the proposals of currently selected runs
        props_runs = [self.table.row_to_proposal_run(ix.row()) for ix in selected_rows]
        proposals = {p for (p, r) in props_runs if p is not pd.NA}

        if len(proposals) > 1:
            QMessageBox.warning(
                self._main_window,
                "Multiple proposals selected",
                "Cannot plot data for runs from different proposals",
            )
            return

        xlabel = self._combo_box_x_axis.currentText()
        if histogram := self._toggle_probability_density.isChecked():
            ylabel = None
        else:
            ylabel = self._combo_box_y_axis.currentText()

        runs = []
        xs, ys = [], []
        strongly_correlated = True

        for p, r in props_runs:
            try:
                correlated, xi, yi = self.get_run_series_data(p, r, xlabel, ylabel)
                strongly_correlated = strongly_correlated and correlated
            except Exception:
                log.warning(f"Couldn't retrieve data for run {r} ({xlabel}, {ylabel})",
                            exc_info=True)
            else:
                runs.append(r)
                xs.append(xi)
                ys.append(yi)

        if len(xs) == 0 or len(ys) == 0:
            log.warning("Error getting data for plot", exc_info=True)
            QMessageBox.warning(
                self._main_window,
                "Plotting failed",
                f"Cannot plot {ylabel} against {xlabel}, some data is missing for the run",
            )
            return

        if histogram:
            log.info("New histogram for %r", xlabel)
            canvas = HistogramPlotWindow(
                self._main_window,
                data=xs,
                xlabel=xlabel,
                legend=runs,
            )
        else:
            # Check that each X/Y pair has the same length
            for run, x, y in zip(runs, xs, ys):
                if len(x) != len(y):
                    QMessageBox.warning(
                        self._main_window,
                        "X and Y have different lengths",
                        f"'{xlabel}' and '{ylabel}' have different lengths for run {run}, they "
                        "must be the same to be plotted.\n\n"
                        f"'{xlabel}' has length {len(x)} and '{ylabel}' has length {len(y)}."
                    )
                    return

            log.info("New plot for x=%r, y=%r", xlabel, ylabel)
            canvas = ScatterPlotWindow(
                self._main_window,
                x=xs,
                y=ys,
                xlabel=xlabel,
                ylabel=ylabel,
                legend=runs,
                strongly_correlated=strongly_correlated,
            )
        canvas.setWindowTitle(f"Runs: {runs}")

        self._plot_windows.append(canvas)

        canvas.show()

    def _plot_summaries_clicked(self):
        xlabel = self._combo_box_x_axis.currentText()

        if self._toggle_probability_density.isChecked():
            log.info("New histogram for %r", xlabel)

            vals = self.table.numbers_for_plotting(xlabel)[0]
            if not vals:
                QMessageBox.warning(
                    self._main_window,
                    "Plotting failed",
                    f"No numeric data found in {xlabel}."
                )
                return

            canvas = HistogramPlotWindow(
                self._main_window,
                data=[vals],
                xlabel=xlabel,
                summary_values=True,
            )
        else:
            ylabel = self._combo_box_y_axis.currentText()
            log.info("New plot for x=%r, y=%r", xlabel, ylabel)

            xvals, yvals = self.table.numbers_for_plotting(xlabel, ylabel)
            if not xvals:
                QMessageBox.warning(
                    self._main_window,
                    "Plotting failed",
                    f"No numeric data found in {xlabel} & {ylabel}."
                )
                return

            canvas = ScatterPlotWindow(
                self._main_window,
                x=[np.array(xvals)],
                y=[np.array(yvals)],
                xlabel=xlabel,
                ylabel=ylabel,
                autoscale=True,
                summary_values=True,
            )
        self._plot_windows.append(canvas)

        canvas.show()

    def update(self):
        for plot_window in self._plot_windows:
            plot_window.update()

    def get_run_series_data(self, proposal, run, xlabel, ylabel=None):
        variables = RunVariables(self._main_window._context_path.parent, run)
        # file_name, dataset = self._main_window.get_run_file(proposal, run)

        x_quantity = self._main_window.col_title_to_name(xlabel)
        x_variable = variables[x_quantity]
        x = x_variable.read()

        if ylabel is None:
            # Get 1 column for histogram
            return True, x, None

        y_quantity = self._main_window.col_title_to_name(ylabel)
        y_variable = variables[y_quantity]
        y = y_variable.read()

        strongly_correlated = False
        if isinstance(x, xr.DataArray) and "trainId" in x.coords and \
           isinstance(y, xr.DataArray) and "trainId" in y.coords:
            tids = np.intersect1d(x.trainId, y.trainId)
            x = x.sel(trainId=tids)
            y = y.sel(trainId=tids)
            strongly_correlated = True

        return strongly_correlated, x, y
