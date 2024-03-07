import logging
import tempfile

import matplotlib
import mplcursors
import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvas,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from mpl_pan_zoom import MouseButton, PanManager, zoom_factory
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

CMAP = matplotlib.colormaps["tab20"]

log = logging.getLogger(__name__)


class CanvasBase(QtWidgets.QDialog):

    _autoscale_checkbox = None

    def __init__(
        self,
        parent=None,
        autoscale=False,
    ):
        super().__init__()
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        self.setStyleSheet("QDialog {background-color: white}")

        self.main_window = parent

        layout = QtWidgets.QVBoxLayout(self)

        self.figure = Figure(figsize=(8, 5))
        self._canvas = FigureCanvas(self.figure)
        self._axis = self._canvas.figure.subplots()

        self._navigation_toolbar = NavigationToolbar(self._canvas, self)
        self._navigation_toolbar.setIconSize(QtCore.QSize(20, 20))
        self._navigation_toolbar.layout().setSpacing(1)

        layout.addWidget(self._canvas)

        self._nan_warning_label = QtWidgets.QLabel(
            "Warning: at least one of the variables is all NaNs, it may not be plotted."
        )
        self._nan_warning_label.setWordWrap(True)
        self._nan_warning_label.hide()
        layout.addWidget(self._nan_warning_label)

        if autoscale:
            self._autoscale_checkbox = QtWidgets.QCheckBox("Autoscale", self)
            self._autoscale_checkbox.setCheckState(QtCore.Qt.CheckState.Checked)
            self._autoscale_checkbox.setLayoutDirection(
                QtCore.Qt.LayoutDirection.RightToLeft
            )
            h1_layout = QtWidgets.QHBoxLayout()
            h1_layout.addStretch()
            h1_layout.addWidget(self._autoscale_checkbox)
            layout.addLayout(h1_layout)

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

        h2_layout = QtWidgets.QHBoxLayout()

        h2_layout.addStretch()
        h2_layout.addWidget(self._display_annotations_checkbox)
        layout.addLayout(h2_layout)

        layout.addWidget(self._navigation_toolbar)

        self._cursors = []
        self._zoom_factory = None
        self._panmanager = PanManager(self.figure, MouseButton.LEFT)

        self.figure.tight_layout()
        self._layout = layout

    def _autoscale_enabled(self):
        return self._autoscale_checkbox and self._autoscale_checkbox.isChecked()

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

    @property
    def has_data(self):
        raise NotImplementedError

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
        raise NotImplementedError


class ImageCanvas(CanvasBase):
    def __init__(self, image, title=None, **kwargs):
        super().__init__(**kwargs)

        self._image = None

        self._dynamic_aspect_checkbox = QtWidgets.QCheckBox("Dynamic aspect ratio")
        self._dynamic_aspect_checkbox.setCheckState(Qt.Unchecked)
        self._dynamic_aspect_checkbox.setLayoutDirection(Qt.RightToLeft)
        self._dynamic_aspect_checkbox.stateChanged.connect(
            lambda state: self.set_dynamic_aspect(state == Qt.Checked)
        )
        self._layout.addWidget(self._dynamic_aspect_checkbox)

        if title is not None:
            self._axis.set_title(title)

        self.update_canvas(image)

    def update_canvas(self, image):
        self._nan_warning_label.hide()
        is_color_image = image.ndim == 3

        if np.all(np.isnan(image)):
            self._nan_warning_label.show()

        interpolation = "antialiased" if is_color_image else "nearest"

        if self._image is None:
            self._image = self._axis.imshow(image, interpolation=interpolation)
            if not is_color_image:
                self.figure.colorbar(self._image, ax=self._axis)
        else:
            self._image.set_array(image)

        # Specific settings for color/noncolor images
        if is_color_image:
            self._axis.tick_params(left=False, bottom=False,
                                   labelleft=False, labelbottom=False)
            self._axis.set_xlabel("")
            self._axis.set_ylabel("")
        else:
            vmin = np.nanquantile(image, 0.01, method='nearest')
            vmax = np.nanquantile(image, 0.99, method='nearest')
            self._image.set_clim(vmin, vmax)

        aspect_ratio = max(image.shape[:2]) / min(image.shape[:2])
        if aspect_ratio > 4:
            self._dynamic_aspect_checkbox.setCheckState(Qt.Checked)


class PlotCanvas:
    # xlabel, ylabel, title?,
    pass


class CorrelationCanvas:
    # plot_type
    pass

