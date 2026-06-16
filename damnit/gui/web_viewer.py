import logging
import os

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QUrl
from PyQt6.QtWebEngineCore import (
    QWebEnginePage, QWebEngineProfile, QWebEngineUrlRequestJob,
    QWebEngineUrlScheme, QWebEngineUrlSchemeHandler
)
from PyQt6.QtWebEngineWidgets import QWebEngineView

from ..api import Damnit
from .widgets import QtWaitingSpinner

log = logging.getLogger(__name__)

LOCAL_SCHEME = QByteArray(b"damn.it")
_scheme_handler = None

scheme = QWebEngineUrlScheme(LOCAL_SCHEME)
scheme.setFlags(
    QWebEngineUrlScheme.Flag.SecureScheme
    | QWebEngineUrlScheme.Flag.LocalScheme
    | QWebEngineUrlScheme.Flag.LocalAccessAllowed
)
QWebEngineUrlScheme.registerScheme(scheme)


class UrlSchemeHandler(QWebEngineUrlSchemeHandler):
    def install(self, profile):
        profile.installUrlSchemeHandler(LOCAL_SCHEME, self)

    def requestStarted(self, job):
        href = job.requestUrl().path()
        *db_path, proposal, run, name = href.split('/')

        db_path = '/'.join(db_path)
        if db_path == '':
            db_path = int(proposal)
 
        try:
            _data = Damnit(db_path)[int(run), name].preview_data()
        except Exception as ex:
            log.error(f"request job failed: {href!r}\n{ex}")
            job.fail(QWebEngineUrlRequestJob.Error.RequestFailed)
            return

        mime = QByteArray(b"text/html")
        buffer = QBuffer(job)
        # Inline Plotly JS to avoid CORS issues from the custom `damn.it:` scheme.
        buffer.setData(_data.to_html(include_plotlyjs=True).encode())
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        job.reply(mime, buffer)


def configure_webengine():
    """Set WebEngine defaults before Chromium is started."""
    existing_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").split()
    for flag in ("--no-sandbox", "--disable-gpu"):
        if flag not in existing_flags:
            existing_flags.append(flag)
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = " ".join(existing_flags)
    os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")


def install_url_scheme_handler(parent=None):
    global _scheme_handler

    if _scheme_handler is None:
        _scheme_handler = UrlSchemeHandler(parent=parent)
        _scheme_handler.install(QWebEngineProfile.defaultProfile())
    elif parent is not None and _scheme_handler.parent() is None:
        _scheme_handler.setParent(parent)

    return _scheme_handler


class PlotlyPlot(QtWidgets.QWidget):
    def __init__(self, context_dir, proposal, run, name, main_window):
        super().__init__()
        self.main_window = main_window

        self.url = f"{context_dir}/{proposal}/{run}/{name}"

        configure_webengine()
        install_url_scheme_handler(QtWidgets.QApplication.instance())

        self.browser = QWebEngineView(self)
        profile = QWebEngineProfile.defaultProfile()
        web_page = QWebEnginePage(profile, self.browser)
        self.browser.setPage(web_page)
        self.browser.loadFinished.connect(self._handleLoaded)
        self.browser.resize(self.browser.sizeHint())

        self.spinner = QtWaitingSpinner(self)
        self.spinner.start()

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.addWidget(self.spinner)
        self.setLayout(self._layout)
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0, 0, 0, 0)

        url = QUrl(self.url)
        url.setScheme(LOCAL_SCHEME.data().decode())
        self.browser.setUrl(url)

    def _handleLoaded(self, ok):
        self._layout.removeWidget(self.spinner)
        self.spinner.stop()
        self.spinner.deleteLater()
        self._layout.addWidget(self.browser)

    def showEvent(self, event):
        if not event.spontaneous():
            geom = self.geometry()
            geom.setSize(QtCore.QSize(1024, 768))
            geom.moveCenter(self.main_window.geometry().center())
            QtCore.QTimer.singleShot(0, lambda: self.setGeometry(geom))

    def closeEvent(self, event):
        self.browser.stop()
        super().closeEvent(event)
