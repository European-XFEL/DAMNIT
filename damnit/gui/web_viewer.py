from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, QUrl
from PyQt5.QtGui import QMovie
from PyQt5.QtWebEngineCore import (QWebEngineUrlRequestJob,
                                   QWebEngineUrlScheme,
                                   QWebEngineUrlSchemeHandler)
from PyQt5.QtWebEngineWidgets import (QWebEnginePage, QWebEngineProfile,
                                      QWebEngineView)

from ..api import Damnit
from .widgets import QtWaitingSpinner

LOCAL_SCHEME = QByteArray(b"damnit")

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
        path = href.removeprefix('damn.it')
        *db_path, proposal, run, name = path.split('/')

        db_path = '/'.join(db_path)
        if db_path == '':
            db_path = int(proposal)

        try:
            _data = Damnit(db_path)[int(run), name].read()
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"ERROR: request job failed: {href!r}")
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return

        mime = QByteArray(b"text/html")
        buffer = QBuffer(job)
        buffer.setData(_data.to_html(include_plotlyjs="cdn").encode())
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        job.reply(mime, buffer)


class PlotlyPlot(QtWidgets.QWidget):
    def __init__(self, variable):
        super().__init__()
        self.setGeometry(0, 0, 1024, 768)

        self.url = f"damn.it{variable._db.path.parent}/{variable.proposal}/{variable.run}/{variable.name}"

        self.browser = QWebEngineView(self)
        profile = QWebEngineProfile.defaultProfile()
        web_page = QWebEnginePage(profile, self.browser)
        self.browser.setPage(web_page)
        self.browser.loadFinished.connect(self._handleLoaded)
        self.browser.resize(self.browser.sizeHint())

        url = QUrl(self.url)
        url.setScheme("damnit")
        self.browser.setUrl(url)

        self.spinner = QtWaitingSpinner(self)
        self.spinner.start()

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.addWidget(self.spinner)
        self.setLayout(self._layout)
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def _handleLoaded(self, ok):
        self._layout.removeWidget(self.spinner)
        self.spinner.stop()
        self.spinner.deleteLater()
        self._layout.addWidget(self.browser)

        if not ok:
            self.browser.setHtml(" <h3>414: URI Too Long</h3>")
