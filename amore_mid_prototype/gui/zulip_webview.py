import logging

import requests

from PyQt5.QtCore import QUrl, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage


log = logging.getLogger(__name__)

class LoginInterceptingPage(QWebEnginePage):
    credentials_ready = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)

        self.username = None
        self.password = None

    def set_instance(self, zulip_instance):
        self._zulip_main_url = QUrl(f"https://{zulip_instance}/")
        self._zulip_login_url = QUrl(f"https://{zulip_instance}/accounts/login/")
        self._zulip_logout_url = QUrl(f"https://{zulip_instance}/#logout")

    def set_username(self, data):
        self.username = data

    def set_password(self, data):
        self.password = data

    def acceptNavigationRequest(self, url, navigation_type, isMainFrame):
        # Only grab the login credentials if we're going to the login endpoint
        # and we haven't just logged out.
        if self._zulip_login_url == url and self.url() != self._zulip_logout_url:
            self.runJavaScript("document.getElementById('id_username').value",
                               self.set_username)
            self.runJavaScript("document.getElementById('id_password').value",
                               self.set_password)
        elif url == self._zulip_logout_url:
            self.username = None
            self.password = None
        elif url == self._zulip_main_url and self.username is not None and self.password is not None:
            self.credentials_ready.emit()

        return True

class ZulipWebView(QWebEngineView):
    logged_in = pyqtSignal()

    def __init__(self):
        super().__init__()

        self._api_key = None
        self.zulip_instance = "euxfel-mid.zulipchat.com"

        self._page = LoginInterceptingPage(self)
        self._page.set_instance(self.zulip_instance)
        self.setPage(self._page)
        self._page.credentials_ready.connect(self.logged_in)

        self.load(QUrl(f"https://{self.zulip_instance}/login/"))

    def credentials(self):
        if self._page.username is None or self._page.password is None:
            raise RuntimeError("Not logged into Zulip, cannot retreive users API key")

        if self._api_key is None:
            r = requests.post(f"https://{self.zulip_instance}/api/v1/fetch_api_key",
                              data={"username": self._page.username, "password": self._page.password})
            r.raise_for_status()
            self._api_key = r.json()["api_key"]
            self._email = r.json()["email"]
            log.info(f"Retrieved token for {self._email}")

        return self._email, self._api_key

    def closeEvent(self, event):
        event.accept()
        self._page.deleteLater()
        self._page = None
