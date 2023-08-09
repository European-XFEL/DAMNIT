import logging
import traceback
import requests
import json

from configparser import ConfigParser
from PyQt5 import QtWidgets

log = logging.getLogger(__name__)


class ZulipMessenger():
 
    # This class should be instantiated only per opened GUI. It's only propose is to
    # to hold cache information as well as a Zulip client, which might be updated.
    def __init__(self, parent = None):
        super().__init__()
        self.main_window = parent
        self.config_path = self.main_window._context_path.parent / 'zulip.cfg'
        self.ok = self.fetch_config()
        if not self.ok:
            return
        
        self.last_topic = None
        self.ok = True
            
    def send_table(self, tb):
        config_dialog = ZulipConfig(self.main_window, self, tb, kind = 'table')
        config_dialog.exec()
        
    def send_figure(self, img):
        config_dialog =  ZulipConfig(self.main_window, self, img=img, kind='figure')
        config_dialog.exec()
        
    def fetch_config(self):
        if not self.config_path.is_file():
            self.error_dialog("Please provide a configuration file for the Logbook to enable posting data.")
            return False
        
        config = ConfigParser()
        try:
            config.read(self.config_path)
            # Only mandatory fields should be in this block
            self.key = config['ZULIP']['key']
            self.url = config['ZULIP']['url']
                        
        except Exception as exc:
            log.error(exc, exc_info=True)
            self.error_dialog("Malformed Logbook configuration file. Check the logs")
            return False 
    
        if 'topics' in config['ZULIP']:
            self.topics = json.loads(config['ZULIP']['topics'])
        else:
            self.topics = self.fetch_topics()
          
        return self.fetch_stream()  
    
    def fetch_stream(self):
        # This is not only intend to fetch the stream name 
        # but also to check the serves' health
        headers =  {
           "accept": "application/json",
           "X-API-key" : self.key,
        }
        
        try: 
            response = requests.get(self.url + '/me', headers=headers, timeout=2)
        except Exception as exc:
            log.error("Can't connect to Logbook proxy server." , exc_info=True)
            self.error_dialog("Can't connect to server.")
            return False
        
        if response.status_code == 200:
            response = json.loads(response.text)
            self.stream = response['stream']
            return True
        
        log.error("Can't connect to Logbook proxy server. Receiving: " + response.text)
        self.error_dialog("Can't connect to server.")
        return False
    
    def fetch_topics(self):
        # not yet implement, since there's some discussion on this yet. 
        # One could add the topics in the config file for the time being.
        return []

    def error_dialog(self, msg):
        dialog = QtWidgets.QMessageBox(self.main_window)
        dialog.setWindowTitle("Error")
        dialog.setText(msg)
        dialog.exec()
        
# This class handles the zulip request within a QDialog. One instance is created
# per right click action (from e.g. the table view or the plot canvas) 
class ZulipConfig(QtWidgets.QDialog):    
    def __init__(self, parent = None, messenger = None, msg = None, kind = None, img = ''):
        super().__init__(parent)        
        self.main_window = parent
        self.messenger = messenger
        self.resize(600, 300)
        self.setModal(False)
        self.config_path = self.messenger.config_path
        self.msg = msg
        self.img = img
        self.kind = kind
        
        self.setWindowTitle(f"Post {self.kind} on Logbook")

        layout = QtWidgets.QGridLayout()        
        self.setLayout(layout)
        self._set_layout(layout)

    def _set_layout(self, layout):
        self.ok_button = QtWidgets.QPushButton('Send')
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.output = QtWidgets.QPlainTextEdit()
        self.show_default_msg()
        self.output.setReadOnly(True)
                
        self.edit_stream = QtWidgets.QLineEdit()
        self.edit_stream.setEnabled(False)
        
        self._edit_topic = QtWidgets.QLineEdit()        
        self.edit_topic = QtWidgets.QComboBox()
        self.edit_topic.setEditable(True)
        self.edit_topic.setLineEdit(self._edit_topic)
        self.edit_topic.addItems(self.messenger.topics)
                
        self.edit_title =  QtWidgets.QLineEdit()
          
        self.edit_stream.setText(self.messenger.stream)
        if self.messenger.last_topic is not None:
            self._edit_topic.setText(self.messenger.last_topic)
                        
        layout.addWidget(QtWidgets.QLabel("<b>Stream</b>"), 0, 0)
        layout.addWidget(self.edit_stream, 0, 1)
        layout.addWidget(QtWidgets.QLabel("<b>Topic</b>*"), 1, 0)
        layout.addWidget(self.edit_topic, 1, 1)

        layout.addWidget(QtWidgets.QLabel('<b>Title:</b>'), 2,0,1,1)
        layout.addWidget(self.edit_title, 2,1,1,2)
        layout.addWidget(self.cancel_button, 3, 0, 1, 1)
        layout.addWidget(self.ok_button, 3, 1, 1, 2)
        layout.addWidget(self.output, 4,0,1,3)
        
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button.clicked.connect(self.handle_form)

    def handle_form(self):
        #Do I want to do something else here?
        self._send_msg()
        
    def show_msg(self, msg, level = 'error'):
        if level == 'error':
            self.output.setStyleSheet("""QPlainTextEdit { color: red };""")
        elif level == 'warning':
            self.output.setStyleSheet("""QPlainTextEdit { color: orange };""")
        elif level == 'debug':
            self.output.setStyleSheet("""QPlainTextEdit { color: gray };""")

        self.output.setPlainText(msg)
        
    def show_default_msg(self):
        self.show_msg('Logs will be printed here', level='debug')
    
    def _send_msg(self):
        files = None     
        if self.kind == 'table':
            if self.edit_title.text() != '':
                self.msg = f"### {self.edit_title.text()}" + "\n" + self.msg
        elif self.kind == "figure":
            self.msg = self.edit_title.text()
            files = { 'image' : self.img }
        
        headers =  {
           "accept": "application/json",
           "X-API-key" : self.messenger.key,
        }
    
        params = {
            'topic' : self.edit_topic.currentText(),
            'content' : self.msg,
        }
        
        try:         
            response = requests.post(self.messenger.url + '/message', 
                                 headers=headers, 
                                 params=params,
                                 files=files,
                                 timeout=3)
            if response.status_code == 200:
                response = json.loads(response.text)
                if response['result'] == 'success':
                    log.info(f"{self.kind} posted to the Logbook stream {self.messenger.stream}, topic {self.edit_topic.currentText()}")
                    self.main_window.show_status_message(f'{self.kind} sent successfully to the Logbook', 
                                                     timeout = 7000,
                                                     stylesheet = "QStatusBar {background-color : green};")
                    self.messenger.last_topic = self.edit_topic.currentText()
                    self.accept()
                else:
                    log.error("Message not sent to Logbook: " + response['msg'])
                    self.show_msg(response['msg'])
                    
            else:
                self.show_msg(response.reason)
                log.error("Message not sent to Logbook: " + response.reason)
       
        except Exception as exc:
            self.show_msg(traceback.format_exc())
            log.error(exc, exc_info=True)
