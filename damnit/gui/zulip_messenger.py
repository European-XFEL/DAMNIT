import zulip
import logging
import traceback
import pandas as pd

from configparser import ConfigParser
from pathlib import Path
from PyQt5 import QtWidgets, QtGui, QtCore

log = logging.getLogger(__name__)

ZULIP_SITE = "euxfel-da.zulipchat.com"
# This class should be instantiated only per opened GUI. It's only propose is to
# to hold cache information as well as a Zulip client, which might be updated.
class ZulipMessenger(QtCore.QObject):
    
    #Signal introduced to move some functionalities from ZulipConfig into
    #this class but still log error in the QDialog. Had to make the class a QObject
    #so it can have its own signals.
    show_log_message = QtCore.pyqtSignal(str)
    
    def __init__(self, parent = None):
        super( ).__init__()
        self.main_window = parent
        self.config_path = Path.home() / ".local" / "state" / "damnit" / ".zuliprc"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.key = self.email = self.stream = self.topic = ''
        self.client = None
        self.streams = []
        self.topics = []
        self.site = ZULIP_SITE

    
    def send_table(self, tb):
        config_dialog = ZulipConfig(self.main_window, self, tb, kind = 'table')
        config_dialog.exec()
        
    def send_figure(self, fn):
        config_dialog =  ZulipConfig(self.main_window, self, fn, kind='figure')
        config_dialog.exec()
        
    def save_config_file(self):
        config = ConfigParser()
        config['api'] = {'email': self.email,
                        'key': self.key,
                        'site': self.site,
                        'stream': self.stream,
                        'topic' : self.topic}
        
        #Handle config_path issues?
        with open(self.config_path, 'w') as f:
            config.write(f)
                    
    def check_cache(self):
        if not self.config_path.is_file():
            return  
        
        config = ConfigParser()
        config.read(self.config_path)
        
        if not 'api' in config.sections():
            return 
        
        if 'key' in config['api']:
            self.key = config['api']['key']
            
        if 'email' in config['api']:
            self.email = config['api']['email']
            
        if 'stream' in config['api']:
            self.stream = config['api']['stream']
            
        if 'topic' in config['api']:
            self.topic = config['api']['topic']
            
        if not '' in [self.key, self.email]:
            self.update_client()
            
    def update_client(self):
        if '' in [self.key, self.email]:
            return
        try:
            self.client = zulip.Client(config_file=self.config_path)
        except Exception as exc:
            self.show_log_message.emit(traceback.format_exc())
            log.error(exc, exc_info=True)
        
# This class handles the zulip request within a QDialog. One instance is created
# per right click action (from e.g. the table view or the plot canvas) 
class ZulipConfig(QtWidgets.QDialog):    
    def __init__(self, parent = None, messenger = None, msg = None, kind = None):
        super().__init__(parent)        
        self.main_window = parent
        self.messenger = messenger
        self.resize(600, 300)
        self.setWindowTitle("Logbook configuration")
        self.setModal(False)
        self.config_path = self.messenger.config_path
        self.msg = msg
        #E.g. table or figure
        self.kind = kind

        layout = QtWidgets.QGridLayout()        

        self.setLayout(layout)
        self._set_layout(layout)
        # print(self.messenger.show_log_message)
        
    def enable_config(self, logic = True):
        self.edit_email.setEnabled(logic)
        self.edit_key.setEnabled(logic)
        if not logic:
            self.edit_key.setEchoMode(2)
        else:
            self.edit_key.setEchoMode(0)
        
    def search_streams(self):
        self.show_msg('Fetching streams and topics, please wait', level = 'warning')
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self._search_streams)
        timer.start()
        
    def _search_streams(self):
        changes =  self.check_for_changes(include_streams=False)
        if changes:
            self.messenger.update_client()
            
        try:
            streams = self.messenger.client.get_streams()
            if streams['result'] == 'success':
                self.messenger.streams = streams['streams']
            else:
                self.show_msg(streams['msg'])
                return 
            
            self.messenger.topics = []
            for stream in self.messenger.streams:
                _topics = self.messenger.client.get_stream_topics(stream['stream_id'])
                _topics = _topics['topics'] 
                topics = [topic['name'] for topic in _topics]
                self.messenger.topics.append(topics)
    
        except Exception as exc:
            self.show_msg(traceback.format_exc())
            log.error(exc, exc_info=True)
            return
        
        self.show_default_msg()
        self.update_streams()
            
    def update_streams(self):
        if len(self.messenger.topics) != len(self.messenger.streams) or\
            len(self.messenger.topics) == 0 or len(self.messenger.streams) == 0:
                return
        for i, stream in enumerate(self.messenger.streams):
            topics = self.messenger.topics[i]
            self.edit_stream.addItem(stream['name'], topics)
                
    def update_config(self):
        changes = self.check_for_changes()
        if changes:
            self.messenger.update_client()
        self.enable_config(False)
        
    def clicker(self, index):
        self.edit_topic.clear()
        if self.edit_stream.itemData(index) is not None:
            self.edit_topic.addItems(self.edit_stream.itemData(index))

    def _set_layout(self, layout):
        self.ok_button = QtWidgets.QPushButton('Send')
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.output = QtWidgets.QPlainTextEdit()
        self.show_default_msg()
        self.output.setEnabled(False)
        
        self.edit_email = QtWidgets.QLineEdit()
        self.edit_key = QtWidgets.QLineEdit()

        self._edit_stream = QtWidgets.QLineEdit()
        self._edit_topic = QtWidgets.QLineEdit()
                
        self.edit_stream = QtWidgets.QComboBox()
        self.edit_topic = QtWidgets.QComboBox()
        self.edit_stream.setEditable(True)
        self.edit_topic.setEditable(True)
        self.edit_stream.setLineEdit(self._edit_stream)
        self.edit_topic.setLineEdit(self._edit_topic)
        self.edit_stream.activated.connect(self.clicker)
        
        self.edit_title =  QtWidgets.QLineEdit()
        self.enable_config(False)
        
        if '' in [self.messenger.key, self.messenger.email,self.messenger.stream,self.messenger.topic]:
            self.messenger.check_cache()
            if '' in [self.messenger.key, self.messenger.email]:
                self.enable_config(True)
          
        self.update_streams()
        self.edit_email.setText(self.messenger.email)
        self.edit_key.setText(self.messenger.key)
        self._edit_stream.setText(self.messenger.stream)
        self._edit_topic.setText(self.messenger.topic)
                
        self.edit_key.returnPressed.connect(self.update_config)
            
        self.button_config =  QtWidgets.QPushButton()
        self.button_config.setIcon(QtGui.QIcon(self.main_window.icon_path('config_icon.png')))
        self.button_config.setToolTip("Edit API Key and email")
        self.button_config.setCheckable(True)
        self.button_config.clicked.connect(self.enable_config)
        
        self.button_search =  QtWidgets.QPushButton()
        self.button_search.setIcon(QtGui.QIcon(self.main_window.icon_path('search_icon.png')))
        self.button_search.setToolTip("Search or refresh available streams and topics")
        self.button_search.clicked.connect(self.search_streams)

        layout.addWidget(QtWidgets.QLabel("<b>Email</b>*"), 0, 0)
        layout.addWidget(self.edit_email, 0, 1)
        layout.addWidget(QtWidgets.QLabel("<b>Zulip Key</b>*"), 1, 0)
        layout.addWidget(self.edit_key, 1, 1)
        layout.addWidget(self.button_config, 0, 2, 2, 1)
        
        layout.addWidget(QtWidgets.QLabel("<b>Stream</b>*"), 2, 0)
        layout.addWidget(self.edit_stream, 2, 1)
        layout.addWidget(QtWidgets.QLabel("<b>Topic</b>*"), 3, 0)
        layout.addWidget(self.edit_topic, 3, 1)
        layout.addWidget(self.button_search, 2, 2, 2, 1)

        layout.addWidget(QtWidgets.QLabel('<b>Title:</b>'), 4,0,1,1)
        layout.addWidget(self.edit_title, 4,1,1,2)
        layout.addWidget(self.cancel_button, 5, 0, 1, 1)
        layout.addWidget(self.ok_button, 5, 1, 1, 2)
        layout.addWidget(self.output, 7,0,1,3)
        
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button.clicked.connect(self.handle_form)

    def check_for_changes(self, include_streams = True):
        changes = self.messenger.email != self.edit_email.text() or \
            self.messenger.key != self.edit_key.text() or\
            include_streams*(
                self.messenger.stream != self._edit_stream.text() or 
                self.messenger.topic != self._edit_topic.text())
                
        if changes:                
            self.messenger.email, self.messenger.key, self.messenger.stream, self.messenger.topic = \
            self.edit_email.text(), self.edit_key.text(), self._edit_stream.text(), self._edit_topic.text()
            self.messenger.save_config_file()
            
        return changes
    
    def handle_form(self):
        if self.check_for_changes(): 
            self.messenger.update_client()
        self._send_msg()
        
        
    def show_msg(self, msg, level = 'error'):
        if level == 'error':
            self.output.setStyleSheet("""QPlainTextEdit { color: red };""")
        elif level == 'warning':
            self.output.setStyleSheet("""QPlainTextEdit { color: orange };""")
        elif level == 'debug':
            self.output.setStyleSheet("""QPlainTextEdit { color: gray };""")

        if msg == 'Invalid API key':
            msg = msg + ' ' + 'or email address'

        self.output.setPlainText(msg)
        
    def show_default_msg(self):
        self.show_msg('Logs will be printed here', level='debug')
    
    def _send_msg(self):     
        if self.kind == 'table':
            if self.edit_title.text() != '':
                self.msg = f"### {self.edit_title.text()}" + "\n" + self.msg
        
        elif self.kind == 'figure':
            try:
                upload = self.messenger.client.upload_file(self.msg)
                if upload['result'] == 'error':
                    self.show_msg(upload['msg'])
                    return
                
            except Exception as exc:
                self.show_msg(traceback.format_exc())
                log.error(exc, exc_info=True)
                return 
                
            self.msg = f"[{self.edit_title.text()}]({upload['uri']})"
            
        request =  {
        "type": "stream",
        "to": f"{self.messenger.stream}",
        "topic": f"{self.messenger.topic}",
        "content": f"{self.msg}"
        }
        
        response = self.messenger.client.send_message(request)
        if response['result'] == 'success':
            self.main_window.show_status_message(f'{self.kind} sent successfully to the Logbook', 
                                                timeout = 7000,
                                                stylesheet = "QStatusBar {background-color : green};")
            log.info(f"{self.kind} posted to the Logbook stream {self.messenger.stream}, topic {self.messenger.topic}")
            self.accept()
        else:
            self.show_msg(response['msg'])                
        
