import zulip
from configparser import ConfigParser
import pandas as pd
from pathlib import Path
from PyQt5 import QtWidgets

class ZulipMessenger():
    def __init__(self, parent = None):
        self.main_window = parent
        self.config_path = Path.home() / ".local" / "state" / "damnit" / ".zuliprc"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.key, self.email, self.stream, self.topic = '','','',''
    
    def send_table(self, tb):
        config_dialog = ZulipConfig(self.main_window, self, tb, kind = 'table')
        config_dialog.exec()      
  
class ZulipConfig(QtWidgets.QDialog):    
    def __init__(self, parent = None, messenger = None, msg = None, kind = None):
        super().__init__(parent)        
        self.main_window = parent
        self.messenger = messenger
        self.resize(300, 200)
        self.setWindowTitle("Zulip configuration")
        self.setModal(True)
        self.config_path = self.messenger.config_path
        self.msg = msg
        self.kind = kind

        layout = QtWidgets.QGridLayout()        

        self.setLayout(layout)
        self._set_layout(layout)

    def _set_layout(self, layout):
        self.ok_button = QtWidgets.QPushButton('Ok')
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.edit_email = QtWidgets.QLineEdit()
        self.edit_key = QtWidgets.QLineEdit()
        self.edit_stream = QtWidgets.QLineEdit()
        self.edit_topic = QtWidgets.QLineEdit()
        self.edit_title =  QtWidgets.QLineEdit()
        self.output = QtWidgets.QPlainTextEdit("Logs will be displayed here in case of error")
        self.output.setEnabled(False)
    
        layout.addWidget(QtWidgets.QLabel("<b>Email</b>*"), 0, 0)
        layout.addWidget(self.edit_email, 0, 1)
        layout.addWidget(QtWidgets.QLabel("<b>Zulip Key</b>*"), 1, 0)
        layout.addWidget(self.edit_key, 1, 1)
        layout.addWidget(QtWidgets.QLabel("<b>Stream</b>*"), 2, 0)
        layout.addWidget(self.edit_stream, 2, 1)
        layout.addWidget(QtWidgets.QLabel("<b>Topic</b>*"), 3, 0)
        layout.addWidget(self.edit_topic, 3, 1)

        layout.addWidget(QtWidgets.QLabel('<b>Title:</b>'), 5,0,1,2)
        layout.addWidget(self.edit_title, 6,0,1,2)
        layout.addWidget(self.cancel_button, 7, 0, 1, 1)
        layout.addWidget(self.ok_button, 7, 1, 1, 1)
        layout.addWidget(self.output, 8,0,2,2)
        
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button.clicked.connect(self.handle_form)

        if '' in [self.messenger.key, self.messenger.email,self.messenger.stream,self.messenger.topic]:
            self.check_cache()
          
        self.edit_email.setText(self.messenger.email)
        self.edit_key.setText(self.messenger.key)
        self.edit_stream.setText(self.messenger.stream)
        self.edit_topic.setText(self.messenger.topic)
        
    def handle_form(self):
        if self.messenger.email != self.edit_email.text() or \
            self.messenger.key != self.edit_key.text() or\
            self.messenger.stream != self.edit_stream.text() or \
            self.messenger.topic != self.edit_topic.text():
            
            self.messenger.email, self.messenger.key, self.messenger.stream, self.messenger.topic = \
            self.edit_email.text(), self.edit_key.text(), self.edit_stream.text(), self.edit_topic.text()
            
            self.save_config_file()
        self._send_msg()
    
    def _send_msg(self):
        if self.kind == 'table':
            if self.edit_title.text() != '':
                self.msg = f"### {self.edit_title.text()}" + "\n" + self.msg
            
            request =  {
            "type": "stream",
            "to": f"{self.messenger.stream}",
            "topic": f"{self.messenger.topic}",
            "content": f"{self.msg}"
            }
        
        try:
            client = zulip.Client(config_file=self.config_path)
            response = client.send_message(request)
        except Exception as exc:
            response = {'result' : '', 'msg': f"{exc}"}
        
        if response['result'] == 'success':
            self.main_window.show_status_message(f'{self.kind} sent successfully to Zulip', 
                                                timeout = 5000,
                                                stylesheet = "QStatusBar {background-color : green};")
            self.accept()
        else:
            self.output.setStyleSheet("""QPlainTextEdit { color: red };""")
            self.output.setPlainText(response['msg'])
                
        
    def save_config_file(self):
        config = ConfigParser()
        config['api'] = {'email': self.messenger.email,
                        'key': self.messenger.key,
                        'site': 'https://euxfel-da.zulipchat.com',
                        'stream': self.messenger.stream,
                        'topic' : self.messenger.topic}
        
        #Handle config_path issues?
        with open(self.config_path, 'w') as f:
            config.write(f)
                    
    def check_cache(self):
        if not self.config_path.is_file():
            return  
        
        config = ConfigParser()
        config.read(self.config_path)
        
        if not 'api' in config.sections():
            self.new_config = True
            return 
        
        if 'key' in config['api']:
            self.messenger.key = config['api']['key']
            
        if 'email' in config['api']:
            self.messenger.email = config['api']['email']
            
        if 'stream' in config['api']:
            self.messenger.stream = config['api']['stream']
            
        if 'topic' in config['api']:
            self.messenger.topic = config['api']['topic']      
        


        