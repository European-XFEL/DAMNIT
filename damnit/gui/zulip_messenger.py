import zulip
import logging
import traceback
import re
import pandas as pd

from configparser import ConfigParser
from pathlib import Path
from PyQt5 import QtWidgets, QtGui, QtCore

log = logging.getLogger(__name__)

ZULIP_SITE = "euxfel-da.zulipchat.com"
MSG_MAX_CHAR = 9000
# This class should be instantiated only per opened GUI. It's only propose is to
# to hold cache information as well as a Zulip client, which might be updated.
class ZulipMessenger(QtCore.QObject):
    
    #Signal introduced to move some functionalities from ZulipConfig into
    #this class but still log error in the QDialog. Had to make the class a QObject
    #so it can have its own signals.
    show_log_message = QtCore.pyqtSignal(str)
    
    def __init__(self, parent = None, config_path = None):
        super( ).__init__()
        self.main_window = parent
        if config_path is None:
            self.config_path = Path.home() / ".local" / "state" / "damnit" / ".zuliprc"
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.config_path = config_path
            
        self.key = self.email = self.stream = self.topic = self.stream_id = ''
        
        self.client = None
        self.topics = []
        self.site = ZULIP_SITE
        self.selected_columns = []

    def send_table(self, tb):
        config_dialog = ZulipConfig(self.main_window, self, table = tb, kind = 'table')
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
    def __init__(self, parent = None, messenger = None, msg = None, kind = None, table = None):
        super().__init__(parent)        
        self.main_window = parent
        self.messenger = messenger
        self.resize(750, 300)
        self.setWindowTitle("Logbook configuration")
        self.setModal(False)
        self.config_path = self.messenger.config_path
        self.msg = msg
        self.table =  table
        #E.g. table or figure
        self.kind = kind

        layout = QtWidgets.QGridLayout()        

        self.setLayout(layout)
        self._set_layout(layout)
        
        if len(self.messenger.topics) == 0:
            self.search_topics()
        else:
            self.update_topics()
        
    def enable_config(self, logic = True):
        self.edit_email.setEnabled(logic)
        self.edit_key.setEnabled(logic)
        if not logic:
            self.edit_key.setEchoMode(2)
        else:
            self.edit_key.setEchoMode(0)
        
    def search_topics(self):
        self.show_msg('Fetching topics, please wait', level = 'warning')
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self._search_topics)
        timer.start()
        
    def _search_topics(self):
        changes =  self.check_for_changes(include_streams=False)
        if changes:
            self.messenger.update_client()
            
        try:
            stream_id_res = self.messenger.client.get_stream_id(self.messenger.stream)
            if stream_id_res['result'] == 'success':
                self.messenger.stream_id = stream_id_res['stream_id']
                topics_res = self.messenger.client.get_stream_topics(self.messenger.stream_id)
                if topics_res['result'] == 'success':
                    self.messenger.topics = [topic['name'] for topic in topics_res['topics']]
                    self.update_topics()
                    self.show_default_msg()
                    return
                
                else:
                    self.show_msg(topics_res['msg'])
                    return
            else:
                self.show_msg(stream_id_res['msg'])
                return 
            
        except Exception as exc:
            self.show_msg(traceback.format_exc())
            log.error(exc, exc_info=True)
            return
        
    def update_topics(self):
        self.edit_topic.addItems(self.messenger.topics)
        self.edit_topic.setEditText(self.messenger.topic)
        
    def update_config(self):
        changes = self.check_for_changes()
        if changes:
            self.messenger.update_client()
        self.enable_config(False)

    def _set_layout(self, layout):
        self.ok_button = QtWidgets.QPushButton('Send')
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.output = QtWidgets.QPlainTextEdit()
        self.show_default_msg()
        self.output.setReadOnly(True)
        
        self.edit_email = QtWidgets.QLineEdit()
        self.edit_key = QtWidgets.QLineEdit()

        self.edit_stream = QtWidgets.QLineEdit()
        self.edit_stream.setDisabled(True)

        self._edit_topic = QtWidgets.QLineEdit()
                
        self.edit_topic = QtWidgets.QComboBox()
        self.edit_topic.setEditable(True)
        self.edit_topic.setLineEdit(self._edit_topic)
        
        self.edit_title =  QtWidgets.QLineEdit()
        self.enable_config(False)
        
        if '' in [self.messenger.key, self.messenger.email,self.messenger.stream,self.messenger.topic]:
            self.messenger.check_cache()
            if '' in [self.messenger.key, self.messenger.email]:
                self.enable_config(True)
          
        self.edit_email.setText(self.messenger.email)
        self.edit_key.setText(self.messenger.key)
        self.edit_stream.setText(self.messenger.stream)
        self.edit_topic.setEditText(self.messenger.topic)
                
        self.edit_key.returnPressed.connect(self.update_config)
            
        self.button_config =  QtWidgets.QPushButton()
        self.button_config.setIcon(QtGui.QIcon(self.main_window.icon_path('config_icon.png')))
        self.button_config.setToolTip("Edit API Key and email")
        self.button_config.setCheckable(True)
        self.button_config.clicked.connect(self.enable_config)
        
        self.button_search =  QtWidgets.QPushButton()
        self.button_search.setIcon(QtGui.QIcon(self.main_window.icon_path('search_icon.png')))
        self.button_search.setToolTip("Refresh available topics in the stream")
        self.button_search.clicked.connect(self.search_topics)

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
        
        if self.kind == 'table':
            line_frame = QtWidgets.QFrame()
            line_frame.setFrameShape(QtWidgets.QFrame.VLine)
            line_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
            layout.addWidget(line_frame, 0, 3, 8, 1)
            
            self.columns = CheckableListWidget(self.table.columns, self.messenger.selected_columns)

            deselect_button = QtWidgets.QPushButton('Deselect all')
            select_button = QtWidgets.QPushButton('Select all')
            deselect_button.clicked.connect(self.columns.deselect_all)
            select_button.clicked.connect(self.columns.select_all)
            
            header_layout = QtWidgets.QHBoxLayout()
            header_layout.addWidget(QtWidgets.QLabel('<b>Column selection:<b/>'))
            header_layout.addWidget(select_button)
            header_layout.addWidget(deselect_button)
            
            layout.addLayout(header_layout, 0, 4, 1, 2)
            
            columns_scroll_area =  QtWidgets.QScrollArea()
            columns_scroll_area.setWidgetResizable(True)
            columns_scroll_area.setWidget(self.columns)
            layout.addWidget(columns_scroll_area, 1, 4, 7, 2)
            layout.setColumnMinimumWidth(4, 300)

        self.cancel_button.clicked.connect(self.reject)
        self.ok_button.clicked.connect(self.handle_form)

    def check_for_changes(self, include_streams = True):
        changes = self.messenger.email != self.edit_email.text() or \
            self.messenger.key != self.edit_key.text() or\
            include_streams*(
                self.messenger.stream != self.edit_stream.text() or 
                self.messenger.topic != self._edit_topic.text())
                
        if changes:                
            self.messenger.email, self.messenger.key, self.messenger.stream, self.messenger.topic = \
            self.edit_email.text(), self.edit_key.text(), self.edit_stream.text(), self._edit_topic.text()
            self.messenger.save_config_file()
            
        return changes
    
    def handle_form(self):
        if self.check_for_changes(): 
            self.messenger.update_client()
        
        if self.kind == 'table':
            self.messenger.selected_columns = self.columns.get_selected_columns()
            _table = pd.DataFrame(self.table,columns=self.messenger.selected_columns)
            self.msg = self.split_md_table(_table)
        
        self.send_msg()
        
        
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
    
    def send_msg(self):     
        if self.kind == 'table':
            if self.edit_title.text() != '':
                self.msg[0] = f"### {self.edit_title.text()}" + "\n" + self.msg[0]
            
            for i, msg in enumerate(self.msg):
                last_one = i == len(self.msg) - 1 
                self._send_msg(msg, last_one)
                                            
            return                
        
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
            self._send_msg(self.msg)
            
    def _send_msg(self, msg, last_one=True):
        request =  {
        "type": "stream",
        "to": f"{self.messenger.stream}",
        "topic": f"{self.messenger.topic}",
        "content": f"{msg}"
        }

        response = self.messenger.client.send_message(request)
        if response['result'] == 'success' and last_one:
            self.main_window.show_status_message(f'{self.kind} sent successfully to the Logbook', 
                                                timeout = 7000,
                                                stylesheet = "QStatusBar {background-color : green};")
            log.info(f"{self.kind} posted to the Logbook stream {self.messenger.stream}, topic {self.messenger.topic}")
            self.accept()
        else:
            self.show_msg(response['msg'])                
        
    def split_md_table(self, table: pd.DataFrame, maxchar=MSG_MAX_CHAR-4):
        tables, start, stop = [], 0, 0
        while True:
            if stop == 0:
                md_table = table.iloc[start:].to_markdown(index=False, disable_numparse=True)
                md_table = self.remove_empty_spaces(md_table)
            else:
                md_table = table.iloc[start:stop].to_markdown(index=False, disable_numparse=True)
                md_table = self.remove_empty_spaces(md_table)

            if len(md_table) > maxchar:
                stop -= 1
            else:
                tables.append(f'\n{md_table}\n')
                if stop == 0:
                    break
                start, stop = stop, 0
                
        return tables
    
    def remove_empty_spaces(self, tb):
        lines = tb.strip().split('\n')
        output_lines = []

        for line in lines:
            cells = line.split('|')
            processed_cells = [re.sub(r'\s+', ' ', cell.strip()) for cell in cells]
            output_lines.append('|'.join(processed_cells))

        return '\n'.join(output_lines)
    
class CheckableListWidget(QtWidgets.QWidget):
    def __init__(self, items, selected_columns):
        super().__init__()
        
        layout =  QtWidgets.QVBoxLayout()
        self.checkboxes = []
        for item in items:
            checkbox = QtWidgets.QCheckBox(item)
            if item in selected_columns:
                checkbox.setCheckState(QtCore.Qt.Checked)
                
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)
            
        if len(selected_columns) == 0:
            self.select_all()
            
        self.setLayout(layout)
        
    def get_selected_columns(self):
        selected_columns = []
        for checkbox in self.checkboxes:
            if checkbox.checkState() == QtCore.Qt.Checked:
                selected_columns.append(checkbox.text())
        return selected_columns
    
    def select_all(self):
        for checkbox in self.checkboxes:
            checkbox.setCheckState(QtCore.Qt.Checked)
            
    def deselect_all(self):
        for checkbox in self.checkboxes:
            checkbox.setCheckState(QtCore.Qt.Unchecked)