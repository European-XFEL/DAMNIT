import logging
import traceback
import requests
import json
import re
import pandas as pd

from configparser import ConfigParser
from PyQt5 import QtWidgets, QtCore

log = logging.getLogger(__name__)
MSG_MAX_CHAR = 10000


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
        self.selected_columns = []
            
    def send_table(self, tb):
        config_dialog = ZulipConfig(self.main_window, self, table = tb, kind = 'table')
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
    def __init__(self, parent = None, messenger = None, msg = None, kind = None, img = '', table = None):
        super().__init__(parent)        
        self.main_window = parent
        self.messenger = messenger
        self.resize(750, 300)
        self.setModal(False)
        self.config_path = self.messenger.config_path
        self.msg = msg
        self.img = img
        self.kind = kind
        self.table = table
        
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
    
    def handle_form(self):
        files = None 
        if self.kind == 'table':
            self.messenger.selected_columns = self.columns.get_selected_columns()
            _table = pd.DataFrame(self.table,columns=self.messenger.selected_columns)
            self.msg = self.split_md_table(_table)
            
            if self.edit_title.text() != '':
                self.msg[0] = f"### {self.edit_title.text()}" + "\n" + self.msg[0]
                    
        if self.kind == "figure":
            self.msg = self.edit_title.text()
            files = { 'image' : self.img }
        
        headers =  {
           "accept": "application/json",
           "X-API-key" : self.messenger.key,
        }
        
        if not isinstance(self.msg, list):
            params = {'topic' : self.edit_topic.currentText()}
            data = {'content' : self.msg}
            self._send_msg(headers, params, data, files)
        else :
            for msg in self.msg:
                params = {'topic' : self.edit_topic.currentText()}
                data = {'content' : msg}
                print([len(i) for i in self.msg])
                self._send_msg(headers, params, data, files)
                
    def _send_msg(self, headers, params, data, files):
        try:         
            response = requests.post(self.messenger.url + '/send_message',
                                 headers=headers, 
                                 params=params,
                                 data=data,
                                 files=files,
                                 timeout=5)
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