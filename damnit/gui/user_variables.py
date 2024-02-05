import re

from PyQt5 import QtCore, QtGui, QtWidgets

from ..backend.user_variables import value_types_by_name
from ..util import icon_path


class AddUserVariableDialog(QtWidgets.QDialog):

    formStatusChanged = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._main_window = parent

        self._field_status = {
            'title' : False,
            'name' : False
        }
        self._form_status = False
        self.resize(300, 100)
        self.setWindowTitle("Add user variable")
        self.setModal(True)
        self._load_icons()

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self._create_form_fields()
        self._compose_form_layout(layout)

        button_add_var = QtWidgets.QPushButton("Add variable")
        button_add_var.setEnabled(False)
        button_cancel = QtWidgets.QPushButton("Cancel")

        self.formStatusChanged.connect(button_add_var.setEnabled)

        layout.addWidget(button_cancel, 6, 0)
        layout.addWidget(button_add_var, 6, 1)

        button_add_var.clicked.connect(self.check_if_variable_is_unique)
        button_cancel.clicked.connect(self.reject)

    @property
    def table(self):
        return self._main_window.table

    def _load_icons(self):
        def build_icon(idx):
            return QtGui.QIcon(icon_path(f"lock_{idx}_icon.png"))

        self._icons = { status : build_icon(status) for status in ["closed", "open"] }

    def _create_form_fields(self):
        self.variable_title = QtWidgets.QLineEdit()

        self.variable_name = QtWidgets.QLineEdit()
        self.variable_name.setValidator(QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r'[a-zA-Z_]\w*')))

        self.variable_title.textChanged.connect(self._set_variable_name)

        self.variable_title.textChanged.connect(lambda x: self._update_form_status('title', len(x) > 0))
        self.variable_name.textChanged.connect(lambda x: self._update_form_status('name', self.variable_name.hasAcceptableInput() > 0))

        self.name_action = self.variable_name.addAction(self._icons["closed"], QtWidgets.QLineEdit.TrailingPosition)
        self.name_action.setToolTip("Set name manually")

        self.name_action.triggered.connect(self._set_field_status)
        self._set_field_status()

        self._setup_types_widgets()

        self.variable_before = QtWidgets.QComboBox()
        columns = self._main_window.table_view.get_movable_columns()
        self.variable_before.addItems(list(columns.keys()) + ['<end>'])
        self.variable_before.setCurrentIndex(len(columns))

        self.variable_description = QtWidgets.QPlainTextEdit()

    def _setup_types_widgets(self):
        self.type_and_example = QtWidgets.QWidget()
        self.type_and_example.setLayout(QtWidgets.QHBoxLayout())
        self.type_and_example.layout().setContentsMargins(0, 0, 0, 0)

        self.variable_type = QtWidgets.QComboBox()

        self.variable_example = QtWidgets.QLabel()
        self.variable_example.setAlignment(self.variable_example.alignment() | QtCore.Qt.AlignRight)

        for ii, (kk, vv) in enumerate(value_types_by_name.items()):
            self.variable_type.addItem(kk)
            self.variable_type.setItemData(ii, vv.description, QtCore.Qt.ToolTipRole)

        self._set_dynamic_type_information(self.variable_type.currentText())

        self.variable_type.textHighlighted.connect(lambda x: self._set_dynamic_type_information(x))
        self.variable_type.currentTextChanged.connect(lambda x: self._set_dynamic_type_information(x))

        self.type_and_example.layout().addWidget(self.variable_type, stretch=1)
        self.type_and_example.layout().addWidget(self.variable_example, stretch=2)


    def _set_dynamic_type_information(self, current_type):

        label = self.variable_example
        format_text = lambda x: f"<span style='color: gray; font-size: 10px;'>{x}</span>"
        cur_type_class = value_types_by_name[current_type]
        type_examples = cur_type_class.examples
        joined_examples = ', '.join(type_examples)

        self.variable_type.setToolTip(cur_type_class.description)

        label_font = label.font()
        label_font.setPixelSize(10)
        metrics = QtGui.QFontMetrics(label_font)

        clipped_text = metrics.elidedText(joined_examples, QtCore.Qt.ElideRight, label.width() - 5)

        label.setText(format_text(clipped_text))
        label.setToolTip(f'<b>Examples of values for type {current_type}</b>: {format_text(joined_examples)}')

    def _compose_form_layout(self, layout):
        layout.addWidget(QtWidgets.QLabel("<b>Title</b>*"), 0, 0)
        layout.addWidget(self.variable_title, 0, 1)
        layout.addWidget(QtWidgets.QLabel("<b>Name</b>*"), 1, 0)
        layout.addWidget(self.variable_name, 1, 1)
        layout.addWidget(QtWidgets.QLabel("<b>Type</b>*"), 2, 0)
        layout.addWidget(self.type_and_example, 2, 1)
        layout.addWidget(QtWidgets.QLabel("Before"), 3, 0)
        layout.addWidget(self.variable_before, 3, 1)
        layout.addWidget(QtWidgets.QLabel("Description"), 4, 0, 1, 2)
        layout.addWidget(self.variable_description, 5, 0, 1, 2)

    def _set_variable_name(self, text):
        if self.variable_name.isReadOnly():
            self.variable_name.setText(self._clean_string(text).lower())

    def _set_field_status(self, checked=None):
        new_status = not self.variable_name.isReadOnly()
        self.variable_name.setReadOnly(new_status)
        self.name_action.setToolTip("Set name {}".format("manually" if new_status else "automatically"))
        self.name_action.setIcon(self._icons["closed" if new_status else "open"])
        self.variable_name.setStyleSheet("color: gray" if new_status else "")
        if new_status:
            self._set_variable_name(self.variable_title.text())

    def _clean_string(self, string):
        res = re.sub(r'\W+', '@', string, flags = re.A).strip('@')
        res = re.sub(r'^\d+', '', res)
        return res.replace('@', '_')

    def _update_form_status(self, name, is_ok):
        self._field_status[name] = is_ok
        self.update_form_status()

    def update_form_status(self):
        new_status = all(self._field_status.values())
        new_status_different = self._form_status != new_status
        if new_status_different:
            self._form_status = new_status
            self.formStatusChanged.emit(new_status)

    def check_if_variable_is_unique(self, x):
        error_type = []

        if self.table.has_column(self.variable_name.text()):
            error_type.append('<b>name</b>')

        if self.table.has_column(self.variable_title.text(), by_title=True):
            error_type.append('<b>title</b>')

        if len(error_type) > 0:

            dialog = QtWidgets.QMessageBox(self)
            dialog.setWindowTitle("Error adding variable")
            dialog.setText(
                "A variable with the same {} is already present.".format(' and '.join(error_type))
            )
            dialog.exec()

            return

        self._main_window.add_variable(
            name=self.variable_name.text(),
            title=self.variable_title.text(),
            variable_type=self.variable_type.currentText(),
            description=self.variable_description.toPlainText(),
            before=self.variable_before.currentIndex()
        )

        self.accept()

