import sys
import traceback
from enum import Enum
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from pyflakes.api import check as pyflakes_check
from pyflakes.reporter import Reporter
from PyQt5.Qsci import QsciCommand, QsciLexerPython, QsciScintilla
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from superqt.utils import signals_blocked

from ..backend.extract_data import get_context_file
from ..context import ContextFile
from ..ctxsupport.ctxrunner import extract_error_info
from .theme import Theme, ThemeManager, set_lexer_theme


class ContextTestResult(Enum):
    OK = 0
    WARNING = 1
    ERROR = 2


class ContextFileCheckerThread(QThread):
    # ContextTestResult, traceback, lineno, offset, checked_code
    check_result = pyqtSignal(object, str, int, int, str)

    def __init__(self, code, db_dir, context_python, context_getter, parent=None):
        super().__init__(parent)
        self.code = code
        self.db_dir = db_dir
        self.context_python = context_python

        # This is a hack to allow us to test throwing an exception when checking
        # the context file. It'll always be get_context_file() except when
        # replaced with a Mock by the tests.
        self.context_getter = context_getter

    def run(self):
        error_info = None

        # If a different environment is not specified, we can evaluate the
        # context file directly.
        if self.context_python is None:
            try:
                ContextFile.from_str(self.code)
            except:
                # Extract the error information
                error_info = extract_error_info(*sys.exc_info())

        # Otherwise, write it to a temporary file to evaluate it from another
        # process.
        else:
            with NamedTemporaryFile(prefix=".tmp_ctx", dir=self.db_dir) as ctx_file:
                ctx_path = Path(ctx_file.name)
                ctx_path.write_text(self.code)

                try:
                    ctx, error_info = self.context_getter(ctx_path, self.context_python)
                except:
                    # Not a failure to evalute the context file, but a failure
                    # to *attempt* to evaluate the context file (e.g. because of
                    # a missing dependency).
                    help_msg = "# This is a partial error, please check the terminal for the full error message and ask for help from DA or the DOC."
                    traceback_str = f"{help_msg}\n\n{traceback.format_exc()}"
                    self.check_result.emit(ContextTestResult.ERROR, traceback_str, 0, 0, self.code)
                    return

        if error_info is not None:
            stacktrace, lineno, offset = error_info
            self.check_result.emit(ContextTestResult.ERROR, stacktrace, lineno, offset, self.code)
            return

        # If that worked, try pyflakes
        out_buffer = StringIO()
        reporter = Reporter(out_buffer, out_buffer)
        pyflakes_check(self.code, "<ctx>", reporter)
        # Disgusting hack to avoid getting warnings for "var#foo", "meta#foo",
        # and "mymdc#foo" type annotations. This needs some tweaking to avoid
        # missing real errors.
        pyflakes_output = "\n".join([line for line in out_buffer.getvalue().split("\n")
                                     if not line.endswith("undefined name 'var'") \
                                     and not line.endswith("undefined name 'meta'") \
                                     and not line.endswith("undefined name 'mymdc'")])

        if len(pyflakes_output) > 0:
            res, info = ContextTestResult.WARNING, pyflakes_output
        else:
            res, info = ContextTestResult.OK, None
        self.check_result.emit(res, info, -1, -1, self.code)


class Editor(QsciScintilla):
    check_result = pyqtSignal(object, str, str)  # result, info, checked_code

    def __init__(self):
        super().__init__()

        # Set initial theme
        self.current_theme = Theme.LIGHT
        self._apply_theme()

        self.setAutoCompletionSource(QsciScintilla.AutoCompletionSource.AcsAll)
        self.setAutoCompletionThreshold(3)
        self.setIndentationsUseTabs(False)
        self.setTabWidth(4)
        self.setAutoIndent(True)
        self.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
        self.setCaretLineVisible(True)
        self.setMarginWidth(0, "0000")
        self.setMarginLineNumbers(0, True)

        # Set Ctrl + D to delete a line
        commands = self.standardCommands()
        ctrl_d = commands.boundTo(Qt.ControlModifier | Qt.Key_D)
        ctrl_d.setKey(0)

        line_del = commands.find(QsciCommand.LineDelete)
        line_del.setKey(Qt.ControlModifier | Qt.Key_D)

    def _apply_theme(self):
        """Apply the current theme to the editor."""
        # Store current text and position
        current_text = self.text()
        current_position = self.SendScintilla(QsciScintilla.SCI_GETCURRENTPOS)

        # Create a new lexer with the theme colors
        font = QFont("Monospace", pointSize=12)
        self._lexer = QsciLexerPython()
        self._lexer.setDefaultFont(font)
        self._lexer.setFont(font, QsciLexerPython.Comment)

        # Get colors from theme manager
        colors = ThemeManager.get_syntax_highlighting_colors(self.current_theme)

        # Apply editor colors
        self.setPaper(colors['background'])
        self.setColor(colors['text'])
        self.setCaretForegroundColor(colors['caret'])
        self.setCaretLineBackgroundColor(colors['caret_line'])
        self.setMarginsForegroundColor(colors['margin_fore'])
        self.setMarginsBackgroundColor(colors['margin_back'])
        self.setSelectionBackgroundColor(colors['selection_back'])
        self.setSelectionForegroundColor(colors['selection_fore'])
        self.setMatchedBraceBackgroundColor(colors['brace_back'])
        self.setMatchedBraceForegroundColor(colors['brace_fore'])
        self.setUnmatchedBraceBackgroundColor(colors['unbrace_back'])
        self.setUnmatchedBraceForegroundColor(colors['unbrace_fore'])

        # Python syntax highlighting colors
        set_lexer_theme(self._lexer, self.current_theme)

        # Apply the new lexer
        self.setLexer(None)  # Clear the old lexer
        self.setLexer(self._lexer)  # Set the new lexer

        # Restore text and position
        with signals_blocked(self):
            self.setText(current_text)
        self.SendScintilla(QsciScintilla.SCI_SETCURRENTPOS, current_position)
        self.SendScintilla(QsciScintilla.SCI_SETSEL, current_position, current_position)

    def update_theme(self, theme: Theme):
        """Update the editor theme."""
        self.current_theme = theme
        self._apply_theme()

    def launch_test_context(self, db):
        context_python = db.metameta.get("context_python")
        thread = ContextFileCheckerThread(self.text(), db.path.parent, context_python, get_context_file, parent=self)
        thread.check_result.connect(self.on_test_result)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def on_test_result(self, res, info, lineno, offset, checked_code):
        if res is ContextTestResult.ERROR:
            if lineno != -1:
                # The line numbers reported by Python are 1-indexed so we
                # decrement before passing them to scintilla.
                lineno -= 1

                self.ensureLineVisible(lineno)

                # If we're already at the line, don't move the cursor unnecessarily
                if lineno != self.getCursorPosition()[0]:
                    self.setCursorPosition(lineno, offset)

        self.check_result.emit(res, info, checked_code)
