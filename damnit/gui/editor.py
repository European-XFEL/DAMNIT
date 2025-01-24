import sys
import traceback
from enum import Enum
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.Qsci import QsciScintilla, QsciLexerPython, QsciCommand

from pyflakes.reporter import Reporter
from pyflakes.api import check as pyflakes_check

from ..backend.extract_data import get_context_file
from ..ctxsupport.ctxrunner import extract_error_info
from ..context import ContextFile
from .theme import Theme

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

        font = QFont("Monospace", pointSize=12)
        self._lexer = QsciLexerPython()
        self._lexer.setDefaultFont(font)
        self._lexer.setFont(font, QsciLexerPython.Comment)
        self.setLexer(self._lexer)

        self.setAutoCompletionSource(QsciScintilla.AutoCompletionSource.AcsAll)
        self.setAutoCompletionThreshold(3)
        self.setIndentationsUseTabs(False)
        self.setTabWidth(4)
        self.setAutoIndent(True)
        self.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
        self.setCaretLineVisible(True)
        self.setCaretLineBackgroundColor(QColor('lightgray'))
        self.setMarginWidth(0, "0000")
        self.setMarginLineNumbers(0, True)

        # Set Ctrl + D to delete a line
        commands = self.standardCommands()
        ctrl_d = commands.boundTo(Qt.ControlModifier | Qt.Key_D)
        ctrl_d.setKey(0)

        line_del = commands.find(QsciCommand.LineDelete)
        line_del.setKey(Qt.ControlModifier | Qt.Key_D)

        # Set initial theme
        self.current_theme = Theme.LIGHT
        self._apply_theme()

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

        if self.current_theme == Theme.DARK:
            # Dark theme colors
            self.setPaper(QColor('#232323'))  # Background
            self.setColor(QColor('white'))    # Default text
            self.setCaretForegroundColor(QColor('white'))
            self.setCaretLineBackgroundColor(QColor('#2a2a2a'))
            self.setMarginsForegroundColor(QColor('#cccccc'))  # Line numbers
            self.setMarginsBackgroundColor(QColor('#2a2a2a'))
            self.setSelectionBackgroundColor(QColor('#404040'))
            self.setSelectionForegroundColor(QColor('white'))
            self.setMatchedBraceBackgroundColor(QColor('#404040'))
            self.setMatchedBraceForegroundColor(QColor('white'))
            self.setUnmatchedBraceBackgroundColor(QColor('#802020'))
            self.setUnmatchedBraceForegroundColor(QColor('white'))
            
            # Python syntax highlighting colors for dark theme
            self._lexer.setDefaultPaper(QColor('#232323'))
            self._lexer.setDefaultColor(QColor('white'))
            self._lexer.setColor(QColor('#66d9ef'), QsciLexerPython.Keyword)
            self._lexer.setColor(QColor('#a6e22e'), QsciLexerPython.ClassName)
            self._lexer.setColor(QColor('#f92672'), QsciLexerPython.Operator)
            self._lexer.setColor(QColor('#fd971f'), QsciLexerPython.FunctionMethodName)
            self._lexer.setColor(QColor('#75715e'), QsciLexerPython.Comment)
            self._lexer.setColor(QColor('#e6db74'), QsciLexerPython.DoubleQuotedString)
            self._lexer.setColor(QColor('#e6db74'), QsciLexerPython.SingleQuotedString)
            self._lexer.setColor(QColor('#ae81ff'), QsciLexerPython.Number)
            
            # Set paper (background) for all styles
            for style in range(128):  # QScintilla uses style numbers 0-127
                self._lexer.setPaper(QColor('#232323'), style)
        else:
            # Light theme colors (default)
            self.setPaper(QColor('white'))
            self.setColor(QColor('black'))
            self.setCaretForegroundColor(QColor('black'))
            self.setCaretLineBackgroundColor(QColor('lightgray'))
            self.setMarginsForegroundColor(QColor('black'))
            self.setMarginsBackgroundColor(QColor('white'))
            self.setSelectionBackgroundColor(QColor('#c0c0c0'))
            self.setSelectionForegroundColor(QColor('black'))
            self.setMatchedBraceBackgroundColor(QColor('#c0c0c0'))
            self.setMatchedBraceForegroundColor(QColor('black'))
            self.setUnmatchedBraceBackgroundColor(QColor('#ff8080'))
            self.setUnmatchedBraceForegroundColor(QColor('black'))
            
            # Python syntax highlighting colors for light theme
            self._lexer.setDefaultPaper(QColor('white'))
            self._lexer.setDefaultColor(QColor('black'))
            self._lexer.setColor(QColor('#0000ff'), QsciLexerPython.Keyword)
            self._lexer.setColor(QColor('#007f7f'), QsciLexerPython.ClassName)
            self._lexer.setColor(QColor('#7f0000'), QsciLexerPython.Operator)
            self._lexer.setColor(QColor('#007f00'), QsciLexerPython.FunctionMethodName)
            self._lexer.setColor(QColor('#7f7f7f'), QsciLexerPython.Comment)
            self._lexer.setColor(QColor('#7f007f'), QsciLexerPython.DoubleQuotedString)
            self._lexer.setColor(QColor('#7f007f'), QsciLexerPython.SingleQuotedString)
            self._lexer.setColor(QColor('#007f7f'), QsciLexerPython.Number)
            
            # Set paper (background) for all styles
            for style in range(128):  # QScintilla uses style numbers 0-127
                self._lexer.setPaper(QColor('white'), style)

        # Apply the new lexer
        self.setLexer(None)  # Clear the old lexer
        self.setLexer(self._lexer)  # Set the new lexer
        
        # Restore text and position
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
