import sys
from enum import Enum
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QGuiApplication, QCursor
from PyQt5.Qsci import QsciScintilla, QsciLexerPython, QsciCommand

from pyflakes.reporter import Reporter
from pyflakes.api import check as pyflakes_check

from ..backend.extract_data import get_context_file
from ..ctxsupport.ctxrunner import extract_error_info
from ..context import ContextFile


class ContextTestResult(Enum):
    OK = 0
    WARNING = 1
    ERROR = 2

class Editor(QsciScintilla):
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

    def test_context(self, db, db_dir):
        """
        Check if the current context file is valid.

        Returns a tuple of (result, output_msg).
        """
        error_info = None
        context_python = db.metameta.get("context_python")

        # If a different environment is not specified, we can evaluate the
        # context file directly.
        if context_python is None:
            try:
                ContextFile.from_str(self.text())
            except:
                # Extract the error information
                error_info = extract_error_info(*sys.exc_info())

        # Otherwise, write it to a temporary file to evaluate it from another
        # process.
        else:
            with NamedTemporaryFile(prefix=".tmp_ctx", dir=db_dir) as ctx_file:
                ctx_path = Path(ctx_file.name)
                ctx_path.write_text(self.text())

                QGuiApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                ctx, error_info = get_context_file(ctx_path, context_python)
                QGuiApplication.restoreOverrideCursor()

        if error_info is not None:
            stacktrace, lineno, offset = error_info

            if lineno != -1:
                # The line numbers reported by Python are 1-indexed so we
                # decrement before passing them to scintilla.
                lineno -= 1

                self.ensureLineVisible(lineno)

                # If we're already at the line, don't move the cursor unnecessarily
                if lineno != self.getCursorPosition()[0]:
                    self.setCursorPosition(lineno, offset)

            return ContextTestResult.ERROR, stacktrace

        # If that worked, try pyflakes
        out_buffer = StringIO()
        reporter = Reporter(out_buffer, out_buffer)
        pyflakes_check(self.text(), "<ctx>", reporter)
        # Disgusting hack to avoid getting warnings for "var#foo", "meta#foo",
        # and "mymdc#foo" type annotations. This needs some tweaking to avoid
        # missing real errors.
        pyflakes_output = "\n".join([line for line in out_buffer.getvalue().split("\n")
                                     if not line.endswith("undefined name 'var'") \
                                     and not line.endswith("undefined name 'meta'") \
                                     and not line.endswith("undefined name 'mymdc'")])

        if len(pyflakes_output) > 0:
            return ContextTestResult.WARNING, pyflakes_output
        else:
            return ContextTestResult.OK, None
