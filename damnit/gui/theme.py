from enum import Enum

from PyQt5.Qsci import QsciLexerPython
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette


class Theme(Enum):
    LIGHT = "light"
    DARK = "dark"

class ThemeManager:
    @staticmethod
    def get_theme_palette(theme: Theme) -> QPalette:
        palette = QPalette()
        
        if theme == Theme.DARK:
            # Dark theme colors
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(35, 35, 35))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, QColor(35, 35, 35))
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.black)
        else:
            # Light theme - use default system palette
            pass
        
        return palette

    @staticmethod
    def get_theme_stylesheet(theme: Theme) -> str:
        if theme == Theme.DARK:
            return """
            QToolTip { 
                color: #ffffff; 
                background-color: #2a2a2a; 
                border: 1px solid #767676; 
            }
            QTableView, QTreeView {
                background-color: #232323;
                alternate-background-color: #2a2a2a;
                color: #ffffff;
                gridline-color: #353535;
            }
            QHeaderView::section {
                background-color: #353535;
                color: #ffffff;
                padding: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #353535;
            }
            QTabBar::tab {
                background-color: #353535;
                color: #ffffff;
                padding: 8px 20px;
            }
            QTabBar::tab:selected {
                background-color: #454545;
            }
            QScrollBar {
                background-color: #2a2a2a;
            }
            QScrollBar::handle {
                background-color: #454545;
            }
            QPlainTextEdit, QTextEdit {
                background-color: #232323;
                color: #ffffff;
            }
            QLineEdit {
                background-color: #232323;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #353535;
                border-radius: 2px;
            }
            QPushButton {
                background-color: #353535;
                color: #ffffff;
                padding: 6px 12px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: #454545;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
            QComboBox {
                background-color: #353535;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #454545;
                border-radius: 2px;
            }
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
            }
            QMenu {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #353535;
            }
            QMenu::item:selected {
                background-color: #454545;
            }
            QToolBar {
                background-color: #353535;
                color: #ffffff;
            }
            QToolButton {
                background-color: #353535;
                color: #ffffff;
            }
            QToolButton:hover {
                background-color: #454545;
            }
            """
        else:
            return ""  # Use default system style for light theme

    @staticmethod
    def get_syntax_highlighting_colors(theme: Theme):
        """Get the Editor's syntax highlighting colors for the given theme
        """
        if theme == Theme.DARK:
            return {
                'background': QColor('#232323'),
                'text': QColor('white'),
                'caret': QColor('white'),
                'caret_line': QColor('#2a2a2a'),
                'margin_fore': QColor('#cccccc'),
                'margin_back': QColor('#2a2a2a'),
                'selection_back': QColor('#404040'),
                'selection_fore': QColor('white'),
                'brace_back': QColor('#404040'),
                'brace_fore': QColor('white'),
                'unbrace_back': QColor('#802020'),
                'unbrace_fore': QColor('white'),
                'keyword': QColor('#66d9ef'),
                'class_name': QColor('#a6e22e'),
                'operator': QColor('#f92672'),
                'function': QColor('#fd971f'),
                'comment': QColor('#75715e'),
                'string': QColor('#e6db74'),
                'number': QColor('#ae81ff'),
            }
        else:
            return {
                'background': QColor('white'),
                'text': QColor('black'),
                'caret': QColor('black'),
                'caret_line': QColor('lightgray'),
                'margin_fore': QColor('black'),
                'margin_back': QColor('white'),
                'selection_back': QColor('#c0c0c0'),
                'selection_fore': QColor('black'),
                'brace_back': QColor('#c0c0c0'),
                'brace_fore': QColor('black'),
                'unbrace_back': QColor('#ff8080'),
                'unbrace_fore': QColor('black'),
                'keyword': QColor('#0000ff'),
                'class_name': QColor('#007f7f'),
                'operator': QColor('#7f0000'),
                'function': QColor('#007f00'),
                'comment': QColor('#7f7f7f'),
                'string': QColor('#7f007f'),
                'number': QColor('#007f7f'),
            }


def set_lexer_theme(lexer, theme: Theme):
    colors = ThemeManager.get_syntax_highlighting_colors(theme)
    lexer.setPaper(colors['background'])
    lexer.setColor(colors['text'])
    lexer.setDefaultPaper(colors['background'])
    lexer.setDefaultColor(colors['text'])
    lexer.setColor(colors['keyword'], QsciLexerPython.Keyword)
    lexer.setColor(colors['class_name'], QsciLexerPython.ClassName)
    lexer.setColor(colors['operator'], QsciLexerPython.Operator)
    lexer.setColor(colors['function'], QsciLexerPython.FunctionMethodName)
    lexer.setColor(colors['comment'], QsciLexerPython.Comment)
    lexer.setColor(colors['string'], QsciLexerPython.DoubleQuotedString)
    lexer.setColor(colors['string'], QsciLexerPython.SingleQuotedString)
    lexer.setColor(colors['number'], QsciLexerPython.Number)
