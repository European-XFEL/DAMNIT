from enum import Enum
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor

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
                border: none;
            }
            QTabWidget::pane {
                border: 1px solid #353535;
            }
            QTabBar::tab {
                background-color: #353535;
                color: #ffffff;
                padding: 8px 20px;
                border: none;
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
            QScrollBar::add-line, QScrollBar::sub-line {
                background: none;
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
                border: none;
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
            QComboBox::drop-down {
                border: none;
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
            """
        else:
            return ""  # Use default system style for light theme
