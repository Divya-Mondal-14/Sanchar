"""
Centralized Theme definitions for Signal Analyzer.
Provides cohesive dark and light theme styles and helpers for widgets and plots.
"""

from __future__ import annotations
import pyqtgraph as pg
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtCore import Qt

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
QToolBar {
    background-color: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 4px;
    spacing: 6px;
}
QToolButton {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 9pt;
    min-width: 80px;
}
QToolButton:hover {
    background-color: #388bfd22;
    border-color: #388bfd;
    color: #58a6ff;
}
QToolButton:pressed {
    background-color: #388bfd44;
}
QTabWidget::pane {
    border: 1px solid #30363d;
    background-color: #0d1117;
    border-radius: 6px;
    top: -1px;
}
QTabBar::tab {
    background-color: #161b22;
    color: #8b949e;
    border: 1px solid #30363d;
    border-bottom: none;
    padding: 8px 18px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-size: 9pt;
    font-weight: 500;
}
QTabBar::tab:hover {
    background-color: #21262d;
    color: #c9d1d9;
}
QTabBar::tab:selected {
    background-color: #0d1117;
    color: #58a6ff;
    border-color: #30363d;
    border-bottom: 2px solid #58a6ff;
    font-weight: 600;
}
QPushButton {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 8pt;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #58a6ff;
    color: #58a6ff;
}
QPushButton:pressed {
    background-color: #1f6feb;
    color: #ffffff;
}
QStatusBar {
    background-color: #161b22;
    color: #8b949e;
    font-size: 8pt;
}
QProgressBar {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    text-align: center;
    color: #e6edf3;
    height: 14px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #388bfd, stop:1 #58a6ff);
    border-radius: 3px;
}
QSplitter::handle {
    background: #21262d;
}
QMenuBar {
    background-color: #161b22;
    color: #e6edf3;
    border-bottom: 1px solid #30363d;
}
QMenuBar::item:selected {
    background-color: #21262d;
}
QMenu {
    background-color: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
}
QMenu::item:selected {
    background-color: #21262d;
    color: #58a6ff;
}
"""

LIGHT_STYLE = """
QMainWindow, QWidget {
    background-color: #f6f8fa;
    color: #1f2328;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
QToolBar {
    background-color: #ffffff;
    border-bottom: 1px solid #d0d7de;
    padding: 4px;
    spacing: 6px;
}
QToolButton {
    background-color: #ffffff;
    color: #1f2328;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 9pt;
    min-width: 80px;
}
QToolButton:hover {
    background-color: #f3f4f6;
    border-color: #0969da;
    color: #0969da;
}
QToolButton:pressed {
    background-color: #e7ecf0;
}
QTabWidget::pane {
    border: 1px solid #d0d7de;
    background-color: #ffffff;
    border-radius: 6px;
    top: -1px;
}
QTabBar::tab {
    background-color: #f6f8fa;
    color: #656d76;
    border: 1px solid #d0d7de;
    border-bottom: none;
    padding: 8px 18px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-size: 9pt;
    font-weight: 500;
}
QTabBar::tab:hover {
    background-color: #f3f4f6;
    color: #1f2328;
}
QTabBar::tab:selected {
    background-color: #ffffff;
    color: #0969da;
    border-color: #d0d7de;
    border-bottom: 2px solid #0969da;
    font-weight: 600;
}
QPushButton {
    background-color: #ffffff;
    color: #1f2328;
    border: 1px solid #d0d7de;
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 8pt;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #f3f4f6;
    border-color: #0969da;
    color: #0969da;
}
QPushButton:pressed {
    background-color: #0969da;
    color: #ffffff;
}
QStatusBar {
    background-color: #ffffff;
    color: #656d76;
    font-size: 8pt;
    border-top: 1px solid #d0d7de;
}
QProgressBar {
    background-color: #eaeef2;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    text-align: center;
    color: #1f2328;
    height: 14px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0969da, stop:1 #218bff);
    border-radius: 3px;
}
QSplitter::handle {
    background: #d0d7de;
}
QMenuBar {
    background-color: #ffffff;
    color: #1f2328;
    border-bottom: 1px solid #d0d7de;
}
QMenuBar::item:selected {
    background-color: #f3f4f6;
}
QMenu {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    color: #1f2328;
}
QMenu::item:selected {
    background-color: #f3f4f6;
    color: #0969da;
}
"""

def apply_plot_theme(plot_widget: pg.PlotWidget, theme: str = "dark"):
    """
    Applies background color, axis pens, text pens, and grid alpha
    to an existing pg.PlotWidget without recreating it or affecting plot data.
    """
    if theme == "light":
        bg_color = "#ffffff"
        axis_pen = pg.mkPen("#57606a", width=1)
        text_pen = pg.mkPen("#24292f")
        grid_alpha = 0.15
    else:
        bg_color = "#0d1117"
        axis_pen = pg.mkPen("#30363d", width=1)
        text_pen = pg.mkPen("#c9d1d9")
        grid_alpha = 0.20

    plot_widget.setBackground(bg_color)
    pi = plot_widget.getPlotItem()
    for ax_name in ['left', 'bottom', 'right', 'top']:
        ax = pi.getAxis(ax_name)
        if ax is not None:
            ax.setPen(axis_pen)
            ax.setTextPen(text_pen)
    pi.showGrid(x=True, y=True, alpha=grid_alpha)


def get_info_box_style(theme: str = "dark") -> str:
    """Returns CSS for the overlay information boxes (PSD, Constellation)."""
    if theme == "light":
        return """
            QLabel {
                background-color: rgba(255, 255, 255, 235);
                border: 1px solid #d0d7de;
                border-radius: 4px;
                padding: 4px 6px;
                color: #1f2328;
                font-family: 'Consolas', 'Segoe UI', monospace;
                font-size: 7.5pt;
            }
        """
    return """
        QLabel {
            background-color: rgba(22, 27, 34, 215);
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 4px 6px;
            color: #c9d1d9;
            font-family: 'Consolas', 'Segoe UI', monospace;
            font-size: 7.5pt;
        }
    """


def format_info_box_html(title: str, rows: list[tuple[str, str]], theme: str = "dark") -> str:
    """Formats HTML content for overlay information boxes with theme-aware colors."""
    if theme == "light":
        title_color = "#0969da"
        lbl_color = "#656d76"
        val_color = "#1f2328"
        border_color = "#d0d7de"
    else:
        title_color = "#58a6ff"
        lbl_color = "#8b949e"
        val_color = "#f0f6fc"
        border_color = "#30363d"

    table_rows = "".join(
        f"<tr><td style=\"color: {lbl_color}; padding-right: 8px; font-size: 7.5pt;\">{k}</td>"
        f"<td style=\"color: {val_color}; text-align: right; font-weight: bold; font-size: 7.5pt;\">{v}</td></tr>"
        for k, v in rows
    )
    return f"""
    <div style="font-weight: bold; color: {title_color}; font-size: 7.5pt; border-bottom: 1px solid {border_color}; padding-bottom: 1px; margin-bottom: 2px; letter-spacing: 0.5px;">
        {title}
    </div>
    <table style="border-collapse: collapse; margin-top: 1px;">
        {table_rows}
    </table>
    """

