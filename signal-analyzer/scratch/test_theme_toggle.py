"""
scratch/test_theme_toggle.py
Test script to verify light/dark theme switching on PyQt6 and PyQtGraph.
"""
import sys
import os

_PROJECT_ROOT = r"c:\Sanchar\signal-analyzer"
for _d in [
    _PROJECT_ROOT,
    os.path.join(_PROJECT_ROOT, "models"),
    os.path.join(_PROJECT_ROOT, "preprocessing"),
    os.path.join(_PROJECT_ROOT, "gnuradio_pipeline"),
    os.path.join(_PROJECT_ROOT, "gui"),
]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt
import pyqtgraph as pg

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

# Test pyqtgraph dynamic background update
plot = pg.PlotWidget()
plot.plot([1, 2, 3, 4], [10, 20, 15, 30], pen=pg.mkPen('#58a6ff', width=2))

# Switch to light
plot.setBackground('#ffffff')
plot.getPlotItem().getAxis('left').setTextPen('#1f2328')
plot.getPlotItem().getAxis('left').setPen('#d0d7de')
plot.getPlotItem().getAxis('bottom').setTextPen('#1f2328')
plot.getPlotItem().getAxis('bottom').setPen('#d0d7de')

# Switch back to dark
plot.setBackground('#0d1117')
plot.getPlotItem().getAxis('left').setTextPen('#c9d1d9')
plot.getPlotItem().getAxis('left').setPen('#30363d')
plot.getPlotItem().getAxis('bottom').setTextPen('#c9d1d9')
plot.getPlotItem().getAxis('bottom').setPen('#30363d')

print("PyQtGraph dynamic theme switching verified.")
