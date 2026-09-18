"""
main.py — SIH Signal Analyzer entry point.

Run from the project root with the venv activated:
    python main.py

The Qt event loop starts here. All analysis runs in a background QThread
inside MainWindow so the GUI never freezes during file loading or inference.
"""

import sys
import os

# Ensure project sub-packages are importable from wherever the user runs this
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _subdir in ["models", "preprocessing", "gnuradio_pipeline", "correlation", "gui"]:
    _p = os.path.join(_ROOT, _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Also insert root itself so `from gui.main_window import ...` resolves
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
import pyqtgraph as pg

# pyqtgraph global dark config (must be set before any widget is created)
pg.setConfigOption('background', '#0d1117')
pg.setConfigOption('foreground', '#e6edf3')
pg.setConfigOption('antialias', True)

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SIH Signal Analyzer")
    app.setOrganizationName("SIH")

    # Set app-wide font (Inter if available, otherwise Segoe UI)
    for font_name in ("Inter", "Segoe UI", "Arial"):
        f = QFont(font_name, 9)
        if f.exactMatch() or font_name in ("Segoe UI", "Arial"):
            app.setFont(f)
            break

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
