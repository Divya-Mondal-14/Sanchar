"""
gui/constellation_panel.py

PyQt6 + pyqtgraph IQ constellation panel for the SIH signal analyzer.

Displays a scatter plot of I vs Q samples with ideal constellation points
overlaid for the detected modulation type.

Public API
----------
    ConstellationPanel(QWidget)
        .set_data(iq_samples, mod_class)
            iq_samples : np.ndarray shape (N, 2) float32 [I, Q]
                         OR complex64 array of length N
            mod_class  : str — 'BPSK', 'QPSK', or 'GMSK'
        .clear()
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QFont
from gui.theme import apply_plot_theme, get_info_box_style, format_info_box_html


# Ideal reference constellations (unit-power normalised, I=x, Q=y)
_IDEAL_CONSTELLATIONS: dict[str, list[tuple[float, float]]] = {
    "BPSK": [(-1.0, 0.0), (1.0, 0.0)],
    "QPSK": [
        ( 0.707,  0.707),
        (-0.707,  0.707),
        (-0.707, -0.707),
        ( 0.707, -0.707),
    ],
    "GMSK": [(1.0, 0.0), (-1.0, 0.0)],   # GMSK is effectively BPSK-like at decision instants
}

_MAX_SCATTER_POINTS = 4096   # cap for GPU/CPU scatter budget


class ConstellationPanel(QWidget):
    """IQ scatter plot with ideal constellation overlay."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme: str = "dark"
        self._cached_const_rows: list[tuple[str, str]] | None = None
        self._axis_lines: list[pg.InfiniteLine] = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Header row
        self._header = QLabel("Constellation")
        self._header.setFont(QFont("Inter", 10, QFont.Weight.DemiBold))
        self._header.setStyleSheet("color: #e6edf3; padding: 2px 0;")
        root.addWidget(self._header)

        # Plot
        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._plot.showGrid(x=True, y=True, alpha=0.12)
        self._plot.setMinimumHeight(200)
        self._plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._plot.getPlotItem().setLabel('bottom', 'I')
        self._plot.getPlotItem().setLabel('left', 'Q')

        # Axis lines
        self._axis_lines.clear()
        for angle, pos in [(90, 0), (0, 0)]:
            line = pg.InfiniteLine(
                angle=angle, pos=pos, movable=False,
                pen=pg.mkPen('#30363d', width=1),
            )
            self._axis_lines.append(line)
            self._plot.addItem(line)

        # Received samples scatter
        self._scatter = pg.ScatterPlotItem(
            size=3, pen=None,
            brush=pg.mkBrush(88, 166, 255, 80),
        )
        self._plot.addItem(self._scatter)

        # Ideal points
        self._ideal_scatter = pg.ScatterPlotItem(
            size=16, pen=pg.mkPen('#f78166', width=2),
            brush=pg.mkBrush(247, 129, 102, 60),
            symbol='o',
        )
        self._plot.addItem(self._ideal_scatter)

        # ── Constellation Information Box (overlay in top-right of plot) ──
        self._const_info_box = QLabel(self._plot)
        self._const_info_box.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._const_info_box.setStyleSheet("""
            QLabel {
                background-color: rgba(22, 27, 34, 215);
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 4px 6px;
                color: #c9d1d9;
                font-family: 'Consolas', 'Segoe UI', monospace;
                font-size: 7.5pt;
            }
        """)
        self._const_info_box.hide()
        self._plot.installEventFilter(self)

        root.addWidget(self._plot)

        # Status
        self._status = QLabel("No data loaded.")
        self._status.setFont(QFont("Inter", 8))
        self._status.setStyleSheet("color: #484f58; padding: 2px 4px;")
        root.addWidget(self._status)

    # ------------------------------------------------------------------
    # Event filter for repositioning overlay
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj == self._plot and event.type() == QEvent.Type.Resize:
            self._reposition_const_info_box()
        return super().eventFilter(obj, event)

    def _reposition_const_info_box(self):
        if hasattr(self, "_const_info_box") and self._const_info_box.isVisible():
            self._const_info_box.adjustSize()
            w = self._const_info_box.width()
            h = self._const_info_box.height()
            x = max(10, self._plot.width() - w - 10)
            y = 10
            self._const_info_box.move(x, y)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_data(self, iq_samples: np.ndarray, mod_class: str = ""):
        """
        Parameters
        ----------
        iq_samples : np.ndarray
            Either shape (N, 2) float32 [I col, Q col]
            or shape (N,) complex64/128
        mod_class  : str
            'BPSK', 'QPSK', 'GMSK', or '' for no ideal overlay
        """
        # Normalise to (N, 2) float32
        if np.iscomplexobj(iq_samples):
            iq = np.stack([iq_samples.real, iq_samples.imag], axis=-1).astype(np.float32)
        else:
            iq = np.asarray(iq_samples, dtype=np.float32)
            if iq.ndim != 2 or iq.shape[1] != 2:
                raise ValueError(f"Expected (N, 2) or complex array, got {iq.shape}")

        # Downsample for display
        n = iq.shape[0]
        if n > _MAX_SCATTER_POINTS:
            step = int(np.ceil(n / _MAX_SCATTER_POINTS))
            iq = iq[::step]

        # Normalise power so scatter fills the unit circle
        rms = float(np.sqrt(np.mean(iq[:, 0] ** 2 + iq[:, 1] ** 2)))
        if rms > 1e-9:
            iq = iq / rms

        # Scatter plot
        self._scatter.setData(iq[:, 0].tolist(), iq[:, 1].tolist())

        # Ideal constellation overlay
        ideal = _IDEAL_CONSTELLATIONS.get(mod_class.upper(), [])
        if ideal:
            ix, iy = zip(*ideal)
            self._ideal_scatter.setData(list(ix), list(iy))
            self._ideal_scatter.setVisible(True)
        else:
            self._ideal_scatter.setData([], [])
            self._ideal_scatter.setVisible(False)

        # --- Constellation Information Box Content ---
        if len(iq) > 0:
            i_mean_str = f"{float(np.mean(iq[:, 0])):+.3f}"
            q_mean_str = f"{float(np.mean(iq[:, 1])):+.3f}"
            i_rms_str = f"{float(np.sqrt(np.mean(iq[:, 0] ** 2))):.3f}"
            q_rms_str = f"{float(np.sqrt(np.mean(iq[:, 1] ** 2))):.3f}"
            avg_mag_str = f"{float(np.mean(np.sqrt(iq[:, 0] ** 2 + iq[:, 1] ** 2))):.3f}"
            n_pts_str = f"{len(iq):,}"
        else:
            i_mean_str = "N/A"
            q_mean_str = "N/A"
            i_rms_str = "N/A"
            q_rms_str = "N/A"
            avg_mag_str = "N/A"
            n_pts_str = "0"

        self._cached_const_rows = [
            ("Modulation", mod_class or 'N/A'),
            ("I/Q Points", n_pts_str),
            ("I Mean", i_mean_str),
            ("Q Mean", q_mean_str),
            ("I RMS", i_rms_str),
            ("Q RMS", q_rms_str),
            ("Avg |IQ|", avg_mag_str),
        ]
        self._const_info_box.setText(
            format_info_box_html("CONSTELLATION INFO", self._cached_const_rows, self._theme)
        )
        self._const_info_box.show()
        self._reposition_const_info_box()

        self._status.setText(
            f"{n:,} samples  |  mod: {mod_class or 'unknown'}  |  "
            f"displayed: {iq.shape[0]:,}"
        )

    def clear(self):
        self._cached_const_rows = None
        self._scatter.setData([], [])
        self._ideal_scatter.setData([], [])
        self._const_info_box.hide()
        self._status.setText("No data loaded.")

    # ------------------------------------------------------------------
    # Theme Support
    # ------------------------------------------------------------------

    def set_theme(self, theme: str = "dark"):
        """Dynamically apply light or dark theme styling."""
        self._theme = theme
        apply_plot_theme(self._plot, theme)

        self._const_info_box.setStyleSheet(get_info_box_style(theme))
        if self._cached_const_rows and self._const_info_box.isVisible():
            self._const_info_box.setText(
                format_info_box_html("CONSTELLATION INFO", self._cached_const_rows, theme)
            )
            self._reposition_const_info_box()

        if theme == "light":
            self._header.setStyleSheet("color: #1f2328; padding: 2px 0;")
            self._status.setStyleSheet("color: #656d76; padding: 2px 4px;")
            self._plot.getPlotItem().setLabel('bottom', 'I', color='#24292f')
            self._plot.getPlotItem().setLabel('left', 'Q', color='#24292f')
            for line in self._axis_lines:
                line.setPen(pg.mkPen('#d0d7de', width=1))
            self._scatter.setBrush(pg.mkBrush(9, 105, 218, 110))
            self._ideal_scatter.setPen(pg.mkPen('#cf222e', width=2))
            self._ideal_scatter.setBrush(pg.mkBrush(207, 34, 46, 60))
        else:
            self._header.setStyleSheet("color: #e6edf3; padding: 2px 0;")
            self._status.setStyleSheet("color: #484f58; padding: 2px 4px;")
            self._plot.getPlotItem().setLabel('bottom', 'I', color='#c9d1d9')
            self._plot.getPlotItem().setLabel('left', 'Q', color='#c9d1d9')
            for line in self._axis_lines:
                line.setPen(pg.mkPen('#30363d', width=1))
            self._scatter.setBrush(pg.mkBrush(88, 166, 255, 80))
            self._ideal_scatter.setPen(pg.mkPen('#f78166', width=2))
            self._ideal_scatter.setBrush(pg.mkBrush(247, 129, 102, 60))

