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
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


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
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Header row
        header = QLabel("Constellation")
        header.setFont(QFont("Inter", 10, QFont.Weight.DemiBold))
        header.setStyleSheet("color: #e6edf3; padding: 2px 0;")
        root.addWidget(header)

        # Plot
        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._plot.showGrid(x=True, y=True, alpha=0.12)
        self._plot.setMinimumHeight(200)
        self._plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._plot.getPlotItem().setLabel('bottom', 'I')
        self._plot.getPlotItem().setLabel('left', 'Q')

        # Axis lines
        for angle, pos in [(90, 0), (0, 0)]:
            line = pg.InfiniteLine(
                angle=angle, pos=pos, movable=False,
                pen=pg.mkPen('#30363d', width=1),
            )
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

        root.addWidget(self._plot)

        # Status
        self._status = QLabel("No data loaded.")
        self._status.setFont(QFont("Inter", 8))
        self._status.setStyleSheet("color: #484f58; padding: 2px 4px;")
        root.addWidget(self._status)

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

        self._status.setText(
            f"{n:,} samples  |  mod: {mod_class or 'unknown'}  |  "
            f"displayed: {iq.shape[0]:,}"
        )

    def clear(self):
        self._scatter.setData([], [])
        self._ideal_scatter.setData([], [])
        self._status.setText("No data loaded.")
