"""
gui/spectrogram_panel.py

PyQt6 + pyqtgraph spectrogram / waterfall panel for the SIH signal analyzer.

Shows:
  - Top: waterfall spectrogram (ImageItem with a thermal color map)
  - Bottom: PSD line plot (Welch's method output from spectral_features.py)

Public API
----------
    SpectrogramPanel(QWidget)
        .set_data(spectral_dict)   — accepts the dict from spectral_features.analyze()
        .clear()                   — reset to empty state
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph import ColorMap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


# ---------------------------------------------------------------------------
# Viridis-like thermal color map (12-stop, no matplotlib dependency)
# ---------------------------------------------------------------------------

_VIRIDIS_STOPS = np.array([
    [0.267, 0.005, 0.329],   # deep purple
    [0.283, 0.141, 0.458],
    [0.253, 0.265, 0.530],
    [0.207, 0.371, 0.553],
    [0.164, 0.471, 0.558],
    [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518],
    [0.267, 0.749, 0.441],
    [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.150],
    [0.993, 0.906, 0.144],   # bright yellow
    [0.993, 0.906, 0.144],
], dtype=np.float32)

_VIRIDIS_POS = np.linspace(0.0, 1.0, len(_VIRIDIS_STOPS))

_CMAP = ColorMap(
    pos=_VIRIDIS_POS,
    color=(_VIRIDIS_STOPS * 255).astype(np.uint8),
)


class SpectrogramPanel(QWidget):
    """Spectrogram waterfall + PSD panel."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._spectral: dict | None = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background: #21262d; }")

        # ── Waterfall ────────────────────────────────────────────────
        self._waterfall_plot = pg.PlotWidget()
        self._waterfall_plot.setMinimumHeight(140)
        self._waterfall_plot.getPlotItem().setLabel('left', 'Freq (Hz)')
        self._waterfall_plot.getPlotItem().setLabel('bottom', 'Time (s)')
        self._waterfall_plot.showGrid(x=False, y=False)

        self._img_item = pg.ImageItem()
        self._img_item.setColorMap(_CMAP)
        self._waterfall_plot.addItem(self._img_item)

        # Color bar
        self._colorbar = pg.ColorBarItem(
            values=(-120, 0),
            colorMap=_CMAP,
            label='dB',
            interactive=False,
        )
        self._colorbar.setImageItem(self._img_item, insert_in=self._waterfall_plot.plotItem)

        splitter.addWidget(self._waterfall_plot)

        # ── PSD plot ─────────────────────────────────────────────────
        self._psd_plot = pg.PlotWidget()
        self._psd_plot.setMinimumHeight(100)
        self._psd_plot.getPlotItem().setLabel('left', 'PSD (dB)')
        self._psd_plot.getPlotItem().setLabel('bottom', 'Freq (Hz)')
        self._psd_plot.showGrid(x=True, y=True, alpha=0.15)

        self._psd_curve = self._psd_plot.plot(
            pen=pg.mkPen(color='#58a6ff', width=1.5),
            fillLevel=-200,
            brush=pg.mkBrush(88, 166, 255, 30),
        )

        # OBW marker lines (clean dashed lines framing occupied bandwidth without text overlap)
        self._obw_lo = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#f78166', width=1.5, style=Qt.PenStyle.DashLine),
        )
        self._obw_hi = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#f78166', width=1.5, style=Qt.PenStyle.DashLine),
        )
        self._psd_plot.addItem(self._obw_lo)
        self._psd_plot.addItem(self._obw_hi)

        splitter.addWidget(self._psd_plot)
        splitter.setSizes([220, 120])
        root.addWidget(splitter)

        # ── Info label ───────────────────────────────────────────────
        self._info = QLabel("No data loaded.")
        self._info.setFont(QFont("Inter", 8))
        self._info.setStyleSheet("""
            color: #c9d1d9;
            background-color: #161b22;
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid #30363d;
        """)
        root.addWidget(self._info)

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    def set_data(self, spectral_dict: dict):
        self._spectral = spectral_dict
        self._refresh()

    def clear(self):
        self._spectral = None
        self._img_item.clear()
        self._psd_curve.setData([], [])
        self._info.setText("No data loaded.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh(self):
        if self._spectral is None:
            return

        spec_db   = self._spectral.get("spec_db")
        spec_times = self._spectral.get("spec_times")
        spec_freqs = self._spectral.get("spec_freqs")
        freqs     = self._spectral.get("freqs")
        psd_db    = self._spectral.get("psd_db")
        obw       = self._spectral.get("occupied_bandwidth_hz", 0)
        offset    = self._spectral.get("center_freq_offset_hz", 0)

        # --- Waterfall ---
        if spec_db is not None and spec_times is not None and spec_freqs is not None:
            # spec_db shape: (n_freqs, n_times) — ImageItem expects (n_times, n_freqs)
            img = spec_db.T.copy()
            vmin, vmax = float(np.percentile(img, 2)), float(np.percentile(img, 98))
            self._img_item.setLevels([vmin, vmax])

            dt = float(spec_times[1] - spec_times[0]) if len(spec_times) > 1 else 1.0
            df = float(spec_freqs[1] - spec_freqs[0]) if len(spec_freqs) > 1 else 1.0
            t0 = float(spec_times[0])
            f0 = float(spec_freqs[0])
            self._img_item.setImage(img, autoLevels=False)
            self._img_item.resetTransform()
            tr = pg.QtGui.QTransform()
            tr.scale(dt, df)
            tr.translate(t0 / dt, f0 / df)
            self._img_item.setTransform(tr)

            self._colorbar.setLevels([vmin, vmax])

        # --- PSD ---
        if freqs is not None and psd_db is not None:
            self._psd_curve.setData(freqs.tolist(), psd_db.tolist())

            # OBW markers
            if obw > 0:
                half = obw / 2.0
                lo = offset - half
                hi = offset + half
                self._obw_lo.setPos(lo)
                self._obw_hi.setPos(hi)
                self._obw_lo.setVisible(True)
                self._obw_hi.setVisible(True)
            else:
                self._obw_lo.setVisible(False)
                self._obw_hi.setVisible(False)

        self._info.setText(
            f"OBW: {obw / 1e3:.2f} kHz  |  "
            f"Center offset: {offset / 1e3:.2f} kHz"
        )
