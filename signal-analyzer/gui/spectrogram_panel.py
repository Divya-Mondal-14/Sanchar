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
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QFont
from gui.theme import apply_plot_theme, get_info_box_style, format_info_box_html


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
        self._theme: str = "dark"
        self._cached_psd_rows: list[tuple[str, str]] | None = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setHandleWidth(4)
        self._splitter.setStyleSheet("QSplitter::handle { background: #21262d; }")
        splitter = self._splitter

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

        # ── PSD Information Box (overlay in top-right of plot) ────────
        self._psd_info_box = QLabel(self._psd_plot)
        self._psd_info_box.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._psd_info_box.setStyleSheet("""
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
        self._psd_info_box.hide()
        self._psd_plot.installEventFilter(self)

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
    # Event filter for repositioning overlay
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj == self._psd_plot and event.type() == QEvent.Type.Resize:
            self._reposition_psd_info_box()
        return super().eventFilter(obj, event)

    def _reposition_psd_info_box(self):
        if hasattr(self, "_psd_info_box") and self._psd_info_box.isVisible():
            self._psd_info_box.adjustSize()
            w = self._psd_info_box.width()
            h = self._psd_info_box.height()
            x = max(10, self._psd_plot.width() - w - 10)
            y = 10
            self._psd_info_box.move(x, y)

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    def set_data(self, spectral_dict: dict):
        self._spectral = spectral_dict
        self._refresh()

    def clear(self):
        self._spectral = None
        self._cached_psd_rows = None
        self._img_item.clear()
        self._psd_curve.setData([], [])
        self._psd_info_box.hide()
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
        obw       = self._spectral.get("occupied_bandwidth_hz")
        offset    = self._spectral.get("center_freq_offset_hz")

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
            if obw is not None and obw > 0:
                half = obw / 2.0
                off_val = offset if offset is not None else 0.0
                lo = off_val - half
                hi = off_val + half
                self._obw_lo.setPos(lo)
                self._obw_hi.setPos(hi)
                self._obw_lo.setVisible(True)
                self._obw_hi.setVisible(True)
            else:
                self._obw_lo.setVisible(False)
                self._obw_hi.setVisible(False)

            # --- PSD Information Box Content ---
            if len(psd_db) > 0 and len(freqs) > 0:
                p_idx = int(np.argmax(psd_db))
                peak_psd_str = f"{float(psd_db[p_idx]):.1f} dB"
                peak_f = float(freqs[p_idx])
                if abs(peak_f) >= 1e6:
                    peak_freq_str = f"{peak_f / 1e6:+.2f} MHz" if peak_f < 0 else f"{peak_f / 1e6:.2f} MHz"
                elif abs(peak_f) >= 1e3:
                    peak_freq_str = f"{peak_f / 1e3:+.2f} kHz" if peak_f < 0 else f"{peak_f / 1e3:.2f} kHz"
                else:
                    peak_freq_str = f"{peak_f:+.2f} Hz" if peak_f < 0 else f"{peak_f:.2f} Hz"

                noise_floor_str = f"{float(np.percentile(psd_db, 10)):.1f} dB"
            else:
                peak_psd_str = "N/A"
                peak_freq_str = "N/A"
                noise_floor_str = "N/A"

            if offset is not None:
                if abs(offset) >= 1e6:
                    cfo_str = f"{offset / 1e6:+.2f} MHz"
                elif abs(offset) >= 1e3:
                    cfo_str = f"{offset / 1e3:+.2f} kHz"
                else:
                    cfo_str = f"{offset:+.2f} Hz"
            else:
                cfo_str = "N/A"

            if obw is not None and obw > 0:
                if obw >= 1e6:
                    obw_str = f"{obw / 1e6:.2f} MHz"
                elif obw >= 1e3:
                    obw_str = f"{obw / 1e3:.2f} kHz"
                else:
                    obw_str = f"{obw:.2f} Hz"
            else:
                obw_str = "N/A"

            self._cached_psd_rows = [
                ("Center Offset", cfo_str),
                ("Occupied BW", obw_str),
                ("Peak Frequency", peak_freq_str),
                ("Peak PSD", peak_psd_str),
                ("Noise Floor", noise_floor_str),
            ]
            self._psd_info_box.setText(
                format_info_box_html("PSD INFORMATION", self._cached_psd_rows, self._theme)
            )
            self._psd_info_box.show()
            self._reposition_psd_info_box()

        obw_val = obw if obw is not None else 0
        off_val = offset if offset is not None else 0
        self._info.setText(
            f"OBW: {obw_val / 1e3:.2f} kHz  |  "
            f"Center offset: {off_val / 1e3:.2f} kHz"
        )

    # ------------------------------------------------------------------
    # Theme Support
    # ------------------------------------------------------------------

    def set_theme(self, theme: str = "dark"):
        """Dynamically apply light or dark theme styling."""
        self._theme = theme
        apply_plot_theme(self._psd_plot, theme)
        apply_plot_theme(self._waterfall_plot, theme)
        self._waterfall_plot.showGrid(x=False, y=False)

        self._psd_info_box.setStyleSheet(get_info_box_style(theme))
        if self._cached_psd_rows and self._psd_info_box.isVisible():
            self._psd_info_box.setText(
                format_info_box_html("PSD INFORMATION", self._cached_psd_rows, theme)
            )
            self._reposition_psd_info_box()

        if theme == "light":
            self._splitter.setStyleSheet("QSplitter::handle { background: #d0d7de; }")
            self._info.setStyleSheet("""
                color: #1f2328;
                background-color: #ffffff;
                padding: 3px 8px;
                border-radius: 4px;
                border: 1px solid #d0d7de;
            """)
            self._psd_curve.setPen(pg.mkPen(color='#0969da', width=1.5))
            self._psd_curve.setBrush(pg.mkBrush(9, 105, 218, 30))
            self._obw_lo.setPen(pg.mkPen('#d97706', width=1.5, style=Qt.PenStyle.DashLine))
            self._obw_hi.setPen(pg.mkPen('#d97706', width=1.5, style=Qt.PenStyle.DashLine))
            self._psd_plot.getPlotItem().setLabel('left', 'PSD (dB)', color='#24292f')
            self._psd_plot.getPlotItem().setLabel('bottom', 'Freq (Hz)', color='#24292f')
            self._waterfall_plot.getPlotItem().setLabel('left', 'Freq (Hz)', color='#24292f')
            self._waterfall_plot.getPlotItem().setLabel('bottom', 'Time (s)', color='#24292f')
        else:
            self._splitter.setStyleSheet("QSplitter::handle { background: #21262d; }")
            self._info.setStyleSheet("""
                color: #c9d1d9;
                background-color: #161b22;
                padding: 3px 8px;
                border-radius: 4px;
                border: 1px solid #30363d;
            """)
            self._psd_curve.setPen(pg.mkPen(color='#58a6ff', width=1.5))
            self._psd_curve.setBrush(pg.mkBrush(88, 166, 255, 30))
            self._obw_lo.setPen(pg.mkPen('#f78166', width=1.5, style=Qt.PenStyle.DashLine))
            self._obw_hi.setPen(pg.mkPen('#f78166', width=1.5, style=Qt.PenStyle.DashLine))
            self._psd_plot.getPlotItem().setLabel('left', 'PSD (dB)', color='#c9d1d9')
            self._psd_plot.getPlotItem().setLabel('bottom', 'Freq (Hz)', color='#c9d1d9')
            self._waterfall_plot.getPlotItem().setLabel('left', 'Freq (Hz)', color='#c9d1d9')
            self._waterfall_plot.getPlotItem().setLabel('bottom', 'Time (s)', color='#c9d1d9')


