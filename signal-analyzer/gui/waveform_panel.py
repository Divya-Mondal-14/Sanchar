"""
gui/waveform_panel.py

PyQt6 + pyqtgraph ultra-fast, zero-lag time-domain waveform panel.

Features:
- Single full-height high-performance PlotWidget using native 'peak' downsampling.
- Multi-channel support: I+Q Overlay, In-Phase (I), Quadrature (Q), Amplitude, Phase, Inst. Freq.
- Butter-smooth panning and zooming (left-drag to pan, mouse wheel or right-drag box to zoom).
- Quick navigation buttons: Fit Full Signal, Zoom to Wave Cycles (first 500 samples), Zoom In, Zoom Out.
- Ultra-lightweight crosshair cursor with real-time O(1) telemetry readout.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# Global dark pyqtgraph configuration
pg.setConfigOption('background', '#0d1117')
pg.setConfigOption('foreground', '#e6edf3')
pg.setConfigOption('antialias', True)


class WaveformPanel(QWidget):
    """Clean, high-performance time-domain waveform viewer."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._waveform: dict | None = None
        self._x_data: np.ndarray = np.array([], dtype=np.float32)
        self._i_data: np.ndarray = np.array([], dtype=np.float32)
        self._q_data: np.ndarray = np.array([], dtype=np.float32)

        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── 1. Top Control & Navigation Bar ─────────────────────────────
        control_bar = QHBoxLayout()
        control_bar.setContentsMargins(2, 2, 2, 2)
        control_bar.setSpacing(8)

        # Channel selector
        chan_lbl = QLabel("Channel:")
        chan_lbl.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        chan_lbl.setStyleSheet("color: #e6edf3;")
        control_bar.addWidget(chan_lbl)

        self._channel_combo = QComboBox()
        self._channel_combo.setFont(QFont("Inter", 9))
        self._channel_combo.setStyleSheet("""
            QComboBox {
                background-color: #161b22;
                color: #f0f6fc;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 4px 10px;
                min-width: 150px;
                font-weight: 500;
            }
            QComboBox:hover {
                border-color: #58a6ff;
            }
            QComboBox QAbstractItemView {
                background-color: #161b22;
                color: #f0f6fc;
                selection-background-color: #1f6feb;
                selection-color: #ffffff;
            }
        """)
        self._channel_combo.addItems([
            "I + Q (Overlay)",
            "I (In-Phase)",
            "Q (Quadrature)",
            "Amplitude Envelope",
            "Phase (rad)",
            "Instantaneous Frequency",
        ])
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        control_bar.addWidget(self._channel_combo)

        # Quick zoom controls
        btn_style_primary = """
            QPushButton {
                background-color: #21262d;
                color: #58a6ff;
                border: 1px solid #388bfd;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: 600;
                font-size: 8.5pt;
            }
            QPushButton:hover {
                background-color: #388bfd33;
            }
        """
        btn_style_secondary = """
            QPushButton {
                background-color: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 8.5pt;
            }
            QPushButton:hover {
                border-color: #58a6ff;
                color: #58a6ff;
            }
        """

        self._btn_zoom_cycles = QPushButton("Wave Zoom")
        self._btn_zoom_cycles.setToolTip("Zoom to first 500 samples to inspect wave cycles")
        self._btn_zoom_cycles.setStyleSheet(btn_style_primary)
        self._btn_zoom_cycles.clicked.connect(self._zoom_to_cycles)
        control_bar.addWidget(self._btn_zoom_cycles)

        self._btn_fit_all = QPushButton("Fit")
        self._btn_fit_all.setToolTip("Fit entire signal into view")
        self._btn_fit_all.setStyleSheet(btn_style_secondary)
        self._btn_fit_all.clicked.connect(self._fit_all)
        control_bar.addWidget(self._btn_fit_all)

        self._btn_zoom_in = QPushButton("+")
        self._btn_zoom_in.setFixedWidth(26)
        self._btn_zoom_in.setToolTip("Zoom in horizontally")
        self._btn_zoom_in.setStyleSheet(btn_style_secondary)
        self._btn_zoom_in.clicked.connect(self._zoom_in)
        control_bar.addWidget(self._btn_zoom_in)

        self._btn_zoom_out = QPushButton("-")
        self._btn_zoom_out.setFixedWidth(26)
        self._btn_zoom_out.setToolTip("Zoom out horizontally")
        self._btn_zoom_out.setStyleSheet(btn_style_secondary)
        self._btn_zoom_out.clicked.connect(self._zoom_out)
        control_bar.addWidget(self._btn_zoom_out)

        control_bar.addStretch(1)

        # Real-time telemetry readout (Ignored horizontal size policy prevents minimum-width inflation)
        self._telemetry_label = QLabel("Hover cursor over waveform")
        self._telemetry_label.setFont(QFont("Inter", 8))
        self._telemetry_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._telemetry_label.setStyleSheet("""
            color: #c9d1d9;
            background-color: #161b22;
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid #30363d;
        """)
        control_bar.addWidget(self._telemetry_label, 1)

        root.addLayout(control_bar)

        # ── 2. Full-Height Waveform Plot ─────────────────────────────────
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.20)
        self._plot.getPlotItem().setLabel('left', 'Amplitude', color='#c9d1d9')
        self._plot.getPlotItem().setLabel('bottom', 'Sample Index', color='#c9d1d9')
        self._plot.getPlotItem().setMenuEnabled(True)

        # Configure high-performance view box
        vb = self._plot.getViewBox()
        vb.setMouseMode(pg.ViewBox.PanMode)

        # Zero baseline
        self._zero_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen('#30363d', width=1, style=Qt.PenStyle.DashLine)
        )
        self._plot.addItem(self._zero_line)

        # Waveform curves (peak downsampling for zero-lag and accurate envelope spikes)
        self._curve_i = self._plot.plot(
            pen=pg.mkPen('#58a6ff', width=1.5),
            name="I (In-Phase)",
            autoDownsample=True,
            clipToView=True,
            downsampleMethod='peak',
        )
        self._curve_q = self._plot.plot(
            pen=pg.mkPen('#3fb950', width=1.5),
            name="Q (Quadrature)",
            autoDownsample=True,
            clipToView=True,
            downsampleMethod='peak',
        )
        self._curve_extra = self._plot.plot(
            pen=pg.mkPen('#f78166', width=1.5),
            autoDownsample=True,
            clipToView=True,
            downsampleMethod='peak',
        )
        self._curve_extra.setVisible(False)

        # Lightweight crosshairs
        self._vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#e3b341', width=1, style=Qt.PenStyle.DashLine)
        )
        self._hline = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen('#484f58', width=1)
        )
        self._plot.addItem(self._vline)
        self._plot.addItem(self._hline)

        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        root.addWidget(self._plot, 1)

        # ── 3. Bottom Status ────────────────────────────────────────────
        self._status_bar = QLabel("No signal loaded.")
        self._status_bar.setFont(QFont("Inter", 8))
        self._status_bar.setStyleSheet("color: #8b949e; padding: 2px 4px;")
        root.addWidget(self._status_bar)

    # ------------------------------------------------------------------
    # Data Loading & Refresh
    # ------------------------------------------------------------------

    def set_data(self, waveform_dict: dict, **kwargs):
        """
        Load waveform data and fit display. Accepts any extra kwargs for compatibility.
        """
        self._waveform = waveform_dict

        i = waveform_dict.get("i")
        q = waveform_dict.get("q")
        if i is None or len(i) == 0:
            self.clear()
            return

        self._i_data = np.asarray(i, dtype=np.float32)
        self._q_data = np.asarray(q, dtype=np.float32) if q is not None else np.zeros_like(self._i_data)
        self._x_data = np.arange(len(self._i_data), dtype=np.float32)

        n_samples = waveform_dict.get("n_samples", len(self._i_data))
        dec = waveform_dict.get("decimation_factor", 1)
        self._status_bar.setText(
            f"Waveform: {len(self._i_data):,} display samples "
            f"({n_samples:,} raw samples, {dec}x decimation)  |  "
            f"Use mouse wheel to zoom smoothly, left-click drag to pan"
        )

        self._refresh_plot()
        self._fit_all()

    def set_show_bits(self, show: bool):
        """Maintained for interface compatibility."""
        pass

    def clear(self):
        """Reset to empty state."""
        self._waveform = None
        self._x_data = np.array([], dtype=np.float32)
        self._i_data = np.array([], dtype=np.float32)
        self._q_data = np.array([], dtype=np.float32)
        self._curve_i.setData([], [])
        self._curve_q.setData([], [])
        self._curve_extra.setData([], [])
        self._status_bar.setText("No signal loaded.")
        self._telemetry_label.setText("Hover cursor over waveform to inspect values")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh_plot(self):
        if self._waveform is None or len(self._x_data) == 0:
            return

        choice = self._channel_combo.currentIndex()
        x = self._x_data
        i = self._i_data
        q = self._q_data

        if choice == 0:  # I + Q (Overlay)
            self._curve_i.setVisible(True)
            self._curve_q.setVisible(True)
            self._curve_extra.setVisible(False)
            self._curve_i.setData(x, i)
            self._curve_q.setData(x, q)
            self._plot.getPlotItem().setLabel('left', 'I & Q Amplitude')

        elif choice == 1:  # I (In-Phase)
            self._curve_i.setVisible(True)
            self._curve_q.setVisible(False)
            self._curve_extra.setVisible(False)
            self._curve_i.setData(x, i)
            self._plot.getPlotItem().setLabel('left', 'In-Phase Amplitude (I)')

        elif choice == 2:  # Q (Quadrature)
            self._curve_i.setVisible(False)
            self._curve_q.setVisible(True)
            self._curve_extra.setVisible(False)
            self._curve_q.setData(x, q)
            self._plot.getPlotItem().setLabel('left', 'Quadrature Amplitude (Q)')

        elif choice == 3:  # Amplitude Envelope
            self._curve_i.setVisible(False)
            self._curve_q.setVisible(False)
            self._curve_extra.setVisible(True)
            amp = self._waveform.get("amplitude")
            if amp is None:
                amp = np.abs(i + 1j * q)
            self._curve_extra.setPen(pg.mkPen('#f78166', width=1.5))
            self._curve_extra.setData(x, amp)
            self._plot.getPlotItem().setLabel('left', 'Envelope |I + jQ|')

        elif choice == 4:  # Phase
            self._curve_i.setVisible(False)
            self._curve_q.setVisible(False)
            self._curve_extra.setVisible(True)
            phase = self._waveform.get("phase_rad")
            if phase is None:
                phase = np.angle(i + 1j * q)
            self._curve_extra.setPen(pg.mkPen('#d2a8ff', width=1.5))
            self._curve_extra.setData(x, phase)
            self._plot.getPlotItem().setLabel('left', 'Phase (radians)')

        elif choice == 5:  # Instantaneous Frequency
            self._curve_i.setVisible(False)
            self._curve_q.setVisible(False)
            self._curve_extra.setVisible(True)
            freq = self._waveform.get("inst_freq")
            if freq is not None and len(freq) > 0:
                self._curve_extra.setPen(pg.mkPen('#ffa657', width=1.5))
                self._curve_extra.setData(np.arange(len(freq), dtype=np.float32), freq)
            self._plot.getPlotItem().setLabel('left', 'Normalized Frequency')

    def _on_channel_changed(self):
        self._refresh_plot()

    # ------------------------------------------------------------------
    # High-Performance Mouse Telemetry (O(1))
    # ------------------------------------------------------------------

    def _on_mouse_moved(self, pos):
        if not self._plot.sceneBoundingRect().contains(pos):
            return

        mp = self._plot.plotItem.vb.mapSceneToView(pos)
        x_val = mp.x()
        y_val = mp.y()

        self._vline.setPos(x_val)
        self._hline.setPos(y_val)

        idx = int(round(x_val))
        if 0 <= idx < len(self._i_data):
            i_val = self._i_data[idx]
            q_val = self._q_data[idx]
            amp = np.sqrt(i_val * i_val + q_val * q_val)
            self._telemetry_label.setText(
                f"Sample: <b style='color:#58a6ff;'>{idx:,}</b> | "
                f"I: <b style='color:#58a6ff;'>{i_val:+.3f}</b> | "
                f"Q: <b style='color:#3fb950;'>{q_val:+.3f}</b> | "
                f"Amp: <b style='color:#f0f6fc;'>{amp:.3f}</b>"
            )
        else:
            self._telemetry_label.setText(f"Sample: {idx:,} (Out of Range)")

    # ------------------------------------------------------------------
    # Quick Navigation Slots
    # ------------------------------------------------------------------

    def _fit_all(self):
        """Fit entire signal into view."""
        if len(self._x_data) > 0:
            self._plot.setXRange(0, len(self._x_data), padding=0.01)

    def _zoom_to_cycles(self):
        """Zoom into first 500 samples so individual wave cycles are large and clear."""
        if len(self._x_data) > 0:
            span = min(500, len(self._x_data))
            self._plot.setXRange(0, max(20, span), padding=0.02)

    def _zoom_in(self):
        """Zoom in 2x horizontally around center."""
        vb = self._plot.getViewBox()
        x_min, x_max = vb.viewRange()[0]
        center = (x_min + x_max) / 2.0
        new_half = (x_max - x_min) / 4.0
        if new_half > 5:
            self._plot.setXRange(center - new_half, center + new_half, padding=0)

    def _zoom_out(self):
        """Zoom out 2x horizontally around center."""
        vb = self._plot.getViewBox()
        x_min, x_max = vb.viewRange()[0]
        center = (x_min + x_max) / 2.0
        new_half = (x_max - x_min)
        if len(self._x_data) > 0:
            x0 = max(0, center - new_half)
            x1 = min(len(self._x_data), center + new_half)
            self._plot.setXRange(x0, x1, padding=0)
