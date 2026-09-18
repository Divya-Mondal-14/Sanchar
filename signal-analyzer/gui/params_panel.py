"""
gui/params_panel.py

PyQt6 detected-parameters panel for the SIH signal analyzer.

Displays a structured, high-contrast table of all detected signal parameters
and confidence scores with vivid color-coding and clear readability.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QBrush


# Confidence thresholds
_GREEN = 0.85
_AMBER = 0.65

# Vibrant, high-contrast palette
_COLOR_GREEN_BG = QColor("#195c27")
_COLOR_GREEN_FG = QColor("#7ee787")
_COLOR_AMBER_BG = QColor("#593f00")
_COLOR_AMBER_FG = QColor("#f2cc60")
_COLOR_RED_BG   = QColor("#5c1d1d")
_COLOR_RED_FG   = QColor("#ff7b72")

_TEXT_PARAM_NAME = QColor("#f0f6fc")  # Crisp high-contrast white
_TEXT_CYAN       = QColor("#79c0ff")  # Electric cyan for RF metrics
_TEXT_AMBER      = QColor("#e3b341")  # Vibrant amber for protocol/FEC
_TEXT_PURPLE     = QColor("#d2a8ff")  # Lilac for byte/bit offsets
_TEXT_MUTED      = QColor("#c9d1d9")  # Bright silver for neutral values


def _confidence_colors(conf: float) -> tuple[QColor, QColor]:
    """Return (bg, fg) QColor pair based on confidence."""
    if conf >= _GREEN:
        return _COLOR_GREEN_BG, _COLOR_GREEN_FG
    elif conf >= _AMBER:
        return _COLOR_AMBER_BG, _COLOR_AMBER_FG
    else:
        return _COLOR_RED_BG, _COLOR_RED_FG


def _hz_fmt(hz: float | None) -> str:
    if hz is None:
        return "None"
    if abs(hz) >= 1e6:
        return f"{hz / 1e6:.3f} MHz"
    elif abs(hz) >= 1e3:
        return f"{hz / 1e3:.3f} kHz"
    return f"{hz:.1f} Hz"


class ParamsPanel(QWidget):
    """Detected signal parameters read-only display with high-contrast styling."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Title
        title = QLabel("Detected Parameters")
        title.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        title.setStyleSheet("""
            color: #ffffff;
            border-bottom: 2px solid #58a6ff;
            padding-bottom: 6px;
            letter-spacing: 0.5px;
        """)
        root.addWidget(title)

        # Table
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(False)
        self._table.setShowGrid(True)
        self._table.setFont(QFont("Inter", 9))
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #0d1117;
                alternate-background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #f0f6fc;
                gridline-color: #21262d;
            }
            QTableWidget::item {
                padding: 6px 10px;
                border-bottom: 1px solid #1c2128;
            }
            QHeaderView::section {
                background-color: #161b22;
                color: #58a6ff;
                font-weight: bold;
                font-size: 9pt;
                padding: 7px 10px;
                border: none;
                border-bottom: 2px solid #30363d;
            }
        """)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._table)

        # All-probabilities mini display
        self._probs_label = QLabel("")
        self._probs_label.setFont(QFont("Consolas", 8))
        self._probs_label.setStyleSheet("""
            QLabel {
                color: #c9d1d9;
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 8px 10px;
            }
        """)
        self._probs_label.setWordWrap(True)
        root.addWidget(self._probs_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_params(self, results: dict):
        """Populate the parameter table from a results dict."""
        self._table.setRowCount(0)

        def _add(param: str, value: str, conf: float | None = None, text_color: QColor | None = None):
            row = self._table.rowCount()
            self._table.insertRow(row)

            # Parameter Name (Bright crisp white, bold)
            p_item = QTableWidgetItem(param)
            p_item.setForeground(QBrush(_TEXT_PARAM_NAME))
            p_font = p_item.font()
            p_font.setBold(True)
            p_item.setFont(p_font)

            # Parameter Value
            v_item = QTableWidgetItem(value)
            v_font = v_item.font()
            v_font.setBold(True)
            v_item.setFont(v_font)

            if conf is not None:
                bg, fg = _confidence_colors(conf)
                v_item.setBackground(QBrush(bg))
                v_item.setForeground(QBrush(fg))
            elif text_color is not None:
                v_item.setForeground(QBrush(text_color))
            else:
                v_item.setForeground(QBrush(_TEXT_MUTED))

            self._table.setItem(row, 0, p_item)
            self._table.setItem(row, 1, v_item)

        # ── 1. Modulation & Confidence ────────────────────────────────
        mod = results.get("modulation", "None")
        conf = results.get("confidence")
        conf_str = f"{conf:.1%}" if conf is not None else "None"
        _add("Modulation", f"{mod}  ({conf_str})", conf=conf)

        # ── 2. Spectral & RF Parameters (Cyan) ────────────────────────
        sr = results.get("sample_rate")
        _add("Sample Rate", _hz_fmt(sr), text_color=_TEXT_CYAN)

        obw = results.get("occupied_bandwidth_hz")
        _add("Occupied BW", _hz_fmt(obw), text_color=_TEXT_CYAN)

        cfo = results.get("center_freq_offset_hz")
        _add("Center Freq Offset", _hz_fmt(cfo), text_color=_TEXT_CYAN)

        # ── 3. Coding & Protocols (Amber / Green) ─────────────────────
        di = results.get("deinterleaver", "None")
        if str(di).lower() in ("none", "—", ""):
            _add("De-interleaver", "None (Direct)", text_color=_TEXT_MUTED)
        else:
            _add("De-interleaver", str(di), conf=0.95)

        fec = results.get("fec_scheme", "None")
        fec_score = results.get("fec_score")
        if str(fec).lower() in ("none", "—", "") or fec_score is None or fec_score == 0.0:
            _add("FEC Scheme", "None (Raw bitstream)", text_color=_TEXT_MUTED)
        else:
            fec_str = f"{fec}  ({fec_score:.1%})"
            _add("FEC Scheme", fec_str, conf=fec_score)

        # ── 4. Frame & Byte Telemetry (Purple) ────────────────────────
        n_bytes = results.get("decoded_bytes")
        _add("Decoded Bytes", f"{n_bytes:,}" if n_bytes is not None else "None", text_color=_TEXT_PURPLE)

        h_off = results.get("header_offset")
        p_off = results.get("payload_offset")
        _add("Header Offset (bits)", str(h_off) if h_off is not None else "None", text_color=_TEXT_PURPLE)
        _add("Payload Offset (bits)", str(p_off) if p_off is not None else "None", text_color=_TEXT_PURPLE)

        # ── 5. All Probabilities ──────────────────────────────────────
        all_probs = results.get("all_probs", {})
        if all_probs:
            prob_lines = ["Class probabilities:"]
            for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                bar_len = int(prob * 18)
                bar = "#" * bar_len + "." * (18 - bar_len)
                prob_lines.append(f"  {cls:<6} [{bar}] {prob:.3f}")
            self._probs_label.setText("\n".join(prob_lines))
        else:
            self._probs_label.setText("")

    def clear(self):
        self._table.setRowCount(0)
        self._probs_label.setText("")
