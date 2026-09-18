"""
gui/synthetic_dialog.py

PyQt6 dialog for generating synthetic IQ signal files.

Opens from File → Generate Synthetic IQ... or the toolbar button.
On Accept it calls preprocessing/synthetic_iq.generate_and_save() and
returns the saved file path so the main window can auto-load it into
the analysis pipeline.

Public API
----------
    SyntheticDialog(parent) -> QDialog
        .exec()      — show modal; returns QDialog.DialogCode.Accepted / Rejected
        .result_path — str path to the generated file (only valid after Accept)
        .result_meta — dict with signal parameters
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
    QPushButton, QLineEdit, QFileDialog, QGroupBox,
    QCheckBox, QProgressBar, QMessageBox, QSizePolicy,
    QDialogButtonBox, QFrame, QScrollArea, QWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPalette


_STYLE = """
QDialog {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 9pt;
}
QGroupBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 6px;
    font-weight: bold;
    color: #8b949e;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QLabel { color: #e6edf3; }
QLabel.dim { color: #8b949e; }
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 26px;
}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
    border-color: #388bfd;
}
QComboBox::drop-down { border: none; }
QComboBox::down-arrow { image: none; width: 0; }
QSlider::groove:horizontal {
    background: #30363d; height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #388bfd; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #388bfd; border-radius: 2px; }
QPushButton {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 9pt;
}
QPushButton:hover { background-color: #30363d; border-color: #8b949e; }
QPushButton#generate_btn {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #1f6feb, stop:1 #388bfd);
    color: white;
    border: none;
    font-weight: bold;
    font-size: 10pt;
    padding: 8px 24px;
}
QPushButton#generate_btn:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #388bfd, stop:1 #58a6ff);
}
QProgressBar {
    background: #21262d; border: 1px solid #30363d; border-radius: 4px;
    text-align: center; color: #e6edf3; height: 14px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #1f6feb, stop:1 #58a6ff);
    border-radius: 3px;
}
QCheckBox { color: #e6edf3; spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid #484f58; background: #21262d; }
QCheckBox::indicator:checked { background: #388bfd; border-color: #388bfd; }
QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #30363d; }
"""

# Constellation preview colors for each mod
_MOD_COLORS = {
    "BPSK": "#58a6ff",
    "QPSK": "#3fb950",
    "GMSK": "#ffa657",
}


class _GenerateWorker(QObject):
    finished = pyqtSignal(str, dict)   # (file_path, meta)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, params: dict):
        super().__init__()
        self._p = params

    def run(self):
        try:
            self.progress.emit(20)
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "preprocessing"
            ))
            from synthetic_iq import generate_and_save

            self.progress.emit(50)
            path, _, meta = generate_and_save(
                modulation=self._p["modulation"],
                n_symbols=self._p["n_symbols"],
                sample_rate=self._p["sample_rate"],
                samples_per_symbol=self._p["samples_per_symbol"],
                snr_db=self._p["snr_db"],
                carrier_offset_hz=self._p["carrier_offset_hz"],
                seed=self._p["seed"] if self._p["use_seed"] else None,
                output_dir=self._p["output_dir"],
                fmt=self._p["fmt"],
                fec_scheme=self._p.get("fec_scheme", "none"),
                interleaver=self._p.get("interleaver", "none"),
                preamble=self._p.get("preamble", "none"),
            )
            self.progress.emit(100)
            self.finished.emit(path, meta)
        except Exception as exc:
            import traceback
            self.error.emit(traceback.format_exc())


class SyntheticDialog(QDialog):
    """Modal dialog for generating a synthetic IQ capture."""

    def __init__(self, parent=None, default_output_dir: str = "."):
        super().__init__(parent)
        self.setWindowTitle("Generate Synthetic IQ Signal")
        self.setModal(True)
        self.setMinimumWidth(720)
        self.resize(760, 520)
        self.setStyleSheet(_STYLE)

        self.result_path: str = ""
        self.result_meta: dict = {}
        self._default_output_dir = default_output_dir
        self._thread: QThread | None = None
        self._worker: _GenerateWorker | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Title ─────────────────────────────────────────────────────
        title = QLabel("Synthetic IQ Signal Generator")
        title.setFont(QFont("Inter", 13, QFont.Weight.DemiBold))
        title.setStyleSheet("color: #e6edf3; margin-bottom: 4px;")
        root.addWidget(title)

        subtitle = QLabel(
            "Creates a pulse-shaped, noise-added IQ capture and loads it directly "
            "into the analysis pipeline."
        )
        subtitle.setStyleSheet("color: #8b949e; font-size: 8pt;")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        # ── Signal parameters ─────────────────────────────────────────
        sig_group = QGroupBox("Signal Parameters")
        sig_layout = QGridLayout(sig_group)
        sig_layout.setColumnStretch(1, 1)
        sig_layout.setSpacing(8)
        row = 0

        # Modulation
        sig_layout.addWidget(QLabel("Modulation:"), row, 0)
        self._mod_combo = QComboBox()
        self._mod_combo.addItems(["BPSK", "QPSK", "GMSK"])
        self._mod_combo.setCurrentText("BPSK")
        self._mod_combo.currentTextChanged.connect(self._on_mod_changed)
        sig_layout.addWidget(self._mod_combo, row, 1)
        self._mod_badge = QLabel("●  BPSK")
        self._mod_badge.setStyleSheet(f"color: {_MOD_COLORS['BPSK']}; font-weight: bold;")
        sig_layout.addWidget(self._mod_badge, row, 2)
        row += 1

        # Number of symbols
        sig_layout.addWidget(QLabel("Symbols:"), row, 0)
        self._n_syms = QSpinBox()
        self._n_syms.setRange(128, 65536)
        self._n_syms.setValue(2048)
        self._n_syms.setSingleStep(256)
        self._n_syms.valueChanged.connect(self._update_preview)
        sig_layout.addWidget(self._n_syms, row, 1, 1, 2)
        row += 1

        # Sample rate
        sig_layout.addWidget(QLabel("Sample Rate:"), row, 0)
        self._sample_rate = QComboBox()
        for label, val in [
            ("250 kHz", 250_000), ("500 kHz", 500_000),
            ("1 MHz", 1_000_000), ("2 MHz", 2_000_000),
            ("4 MHz", 4_000_000),
        ]:
            self._sample_rate.addItem(label, val)
        self._sample_rate.setCurrentIndex(2)   # 1 MHz default
        self._sample_rate.currentIndexChanged.connect(self._update_preview)
        sig_layout.addWidget(self._sample_rate, row, 1, 1, 2)
        row += 1

        # Samples per symbol
        sig_layout.addWidget(QLabel("Samples / Symbol:"), row, 0)
        self._sps = QSpinBox()
        self._sps.setRange(2, 64)
        self._sps.setValue(8)
        self._sps.valueChanged.connect(self._update_preview)
        sig_layout.addWidget(self._sps, row, 1, 1, 2)
        row += 1

        # ── Coding & Framing ──────────────────────────────────────────
        code_group = QGroupBox("Coding & Framing (FEC / Interleaving / Sync)")
        code_layout = QGridLayout(code_group)
        code_layout.setColumnStretch(1, 1)
        code_layout.setSpacing(8)
        crow = 0

        # FEC Scheme
        code_layout.addWidget(QLabel("FEC Scheme:"), crow, 0)
        self._fec_combo = QComboBox()
        self._fec_combo.addItem("None (Raw bitstream)", "none")
        self._fec_combo.addItem("Convolutional (k=7, CCSDS / 802.11)", "viterbi_k7")
        self._fec_combo.addItem("Convolutional (k=3, simple rate-1/2)", "viterbi_k3")
        self._fec_combo.addItem("Reed-Solomon (nsym=8)", "rs_8")
        self._fec_combo.currentIndexChanged.connect(self._update_preview)
        code_layout.addWidget(self._fec_combo, crow, 1, 1, 2)
        crow += 1

        # Interleaver
        code_layout.addWidget(QLabel("Interleaver:"), crow, 0)
        self._il_combo = QComboBox()
        self._il_combo.addItem("None (Direct transmission)", "none")
        self._il_combo.addItem("Block (8x8 Matrix)", "block_8x8")
        self._il_combo.addItem("Block (4x8 Matrix)", "block_4x8")
        self._il_combo.addItem("Convolutional (Depth=4)", "conv_depth4")
        self._il_combo.currentIndexChanged.connect(self._on_interleaver_changed)
        code_layout.addWidget(self._il_combo, crow, 1, 1, 2)
        crow += 1

        # Framing / Preamble
        code_layout.addWidget(QLabel("Preamble / Sync:"), crow, 0)
        self._preamble_combo = QComboBox()
        self._preamble_combo.addItem("CCSDS ASM (0x1ACFFC1D)", "ccsds")
        self._preamble_combo.addItem("Barker-13 Code", "barker13")
        self._preamble_combo.addItem("None (Raw stream)", "none")
        self._preamble_combo.currentIndexChanged.connect(self._update_preview)
        code_layout.addWidget(self._preamble_combo, crow, 1, 1, 2)
        crow += 1

        # Helper hint
        il_hint = QLabel("ℹ Interleaver detection requires an FEC Scheme (e.g. Conv k=7) to validate.")
        il_hint.setStyleSheet("color: #8b949e; font-size: 7.5pt; font-style: italic;")
        code_layout.addWidget(il_hint, crow, 0, 1, 3)
        crow += 1

        # ── Channel parameters ────────────────────────────────────────
        chan_group = QGroupBox("Channel Parameters")
        chan_layout = QGridLayout(chan_group)
        chan_layout.setColumnStretch(1, 1)
        chan_layout.setSpacing(8)
        row = 0

        # SNR slider + spinbox
        chan_layout.addWidget(QLabel("SNR (dB):"), row, 0)
        snr_row = QHBoxLayout()
        self._snr_slider = QSlider(Qt.Orientation.Horizontal)
        self._snr_slider.setRange(-10, 40)
        self._snr_slider.setValue(20)
        self._snr_spin = QSpinBox()
        self._snr_spin.setRange(-10, 40)
        self._snr_spin.setValue(20)
        self._snr_spin.setFixedWidth(60)
        self._snr_slider.valueChanged.connect(self._snr_spin.setValue)
        self._snr_spin.valueChanged.connect(self._snr_slider.setValue)
        snr_row.addWidget(self._snr_slider)
        snr_row.addWidget(self._snr_spin)
        chan_layout.addLayout(snr_row, row, 1, 1, 2)
        row += 1

        # Carrier offset
        chan_layout.addWidget(QLabel("Carrier Offset (Hz):"), row, 0)
        self._cfo = QDoubleSpinBox()
        self._cfo.setRange(-500_000, 500_000)
        self._cfo.setValue(0.0)
        self._cfo.setSingleStep(1000)
        self._cfo.setDecimals(0)
        chan_layout.addWidget(self._cfo, row, 1, 1, 2)
        row += 1

        # Seed
        seed_row = QHBoxLayout()
        self._use_seed = QCheckBox("Fixed seed:")
        self._use_seed.setChecked(True)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setValue(42)
        seed_row.addWidget(self._use_seed)
        seed_row.addWidget(self._seed_spin)
        seed_row.addStretch(1)
        chan_layout.addLayout(seed_row, row, 0, 1, 3)

        # ── Output settings ───────────────────────────────────────────
        out_group = QGroupBox("Output")
        out_layout = QGridLayout(out_group)
        out_layout.setColumnStretch(1, 1)
        out_layout.setSpacing(8)
        row = 0

        out_layout.addWidget(QLabel("Format:"), row, 0)
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(["WAV (stereo IQ, 16-bit)", "IQ raw (float32)"])
        out_layout.addWidget(self._fmt_combo, row, 1, 1, 2)
        row += 1

        out_layout.addWidget(QLabel("Save to:"), row, 0)
        self._out_path = QLineEdit(self._default_output_dir)
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_dir)
        out_layout.addWidget(self._out_path, row, 1)
        out_layout.addWidget(browse_btn, row, 2)

        # ── Preview label ─────────────────────────────────────────────
        self._preview = QLabel()
        self._preview.setStyleSheet(
            "color: #8b949e; font-size: 8pt; font-family: 'Courier New';"
            "background: #161b22; border-radius: 4px; padding: 6px 10px;"
        )
        self._preview.setWordWrap(True)
        self._update_preview()

        # ── 2-Column Scroll Area for content ──────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        cols_layout = QHBoxLayout()
        cols_layout.setSpacing(10)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        left_col.addWidget(sig_group)
        left_col.addWidget(chan_group)
        left_col.addStretch(1)

        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        right_col.addWidget(code_group)
        right_col.addWidget(out_group)
        right_col.addStretch(1)

        cols_layout.addLayout(left_col, 1)
        cols_layout.addLayout(right_col, 1)

        scroll_layout.addLayout(cols_layout)
        scroll_layout.addWidget(self._preview)

        scroll.setWidget(scroll_widget)
        root.addWidget(scroll, 1)

        # ── Progress bar & Pinned Bottom Buttons (Always Visible) ─────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._gen_btn = QPushButton("Generate & Load")
        self._gen_btn.setObjectName("generate_btn")
        self._gen_btn.setDefault(True)
        self._gen_btn.clicked.connect(self._start_generate)
        btn_row.addWidget(self._gen_btn)

        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_interleaver_changed(self, idx: int):
        # If an interleaver is selected and FEC is currently None, auto-select standard Conv k=7
        if idx > 0 and self._fec_combo.currentIndex() == 0:
            self._fec_combo.setCurrentIndex(1)  # Convolutional k=7
        self._update_preview()

    def _on_mod_changed(self, mod: str):
        color = _MOD_COLORS.get(mod, "#e6edf3")
        self._mod_badge.setText(f"●  {mod}")
        self._mod_badge.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._update_preview()

    def _update_preview(self):
        n_sym = self._n_syms.value()
        sr = self._sample_rate.currentData()
        sps = self._sps.value()
        mod = self._mod_combo.currentText()
        bps = 1 if mod in ("BPSK", "GMSK") else 2
        total_samples = n_sym * sps
        duration_ms = total_samples / sr * 1000
        n_bits = n_sym * bps

        fec_txt = self._fec_combo.currentText() if hasattr(self, "_fec_combo") else "None"
        il_txt  = self._il_combo.currentText() if hasattr(self, "_il_combo") else "None"
        pre_txt = self._preamble_combo.currentText() if hasattr(self, "_preamble_combo") else "None"

        # Approx file size
        wav_kb = (total_samples * 2 * 2) / 1024   # 2 ch * 2 bytes
        iq_kb  = (total_samples * 2 * 4) / 1024   # 2 float32

        self._preview.setText(
            f"  {mod}  |  {n_sym} symbols  |  ~{n_bits} bits  |  {total_samples:,} samples @ {sr/1e3:.0f} kHz\n"
            f"  FEC: {fec_txt.split(' (')[0]}  |  Interleaver: {il_txt.split(' (')[0]}  |  Sync: {pre_txt.split(' (')[0]}\n"
            f"  Capture: {duration_ms:.1f} ms  |  WAV: ~{wav_kb:.0f} KB  |  .iq: ~{iq_kb:.0f} KB"
        )

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose output directory", self._out_path.text()
        )
        if d:
            self._out_path.setText(d)

    def _start_generate(self):
        fmt_str = "wav" if self._fmt_combo.currentIndex() == 0 else "iq"
        params = {
            "modulation":         self._mod_combo.currentText(),
            "n_symbols":          self._n_syms.value(),
            "sample_rate":        float(self._sample_rate.currentData()),
            "samples_per_symbol": self._sps.value(),
            "snr_db":             float(self._snr_spin.value()),
            "carrier_offset_hz":  self._cfo.value(),
            "use_seed":           self._use_seed.isChecked(),
            "seed":               self._seed_spin.value(),
            "output_dir":         self._out_path.text() or ".",
            "fmt":                fmt_str,
            "fec_scheme":         self._fec_combo.currentData(),
            "interleaver":        self._il_combo.currentData(),
            "preamble":           self._preamble_combo.currentData(),
        }

        self._gen_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        self._worker = _GenerateWorker(params)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_generated)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_generated(self, path: str, meta: dict):
        self._progress.setVisible(False)
        self._gen_btn.setEnabled(True)
        self.result_path = path
        self.result_meta = meta
        self.accept()

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._gen_btn.setEnabled(True)
        QMessageBox.critical(self, "Generation Error", msg)
