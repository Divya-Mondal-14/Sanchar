"""
gui/main_window.py

PyQt6 main application window for the SIH Signal Analyzer.

Architecture
------------
    MainWindow (QMainWindow)
    ├── Toolbar: Open File, Analyze
    ├── Central widget (QSplitter horizontal)
    │   ├── Left panel (QSplitter vertical)
    │   │   ├── WaveformPanel
    │   │   ├── SpectrogramPanel
    │   │   ├── ConstellationPanel
    │   │   └── BitstreamPanel
    │   └── Right panel: ParamsPanel
    └── Status bar + progress bar

Analysis runs in a QThread worker (AnalyzerWorker) so the GUI stays
responsive during file loading and inference.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout,
    QHBoxLayout, QToolBar, QStatusBar, QProgressBar, QFileDialog,
    QMessageBox, QLabel, QSizePolicy, QTabWidget, QPushButton,
    QDialog, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QAction, QIcon, QFont

# Resolve import paths relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in [
    _PROJECT_ROOT,
    os.path.join(_PROJECT_ROOT, "models"),
    os.path.join(_PROJECT_ROOT, "preprocessing"),
    os.path.join(_PROJECT_ROOT, "gnuradio_pipeline"),
    os.path.join(_PROJECT_ROOT, "correlation"),
    os.path.join(_PROJECT_ROOT, "gui"),
]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from gui.waveform_panel import WaveformPanel
from gui.spectrogram_panel import SpectrogramPanel
from gui.constellation_panel import ConstellationPanel
from gui.params_panel import ParamsPanel
from gui.bitstream_panel import BitstreamPanel


# ---------------------------------------------------------------------------
# Analyzer worker (runs in a separate QThread)
# ---------------------------------------------------------------------------

class AnalyzerWorker(QObject):
    """
    Runs the full analysis pipeline off the main thread.

    Pipeline:
        file_loader.load_file()
        -> models/inference.predict_modulation()
        -> preprocessing/spectral_features.analyze()
        -> preprocessing/waveform_view.waveform_display_data()
        -> gnuradio_pipeline/demod.demodulate()          [optional — needs GNU Radio]
        -> gnuradio_pipeline/deinterleave.try_all_deinterleavers()
        -> gnuradio_pipeline/fec_decode.try_all_fec()
        -> correlation/sync_correlator.find_preamble()
    """

    progress = pyqtSignal(int, str)      # (percent, message)
    finished = pyqtSignal(dict)          # full results dict
    error    = pyqtSignal(str)           # error message

    def __init__(self, file_path: str, parent: QObject | None = None):
        super().__init__(parent)
        self.file_path = file_path

    def run(self):
        results: dict = {}
        try:
            # ── 1. Load file ─────────────────────────────────────────
            self.progress.emit(5, "Loading file...")
            from file_loader import load_file, load_wav, load_iq_raw
            windows, meta = load_file(self.file_path)
            results["meta"] = meta

            if meta["num_windows"] == 0:
                self.error.emit("File too short — no full 1024-sample windows available.")
                return

            # Re-load unwindowed complex samples for spectral / waveform analysis
            ext = os.path.splitext(self.file_path)[1].lower()
            if ext == ".wav":
                complex_samples, sample_rate = load_wav(self.file_path)
            else:
                from file_loader import load_iq_raw
                complex_samples, sample_rate = load_iq_raw(self.file_path)

            results["sample_rate"] = sample_rate or meta.get("sample_rate")

            # ── 2. Modulation classification ─────────────────────────
            self.progress.emit(20, "Classifying modulation...")
            from inference import predict_modulation
            # Multi-window batched classification (robust against startup edge transients)
            ml_result = predict_modulation(windows)
            results["modulation"]  = ml_result["class"]
            results["confidence"]  = ml_result["confidence"]
            results["all_probs"]   = ml_result["all_probs"]

            # ── 3. Spectral analysis ──────────────────────────────────
            self.progress.emit(35, "Computing spectrum...")
            from spectral_features import analyze as spectral_analyze
            sr = results["sample_rate"] or 1_000_000
            spectral = spectral_analyze(complex_samples, sr)
            results["spectral"] = spectral
            results["occupied_bandwidth_hz"]  = spectral["occupied_bandwidth_hz"]
            results["center_freq_offset_hz"]  = spectral["center_freq_offset_hz"]

            # ── 4. Waveform data ──────────────────────────────────────
            self.progress.emit(50, "Preparing waveform data...")
            from waveform_view import waveform_display_data
            results["waveform"] = waveform_display_data(complex_samples, max_points=32768)

            # IQ for constellation: use a settled window (avoiding turn-on transients)
            settled_idx = min(1, len(windows) - 1)
            results["iq_window"] = windows[settled_idx]   # (1024, 2)

            # ── 5. Demodulation (GNU Radio) ───────────────────────────
            self.progress.emit(60, "Demodulating...")
            decoded_bits = np.array([], dtype=np.uint8)
            sps = 8
            try:
                from demod import demodulate, estimate_samples_per_symbol
                all_iq = windows.reshape(-1, 2)
                obw = results.get("occupied_bandwidth_hz")
                baud_hint = (obw / 1.35) if (obw is not None and obw > 1000) else None
                try:
                    cs_sps = complex_samples[:16384] if len(complex_samples) > 16384 else complex_samples
                    sps = estimate_samples_per_symbol(cs_sps, float(sr), baud_rate_hint=baud_hint)
                except Exception:
                    sps = 8
                decoded_bits = demodulate(
                    all_iq,
                    modulation_class=ml_result["class"],
                    sample_rate=float(sr),
                    samples_per_symbol=sps,
                    baud_rate_hint=baud_hint,
                )
            except ImportError as e:
                results["demod_error"] = f"GNU Radio not available: {e}"
            except Exception as e:
                results["demod_error"] = f"Demodulation error: {e}"

            results["sps"] = sps
            results["decoded_bits_raw"] = decoded_bits

            # ── 6. Frame sync & polarity resolution ────────────────────
            self.progress.emit(70, "Synchronizing frame & resolving polarity...")
            header_start = 0
            payload_start = 0
            aligned_bits = decoded_bits.copy()

            if len(decoded_bits) > 32:
                try:
                    from sync_correlator import find_preamble
                    # Test normal and inverted polarity (Costas phase flip)
                    corr_norm = find_preamble(decoded_bits)
                    corr_inv  = find_preamble(1 - decoded_bits)

                    norm_score = corr_norm.get("confidence", 0.0) if corr_norm.get("found") else 0.0
                    inv_score  = corr_inv.get("confidence", 0.0) if corr_inv.get("found") else 0.0

                    if inv_score > norm_score and inv_score >= 0.70:
                        corr = corr_inv
                        decoded_bits = 1 - decoded_bits
                        results["phase_inverted"] = True
                    else:
                        corr = corr_norm
                        results["phase_inverted"] = False

                    results["preamble"] = corr
                    if corr.get("found") and corr.get("confidence", 0) >= 0.70:
                        header_start  = corr["header_start"]
                        payload_start = corr["payload_start"]
                        aligned_bits  = decoded_bits[payload_start:]
                    else:
                        aligned_bits  = decoded_bits
                except Exception as e:
                    results["corr_error"] = str(e)
                    aligned_bits = decoded_bits

            results["header_offset"]  = header_start
            results["payload_offset"] = payload_start

            # ── 7. De-interleaving ────────────────────────────────────
            self.progress.emit(80, "De-interleaving...")
            deinterleaved_bits = aligned_bits.copy()
            di_result = {"method": "none", "params": {}, "score": 0.0, "bits": aligned_bits}

            if len(aligned_bits) > 64:
                try:
                    from deinterleave import try_all_deinterleavers
                    from fec_decode import fast_fec_score

                    di_result = try_all_deinterleavers(aligned_bits, fast_fec_score)
                    deinterleaved_bits = di_result["bits"]
                except Exception as e:
                    results["deinterleave_error"] = str(e)

            results["deinterleaver"] = _fmt_di(di_result)

            # ── 8. FEC decoding ───────────────────────────────────────
            self.progress.emit(90, "FEC decoding...")
            fec_result = {"method": "none", "score": 0.0,
                          "decoded_bits": deinterleaved_bits,
                          "decoded_bytes": bits_to_bytes_safe(deinterleaved_bits),
                          "params": {}}

            if len(deinterleaved_bits) > 16:
                try:
                    from fec_decode import try_all_fec
                    fec_result = try_all_fec(deinterleaved_bits)
                except Exception as e:
                    results["fec_error"] = str(e)

            results["fec_scheme"] = _fmt_fec(fec_result)
            results["fec_score"]  = fec_result["score"]
            results["decoded_bytes"]  = len(fec_result.get("decoded_bytes", b""))
            results["final_bits"] = fec_result.get("decoded_bits", deinterleaved_bits)
            results["final_bytes"] = fec_result.get("decoded_bytes", b"")

            self.progress.emit(100, "Done.")
            self.finished.emit(results)

        except Exception:
            self.error.emit(traceback.format_exc())


def bits_to_bytes_safe(bits: np.ndarray) -> bytes:
    if len(bits) == 0:
        return b""
    b = np.asarray(bits, dtype=np.uint8)
    pad = (8 - len(b) % 8) % 8
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(b, bitorder='big').tobytes()


def _fmt_di(di_result: dict) -> str:
    m = di_result.get("method", "none")
    p = di_result.get("params", {})
    if m == "block":
        return f"block ({p.get('rows','?')}x{p.get('cols','?')})"
    if m == "convolutional":
        return f"conv (depth={p.get('depth','?')})"
    if m == "pseudo_random":
        return f"PR (seed={p.get('seed','?')})"
    return "none"


def _fmt_fec(fec_result: dict) -> str:
    m = fec_result.get("method", "none")
    p = fec_result.get("params", {})
    if m == "viterbi":
        return f"Viterbi k={p.get('constraint','?')}"
    if m == "reed_solomon":
        return f"RS nsym={p.get('nsym','?')}"
    return "none"


# ---------------------------------------------------------------------------
# Detached Window (Pop-out detail view)
# ---------------------------------------------------------------------------

class DetachedWindow(QDialog):
    """Floating window to view any diagram in dedicated high resolution."""

    def __init__(self, title: str, widget: QWidget, on_close_callback, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"SIH Signal Analyzer — {title}")
        self.resize(1150, 750)
        self.setMinimumSize(700, 450)
        self.setStyleSheet(_STYLE)
        self._widget = widget
        self._on_close_callback = on_close_callback

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Top bar
        top_bar = QHBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 11pt; font-weight: bold; color: #58a6ff;")
        top_bar.addWidget(title_lbl)
        top_bar.addStretch(1)

        redock_btn = QPushButton("📥 Re-dock to Main Window")
        redock_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                color: #58a6ff;
                border: 1px solid #388bfd;
                border-radius: 6px;
                padding: 5px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #388bfd33;
            }
        """)
        redock_btn.clicked.connect(self.close)
        top_bar.addWidget(redock_btn)
        layout.addLayout(top_bar)

        layout.addWidget(widget)

    def closeEvent(self, event):
        self._on_close_callback(self._widget)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

_STYLE = """
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
}
QMenu::item:selected {
    background-color: #21262d;
    color: #58a6ff;
}
"""


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIH Signal Analyzer")
        self.setMinimumSize(1280, 720)
        self.resize(1440, 860)
        self.setStyleSheet(_STYLE)

        self._worker: AnalyzerWorker | None = None
        self._thread: QThread | None = None
        self._current_file: str = ""

        self._active_detached: dict[str, DetachedWindow] = {}
        self._grid_slots: dict[str, QVBoxLayout] = {}
        self._tab_slots: dict[str, QVBoxLayout] = {}

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        open_action = QAction("&Open (.wav / .iq)...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        synth_action = QAction("&Generate Synthetic IQ...", self)
        synth_action.setShortcut("Ctrl+G")
        synth_action.triggered.connect(self._open_synthetic)
        file_menu.addAction(synth_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = mb.addMenu("&View")
        grid_action = QAction("Overview (2x2 Grid)", self)
        grid_action.setShortcut("Ctrl+1")
        grid_action.triggered.connect(lambda: self._tab_widget.setCurrentIndex(0))
        view_menu.addAction(grid_action)

        wave_action = QAction("Waveform", self)
        wave_action.setShortcut("Ctrl+2")
        wave_action.triggered.connect(lambda: self._tab_widget.setCurrentIndex(1))
        view_menu.addAction(wave_action)

        spec_action = QAction("Spectrogram & PSD", self)
        spec_action.setShortcut("Ctrl+3")
        spec_action.triggered.connect(lambda: self._tab_widget.setCurrentIndex(2))
        view_menu.addAction(spec_action)

        const_action = QAction("IQ Constellation", self)
        const_action.setShortcut("Ctrl+4")
        const_action.triggered.connect(lambda: self._tab_widget.setCurrentIndex(3))
        view_menu.addAction(const_action)

        bits_action = QAction("Decoded Bitstream", self)
        bits_action.setShortcut("Ctrl+5")
        bits_action.triggered.connect(lambda: self._tab_widget.setCurrentIndex(4))
        view_menu.addAction(bits_action)

        view_menu.addSeparator()
        sidebar_action = QAction("Toggle Parameters Sidebar", self)
        sidebar_action.setShortcut("Ctrl+B")
        sidebar_action.triggered.connect(self._toggle_sidebar)
        view_menu.addAction(sidebar_action)

        help_menu = mb.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._open_btn = QAction("Open File", self)
        self._open_btn.triggered.connect(self._open_file)
        tb.addAction(self._open_btn)

        self._synth_btn = QAction("Generate Synthetic IQ", self)
        self._synth_btn.triggered.connect(self._open_synthetic)
        tb.addAction(self._synth_btn)

        tb.addSeparator()

        self._analyze_btn = QAction("Analyze", self)
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.triggered.connect(self._run_analysis)
        tb.addAction(self._analyze_btn)

        tb.addSeparator()

        self._sidebar_btn = QAction("Hide Sidebar", self)
        self._sidebar_btn.setToolTip("Toggle parameters sidebar to maximize diagram space")
        self._sidebar_btn.triggered.connect(self._toggle_sidebar)
        tb.addAction(self._sidebar_btn)

        tb.addSeparator()

        self._file_label = QLabel("  No file loaded.")
        self._file_label.setStyleSheet("color: #8b949e; font-size: 9pt;")
        tb.addWidget(self._file_label)

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Main splitter: Left (Tabbed diagrams) + Right (ParamsPanel)
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setHandleWidth(6)

        self._tab_widget = QTabWidget()
        self._tab_widget.setDocumentMode(True)

        # Instantiate panels
        self._waveform_panel = WaveformPanel()
        self._spectrogram_panel = SpectrogramPanel()
        self._constellation_panel = ConstellationPanel()
        self._bitstream_panel = BitstreamPanel()

        self._panels = {
            "wave": self._waveform_panel,
            "spec": self._spectrogram_panel,
            "const": self._constellation_panel,
            "bits": self._bitstream_panel,
        }

        # ── Tab 0: ⊞ Overview (2x2 Grid) ───────────────────────────
        page_grid = QWidget()
        grid_vbox = QVBoxLayout(page_grid)
        grid_vbox.setContentsMargins(2, 2, 2, 2)
        grid_vbox.setSpacing(4)

        self._top_grid_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._bot_grid_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._v_grid_splitter = QSplitter(Qt.Orientation.Vertical)
        self._v_grid_splitter.setHandleWidth(6)
        self._top_grid_splitter.setHandleWidth(6)
        self._bot_grid_splitter.setHandleWidth(6)

        configs = [
            ("wave", "Time Waveform", self._top_grid_splitter, 1),
            ("spec", "Spectrogram & PSD", self._top_grid_splitter, 2),
            ("const", "IQ Constellation", self._bot_grid_splitter, 3),
            ("bits", "Decoded Bitstream", self._bot_grid_splitter, 4),
        ]

        for key, title, parent_splitter, tab_idx in configs:
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 6px;
                }
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(6, 6, 6, 6)
            card_layout.setSpacing(4)

            # Clean header row (no view in detail or pop out buttons)
            header_bar = QHBoxLayout()
            header_bar.setContentsMargins(4, 2, 4, 4)
            lbl = QLabel(title)
            lbl.setStyleSheet("font-size: 9.5pt; font-weight: 600; color: #58a6ff; border: none;")
            header_bar.addWidget(lbl)
            header_bar.addStretch(1)

            card_layout.addLayout(header_bar)

            # Slot layout
            slot = QVBoxLayout()
            slot.setContentsMargins(0, 0, 0, 0)
            card_layout.addLayout(slot, 1)
            self._grid_slots[key] = slot

            parent_splitter.addWidget(card)

        self._v_grid_splitter.addWidget(self._top_grid_splitter)
        self._v_grid_splitter.addWidget(self._bot_grid_splitter)
        self._v_grid_splitter.setSizes([380, 380])

        self._top_grid_splitter.setChildrenCollapsible(False)
        self._bot_grid_splitter.setChildrenCollapsible(False)
        self._top_grid_splitter.setStretchFactor(0, 1)
        self._top_grid_splitter.setStretchFactor(1, 1)
        self._bot_grid_splitter.setStretchFactor(0, 1)
        self._bot_grid_splitter.setStretchFactor(1, 1)
        self._top_grid_splitter.setSizes([520, 520])
        self._bot_grid_splitter.setSizes([520, 520])
        grid_vbox.addWidget(self._v_grid_splitter)

        self._tab_widget.addTab(page_grid, "Overview (2x2 Grid)")

        # ── Tabs 1-4: Dedicated Full-Size Views ────────────────────
        tab_defs = [
            ("wave", "Waveform", "Time-Domain Waveform"),
            ("spec", "Spectrogram & PSD", "Spectrogram Waterfall & Power Spectral Density (PSD)"),
            ("const", "Constellation", "In-Phase / Quadrature (IQ) Constellation Diagram"),
            ("bits", "Bitstream", "Decoded Bitstream & Frame Sync Markers"),
        ]

        for key, tab_label, full_title in tab_defs:
            tab_page = QWidget()
            tab_page_layout = QVBoxLayout(tab_page)
            tab_page_layout.setContentsMargins(6, 6, 6, 6)
            tab_page_layout.setSpacing(4)

            # Banner bar
            banner = QHBoxLayout()
            banner.setContentsMargins(4, 2, 4, 4)
            b_lbl = QLabel(full_title)
            b_lbl.setStyleSheet("font-size: 10pt; font-weight: 600; color: #58a6ff;")
            banner.addWidget(b_lbl)
            banner.addStretch(1)

            back_btn = QPushButton("Back to Overview")
            back_btn.setToolTip("Return to 2x2 multi-panel grid")
            back_btn.clicked.connect(lambda: self._tab_widget.setCurrentIndex(0))
            banner.addWidget(back_btn)

            tab_page_layout.addLayout(banner)

            slot = QVBoxLayout()
            slot.setContentsMargins(0, 0, 0, 0)
            tab_page_layout.addLayout(slot, 1)
            self._tab_slots[key] = slot

            self._tab_widget.addTab(tab_page, tab_label)

        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        # Place panels in Overview initially
        self._refresh_panel_placement(0)

        # ── Right: parameters panel ───────────────────────────────────
        self._params_panel = ParamsPanel()
        self._params_panel.setMaximumWidth(380)

        self._main_splitter.addWidget(self._tab_widget)
        self._main_splitter.addWidget(self._params_panel)
        self._main_splitter.setSizes([1050, 310])

        layout.addWidget(self._main_splitter)

    def _refresh_panel_placement(self, current_tab_idx: int | None = None):
        if current_tab_idx is None:
            current_tab_idx = self._tab_widget.currentIndex()

        tab_key_map = {1: "wave", 2: "spec", 3: "const", 4: "bits"}

        if current_tab_idx == 0:
            for key, panel in self._panels.items():
                if key not in self._active_detached:
                    self._grid_slots[key].addWidget(panel)
                    panel.show()
        else:
            focused_key = tab_key_map.get(current_tab_idx)
            if focused_key and focused_key not in self._active_detached:
                panel = self._panels[focused_key]
                self._tab_slots[focused_key].addWidget(panel)
                panel.show()

    def _on_tab_changed(self, index: int):
        self._refresh_panel_placement(index)

    def _pop_out_panel(self, key: str):
        if key in self._active_detached:
            dlg = self._active_detached[key]
            dlg.raise_()
            dlg.activateWindow()
            return

        panel = self._panels[key]
        title_map = {
            "wave": "Waveform (Time Domain)",
            "spec": "Spectrogram & Power Spectral Density",
            "const": "IQ Constellation Diagram",
            "bits": "Decoded Bitstream & Frame Analysis",
        }
        title = title_map.get(key, "Signal Diagram")

        dlg = DetachedWindow(title, panel, lambda w, k=key: self._on_panel_redocked(k), parent=self)
        self._active_detached[key] = dlg
        dlg.show()

    def _on_panel_redocked(self, key: str):
        if key in self._active_detached:
            del self._active_detached[key]
        self._refresh_panel_placement()

    def _toggle_sidebar(self):
        vis = not self._params_panel.isVisible()
        self._params_panel.setVisible(vis)
        self._sidebar_btn.setText("Hide Sidebar" if vis else "Show Sidebar")

    def showEvent(self, event):
        super().showEvent(event)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(80, self._balance_grid_splitters)

    def _balance_grid_splitters(self):
        w = self._top_grid_splitter.width()
        if w > 200:
            half = w // 2
            self._top_grid_splitter.setSizes([half, half])
            self._bot_grid_splitter.setSizes([half, half])

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._status_label = QLabel("Ready.")
        self._status_label.setStyleSheet("font-size: 8pt; color: #8b949e;")
        sb.addWidget(self._status_label, 1)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(200)
        self._progress.setVisible(False)
        sb.addPermanentWidget(self._progress)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open RF Capture File",
            "",
            "RF Captures (*.wav *.iq);;WAV files (*.wav);;IQ files (*.iq);;All files (*)",
        )
        if not path:
            return
        self._load_path(path)

    def _open_synthetic(self):
        """Show the synthetic IQ generator dialog, then auto-load the result."""
        from gui.synthetic_dialog import SyntheticDialog
        default_dir = os.path.dirname(self._current_file) if self._current_file else os.getcwd()
        dlg = SyntheticDialog(self, default_output_dir=default_dir)
        if dlg.exec() == SyntheticDialog.DialogCode.Accepted and dlg.result_path:
            meta = dlg.result_meta
            self._status_label.setText(
                f"Generated: {meta.get('modulation')} | "
                f"{meta.get('n_symbols')} symbols | "
                f"SNR {meta.get('snr_db'):.0f} dB | "
                f"{meta.get('sample_rate')/1e3:.0f} kHz"
            )
            self._load_path(dlg.result_path)

    def _load_path(self, path: str):
        """Set the current file path and trigger analysis."""
        self._current_file = path
        fname = os.path.basename(path)
        self._file_label.setText(f"  {fname}")
        self._analyze_btn.setEnabled(True)
        self._run_analysis()

    def _run_analysis(self):
        if not self._current_file:
            return
        if self._thread and self._thread.isRunning():
            return

        self._analyze_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Analyzing...")

        self._worker = AnalyzerWorker(self._current_file)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self._analyze_btn.setEnabled(True))

        self._thread.start()

    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._status_label.setText(msg)

    def _on_finished(self, results: dict):
        self._progress.setVisible(False)
        self._progress.setValue(0)
        self._status_label.setText(
            f"Analysis complete — {results.get('modulation','?')} "
            f"({results.get('confidence', 0):.1%} confidence)"
        )

        # ── Waveform panel ─────────────────────────────────────────
        if "waveform" in results:
            self._waveform_panel.set_data(results["waveform"])

        # ── Spectrogram panel ──────────────────────────────────────
        if "spectral" in results:
            self._spectrogram_panel.set_data(results["spectral"])

        # ── Constellation panel ────────────────────────────────────
        if "iq_window" in results:
            self._constellation_panel.set_data(
                results["iq_window"],
                results.get("modulation", ""),
            )

        # ── Params panel ───────────────────────────────────────────
        self._params_panel.set_params({
            "modulation":            results.get("modulation"),
            "confidence":            results.get("confidence"),
            "all_probs":             results.get("all_probs", {}),
            "sample_rate":           results.get("sample_rate"),
            "occupied_bandwidth_hz": results.get("occupied_bandwidth_hz"),
            "center_freq_offset_hz": results.get("center_freq_offset_hz"),
            "deinterleaver":         results.get("deinterleaver", "none"),
            "fec_scheme":            results.get("fec_scheme", "none"),
            "fec_score":             results.get("fec_score"),
            "decoded_bytes":         results.get("decoded_bytes"),
            "header_offset":         results.get("header_offset"),
            "payload_offset":        results.get("payload_offset"),
        })

        # ── Bitstream panel ────────────────────────────────────────
        final_bits = results.get("final_bits", np.array([], dtype=np.uint8))
        self._bitstream_panel.set_data(
            final_bits,
            header_end_bit=results.get("payload_offset", 0),
        )

        # Show any non-critical errors in status bar
        errs = [
            results.get("demod_error"),
            results.get("deinterleave_error"),
            results.get("fec_error"),
            results.get("corr_error"),
        ]
        errs = [e for e in errs if e]
        if errs:
            self._status_label.setText(
                f"Analysis complete (with warnings): {errs[0]}"
            )

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._analyze_btn.setEnabled(True)
        self._status_label.setText("Analysis failed — see error dialog.")
        QMessageBox.critical(self, "Analysis Error", msg)

    def _show_about(self):
        QMessageBox.about(
            self,
            "SIH Signal Analyzer",
            "<b>SIH Signal Analyzer</b><br><br>"
            "Automated RF signal analysis pipeline.<br>"
            "Modulation: BPSK / QPSK / GMSK (via ML)<br>"
            "Demodulation: GNU Radio gr-digital<br>"
            "FEC: Viterbi (commpy) + Reed-Solomon (reedsolo)<br>"
            "De-interleave: Block / Convolutional / Pseudo-Random<br><br>"
            "Built for SIH 2024."
        )
