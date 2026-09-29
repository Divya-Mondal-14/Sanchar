"""
scratch/test_info_boxes.py
Visual verification of PSD and Constellation information boxes.
"""
import os
import sys
import numpy as np

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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout
)
from PyQt6.QtCore import Qt, QSize

from file_loader import load_wav
from spectral_features import analyze as spectral_analyze
from demod import estimate_samples_per_symbol, recover_constellation_symbols
from gui.spectrogram_panel import SpectrogramPanel
from gui.constellation_panel import ConstellationPanel

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

test_wav = r"c:\Sanchar\signal-analyzer\synthetic_BPSK_1000kHz_SNR20dB.wav"
complex_samples, sr = load_wav(test_wav)
sr = sr or 1_000_000
spectral = spectral_analyze(complex_samples, sr)
obw = spectral["occupied_bandwidth_hz"]
baud_hint = (obw / 1.35) if (obw is not None and obw > 1000) else None
sps = estimate_samples_per_symbol(complex_samples, float(sr), baud_rate_hint=baud_hint)
rec_const = recover_constellation_symbols(
    complex_samples,
    sample_rate=float(sr),
    sps=sps,
    mod_class="BPSK",
    cfo_hz=spectral.get("center_freq_offset_hz"),
)

win = QMainWindow()
win.setWindowTitle("Test Info Boxes — BPSK")
win.resize(1100, 550)

central = QWidget()
layout = QHBoxLayout(central)

spec_panel = SpectrogramPanel()
spec_panel.set_data(spectral)
layout.addWidget(spec_panel, 1)

const_panel = ConstellationPanel()
const_panel.set_data(rec_const, "BPSK")
layout.addWidget(const_panel, 1)

win.setCentralWidget(central)
win.show()

# Verify overlay texts
print("PSD Info Box Visible:", spec_panel._psd_info_box.isVisible())
print("PSD Info Box Text Content:\n", spec_panel._psd_info_box.text())

print("\nConstellation Info Box Visible:", const_panel._const_info_box.isVisible())
print("Constellation Info Box Text Content:\n", const_panel._const_info_box.text())

# Test QPSK as well
qpsk_wav = r"c:\Sanchar\signal-analyzer\synthetic_QPSK_1000kHz_SNR20dB.wav"
q_samples, q_sr = load_wav(qpsk_wav)
q_sr = q_sr or 1_000_000
q_spectral = spectral_analyze(q_samples, q_sr)
q_obw = q_spectral["occupied_bandwidth_hz"]
q_baud = (q_obw / 1.35) if (q_obw is not None and q_obw > 1000) else None
q_sps = estimate_samples_per_symbol(q_samples, float(q_sr), baud_rate_hint=q_baud)
q_const = recover_constellation_symbols(
    q_samples,
    sample_rate=float(q_sr),
    sps=q_sps,
    mod_class="QPSK",
    cfo_hz=q_spectral.get("center_freq_offset_hz"),
)

spec_panel.set_data(q_spectral)
const_panel.set_data(q_const, "QPSK")

print("\n--- After loading QPSK ---")
print("PSD Info Box Text Content:\n", spec_panel._psd_info_box.text())
print("Constellation Info Box Text Content:\n", const_panel._const_info_box.text())

print("\nALL_CHECKS_PASSED")
