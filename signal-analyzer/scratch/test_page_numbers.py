"""
scratch/test_page_numbers.py
"""

import os
import sys

_PROJECT_ROOT = r"c:\Sanchar\signal-analyzer"
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

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QMarginsF, QRectF, QSizeF, Qt
from PyQt6.QtGui import QColor, QFont, QPageLayout, QPageSize, QPainter, QPdfWriter, QPen, QTextDocument
import numpy as np

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from file_loader import load_file, load_wav
from inference import predict_modulation
from spectral_features import analyze as spectral_analyze
from waveform_view import waveform_display_data
from demod import demodulate, estimate_samples_per_symbol
from sync_correlator import find_preamble
from deinterleave import try_all_deinterleavers
from fec_decode import try_all_fec, fast_fec_score
from gui.report_generator import build_html_report, capture_all_plots

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

test_wav = r"c:\Sanchar\signal-analyzer\synthetic_BPSK_1000kHz_SNR20dB.wav"
windows, meta = load_file(test_wav)
complex_samples, sample_rate = load_wav(test_wav)
sr = sample_rate or meta.get("sample_rate") or 1_000_000

ml_result = predict_modulation(windows)
spectral = spectral_analyze(complex_samples, sr)
waveform = waveform_display_data(complex_samples, max_points=32768)
settled_idx = min(1, len(windows) - 1)
iq_window = windows[settled_idx]

obw = spectral["occupied_bandwidth_hz"]
baud_hint = (obw / 1.35) if (obw is not None and obw > 1000) else None
try:
    cs_sps = complex_samples[:16384] if len(complex_samples) > 16384 else complex_samples
    sps = estimate_samples_per_symbol(cs_sps, float(sr), baud_rate_hint=baud_hint)
except Exception:
    sps = 8

all_iq = windows.reshape(-1, 2)
decoded_bits = demodulate(
    all_iq,
    modulation_class=ml_result["class"],
    sample_rate=float(sr),
    samples_per_symbol=sps,
    baud_rate_hint=baud_hint,
)

header_start = 0
payload_start = 0
aligned_bits = decoded_bits.copy()
phase_inverted = False

if len(decoded_bits) > 32:
    corr_norm = find_preamble(decoded_bits)
    corr_inv = find_preamble(1 - decoded_bits)
    norm_score = corr_norm.get("confidence", 0.0) if corr_norm.get("found") else 0.0
    inv_score = corr_inv.get("confidence", 0.0) if corr_inv.get("found") else 0.0

    if inv_score > norm_score and inv_score >= 0.70:
        corr = corr_inv
        decoded_bits = 1 - decoded_bits
        phase_inverted = True
    else:
        corr = corr_norm

    if corr.get("found") and corr.get("confidence", 0) >= 0.70:
        header_start = corr["header_start"]
        payload_start = corr["payload_start"]
        aligned_bits = decoded_bits[payload_start:]
    else:
        aligned_bits = decoded_bits
else:
    corr = {"found": False}

di_result = try_all_deinterleavers(aligned_bits, fast_fec_score) if len(aligned_bits) > 64 else {"method": "none", "params": {}, "bits": aligned_bits}
deinterleaved_bits = di_result["bits"]

fec_result = try_all_fec(deinterleaved_bits) if len(deinterleaved_bits) > 16 else {"method": "none", "score": 0.0, "decoded_bits": deinterleaved_bits, "decoded_bytes": bits_to_bytes_safe(deinterleaved_bits)}

results = {
    "meta": meta,
    "sample_rate": sr,
    "modulation": ml_result["class"],
    "confidence": ml_result["confidence"],
    "all_probs": ml_result["all_probs"],
    "spectral": spectral,
    "occupied_bandwidth_hz": obw,
    "center_freq_offset_hz": spectral["center_freq_offset_hz"],
    "waveform": waveform,
    "iq_window": iq_window,
    "sps": sps,
    "decoded_bits_raw": decoded_bits,
    "preamble": corr,
    "header_offset": header_start,
    "payload_offset": payload_start,
    "phase_inverted": phase_inverted,
    "deinterleaver": _fmt_di(di_result),
    "fec_scheme": _fmt_fec(fec_result),
    "fec_score": fec_result["score"],
    "decoded_bytes": len(fec_result.get("decoded_bytes", b"")),
    "final_bits": fec_result.get("decoded_bits", deinterleaved_bits),
    "final_bytes": fec_result.get("decoded_bytes", b""),
}

plot_images = capture_all_plots(results)
html_content = build_html_report(results, test_wav, plot_images)

pdf_out = r"c:\Sanchar\signal-analyzer\signal_analysis_report_numbered.pdf"
os.makedirs(os.path.dirname(os.path.abspath(pdf_out)), exist_ok=True)

doc = QTextDocument()
doc.setHtml(html_content)

writer = QPdfWriter(pdf_out)
writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
writer.setPageOrientation(QPageLayout.Orientation.Portrait)
writer.setResolution(96)

layout = QPageLayout(
    QPageSize(QPageSize.PageSizeId.A4),
    QPageLayout.Orientation.Portrait,
    QMarginsF(10, 10, 10, 10),
    QPageLayout.Unit.Millimeter,
)
writer.setPageLayout(layout)

paint_rect = writer.pageLayout().paintRectPixels(96)
w = paint_rect.width()
h = paint_rect.height()
doc.setPageSize(QSizeF(w, h - 22))

page_count = doc.pageCount()
print(f"Document Page Count: {page_count}")

painter = QPainter(writer)
for page_idx in range(page_count):
    if page_idx > 0:
        writer.newPage()

    painter.save()
    painter.translate(0, -page_idx * (h - 22))
    doc.drawContents(painter, QRectF(0, page_idx * (h - 22), w, h - 22))
    painter.restore()

    # Running footer
    painter.save()
    painter.setFont(QFont("sans-serif", 7))
    painter.setPen(QPen(QColor("#64748b"), 1))
    footer_y = h - 6
    painter.drawLine(0, int(footer_y - 12), int(w), int(footer_y - 12))
    painter.drawText(QRectF(0, footer_y - 10, w * 0.45, 12), Qt.AlignmentFlag.AlignLeft, "SIH Signal Analyzer • Technical Report")
    painter.drawText(QRectF(w * 0.35, footer_y - 10, w * 0.30, 12), Qt.AlignmentFlag.AlignCenter, "National Technical Research Organisation (NTRO)")
    painter.drawText(QRectF(w * 0.75, footer_y - 10, w * 0.25, 12), Qt.AlignmentFlag.AlignRight, f"Page {page_idx + 1} of {page_count}")
    painter.restore()

painter.end()

print(f"Generated PDF with page numbers: {pdf_out} ({os.path.getsize(pdf_out):,} bytes)")
