"""
gui/report_generator.py

Comprehensive, Print-Ready Signal Analysis Report Generator for SIH Signal Analyzer.

Generates self-contained HTML and high-resolution A4 PDF engineering reports
from actual live pipeline telemetry and analysis diagram renderings.
Follows formal technical publication standards with a clean white background,
restrained color palette, and official NTRO header integration.
"""

from __future__ import annotations

import base64
from datetime import datetime
import os
import sys
from typing import Any, Mapping

import numpy as np

from PyQt6.QtCore import QBuffer, QIODevice, QMarginsF, QPointF, QRectF, QSizeF, Qt
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QImage, QPageLayout, QPageSize,
    QPainter, QPdfWriter, QPen, QPolygonF, QTextDocument
)
from PyQt6.QtWidgets import QApplication, QWidget


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _hz_fmt(hz: float | None) -> str:
    """Format frequency in Hz, kHz, or MHz with clean engineering notation."""
    if hz is None:
        return "Not available"
    abs_hz = abs(hz)
    if abs_hz >= 1e6:
        return f"{hz / 1e6:.4f} MHz ({hz:,.1f} Hz)"
    if abs_hz >= 1e3:
        return f"{hz / 1e3:.3f} kHz ({hz:,.1f} Hz)"
    return f"{hz:.2f} Hz"


def _time_fmt(seconds: float | None) -> str:
    """Format duration in seconds, milliseconds, or microseconds."""
    if seconds is None:
        return "Not available"
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms ({seconds:.5f} s)"
    return f"{seconds:.3f} s"


def _bytes_to_hexdump(data: bytes, max_bytes: int = 32) -> str:
    """Format bytes into standard 16-byte hex dump with ASCII sidebar."""
    if not data:
        return "No decoded bytes available."

    lines = []
    chunk_data = data[:max_bytes]
    for offset in range(0, len(chunk_data), 16):
        row = chunk_data[offset : offset + 16]
        hex_row_1 = " ".join(f"{b:02X}" for b in row[:8])
        hex_row_2 = " ".join(f"{b:02X}" for b in row[8:])
        hex_part = f"{hex_row_1:<23}  {hex_row_2:<23}".rstrip()
        asc_part = "".join(chr(b) if 0x20 <= b <= 0x7E else "." for b in row)
        lines.append(f"{offset:06X}   {hex_part:<48}   |{asc_part}|")

    if len(data) > max_bytes:
        lines.append(f"... ({len(data) - max_bytes} additional bytes omitted)")

    return "\n".join(lines)


def _extract_ascii(data: bytes, max_len: int = 256) -> str:
    """Extract printable ASCII text from byte stream."""
    if not data:
        return "Not available"
    preview = "".join(chr(b) if 0x20 <= b <= 0x7E else "." for b in data[:max_len])
    if len(data) > max_len:
        preview += "..."
    return preview


def _format_bits_summary(bits: np.ndarray, max_preview_bits: int = 96) -> dict[str, str]:
    """Generate statistical summary and formatted octet string of bitstream."""
    if bits is None or len(bits) == 0:
        return {
            "total_bits": "0",
            "ones_count": "0",
            "zeros_count": "0",
            "bit_density": "Not available",
            "preview": "Not available",
        }

    b = np.asarray(bits, dtype=np.uint8).ravel()
    n_ones = int(np.sum(b == 1))
    n_zeros = int(np.sum(b == 0))
    total = len(b)
    density = (n_ones / total) * 100.0 if total > 0 else 0.0

    preview_b = b[:max_preview_bits]
    groups = []
    for i in range(0, len(preview_b), 8):
        byte_bits = "".join(str(bit) for bit in preview_b[i : i + 8])
        groups.append(byte_bits)

    preview_str = " ".join(groups)
    if total > max_preview_bits:
        preview_str += f" ... (+{total - max_preview_bits} bits)"

    return {
        "total_bits": f"{total:,}",
        "ones_count": f"{n_ones:,} ({density:.1f}%)",
        "zeros_count": f"{n_zeros:,} ({100.0 - density:.1f}%)",
        "bit_density": f"{density:.2f}% Ones",
        "preview": preview_str,
    }


# ---------------------------------------------------------------------------
# NTRO Logo Loader
# ---------------------------------------------------------------------------

def _load_ntro_logo_b64() -> str | None:
    """
    Search and load the official NTRO logo image from the assets directory
    and return as an embedded base64 data URI.
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "ntro_logo.png"),
        os.path.join("assets", "ntro_logo.png"),
        r"C:\Sanchar\signal-analyzer\assets\ntro_logo.png",
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    data = f.read()
                    if data:
                        b64 = base64.b64encode(data).decode("ascii")
                        return f"data:image/png;base64,{b64}"
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Native Print-Ready Plot Renderers (High Resolution, Pure White Background)
# ---------------------------------------------------------------------------

def _image_to_base64(img: QImage) -> str:
    """Save a QImage into an embedded base64 PNG data URI string."""
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    b64 = base64.b64encode(bytes(buf.data())).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_waveform_plot(results: dict[str, Any], w: int = 700, h: int = 140) -> str | None:
    """Render a clean, publication-ready time-domain waveform with I/Q traces."""
    wf = results.get("waveform")
    if not wf or "i" not in wf or len(wf["i"]) == 0:
        return None
    i_arr = np.asarray(wf["i"], dtype=np.float32)
    q_arr = np.asarray(wf.get("q", np.zeros_like(i_arr)), dtype=np.float32)
    sr = float(results.get("sample_rate") or 1_000_000)

    n_pts = min(len(i_arr), 500)
    i_pts = i_arr[:n_pts]
    q_pts = q_arr[:n_pts]
    t_max_ms = (n_pts / sr) * 1e3

    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(QColor("#ffffff"))

    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    margin_l, margin_r, margin_t, margin_b = 48, 20, 20, 24
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b

    # Background grid
    p.setPen(QPen(QColor("#e2e8f0"), 1, Qt.PenStyle.DashLine))
    for gy in range(5):
        y_val = margin_t + gy * (plot_h / 4)
        p.drawLine(int(margin_l), int(y_val), int(margin_l + plot_w), int(y_val))
    for gx in range(6):
        x_val = margin_l + gx * (plot_w / 5)
        p.drawLine(int(x_val), int(margin_t), int(x_val), int(margin_t + plot_h))

    # Border
    p.setPen(QPen(QColor("#94a3b8"), 1))
    p.drawRect(int(margin_l), int(margin_t), int(plot_w), int(plot_h))

    # Center axis
    zero_y = margin_t + plot_h / 2
    p.setPen(QPen(QColor("#cbd5e1"), 1, Qt.PenStyle.SolidLine))
    p.drawLine(int(margin_l), int(zero_y), int(margin_l + plot_w), int(zero_y))

    # Labels & Ticks
    font = QFont("sans-serif", 7)
    p.setFont(font)
    p.setPen(QPen(QColor("#475569"), 1))

    # Y ticks
    p.drawText(QRectF(0, margin_t - 5, margin_l - 4, 12), Qt.AlignmentFlag.AlignRight, "+1.0")
    p.drawText(QRectF(0, zero_y - 5, margin_l - 4, 12), Qt.AlignmentFlag.AlignRight, "0.0")
    p.drawText(QRectF(0, margin_t + plot_h - 7, margin_l - 4, 12), Qt.AlignmentFlag.AlignRight, "-1.0")

    # X ticks
    for gx in range(6):
        x_val = margin_l + gx * (plot_w / 5)
        t_val = (gx / 5.0) * t_max_ms
        p.drawText(QRectF(x_val - 20, margin_t + plot_h + 3, 40, 12), Qt.AlignmentFlag.AlignCenter, f"{t_val:.2f}")

    p.drawText(QRectF(margin_l, h - 12, plot_w, 12), Qt.AlignmentFlag.AlignCenter, "Time (ms)")

    # Plot I curve (Blue)
    poly_i = QPolygonF()
    for idx in range(n_pts):
        px = margin_l + (idx / (n_pts - 1)) * plot_w
        py = zero_y - (float(i_pts[idx]) * (plot_h / 2.2))
        poly_i.append(QPointF(px, py))

    p.setPen(QPen(QColor("#1d4ed8"), 1.6))
    p.drawPolyline(poly_i)

    # Plot Q curve (Red dashed)
    poly_q = QPolygonF()
    for idx in range(n_pts):
        px = margin_l + (idx / (n_pts - 1)) * plot_w
        py = zero_y - (float(q_pts[idx]) * (plot_h / 2.2))
        poly_q.append(QPointF(px, py))

    p.setPen(QPen(QColor("#dc2626"), 1.2, Qt.PenStyle.DashLine))
    p.drawPolyline(poly_q)

    # Legend
    p.setFont(QFont("sans-serif", 7, QFont.Weight.Bold))
    p.setPen(QPen(QColor("#1d4ed8"), 2))
    p.drawLine(w - 150, 10, w - 135, 10)
    p.setPen(QPen(QColor("#1e293b"), 1))
    p.drawText(w - 130, 13, "In-Phase (I)")

    p.setPen(QPen(QColor("#dc2626"), 2, Qt.PenStyle.DashLine))
    p.drawLine(w - 75, 10, w - 60, 10)
    p.setPen(QPen(QColor("#1e293b"), 1))
    p.drawText(w - 55, 13, "Quadrature (Q)")

    p.end()
    return _image_to_base64(img)


def _render_spectrum_plot(results: dict[str, Any], w: int = 700, h: int = 140) -> str | None:
    """Render a clean, publication-ready Power Spectral Density curve with OBW highlight."""
    sp = results.get("spectral")
    if not sp or "freqs" not in sp or "psd_db" not in sp:
        return None
    freqs = np.asarray(sp["freqs"], dtype=np.float32)
    psd_db = np.asarray(sp["psd_db"], dtype=np.float32)
    if len(freqs) == 0:
        return None

    max_f = np.max(np.abs(freqs))
    scale_f, unit_f = (1e-6, "MHz") if max_f >= 1e6 else (1e-3, "kHz")
    f_min = float(freqs[0]) * scale_f
    f_max = float(freqs[-1]) * scale_f

    min_db = float(np.min(psd_db))
    max_db = float(np.max(psd_db))
    db_range = max(1.0, max_db - min_db)

    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(QColor("#ffffff"))

    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    margin_l, margin_r, margin_t, margin_b = 48, 20, 20, 26
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b

    # Bandwidth highlight if available
    obw = results.get("occupied_bandwidth_hz")
    if obw is not None and obw > 0:
        cfo = (results.get("center_freq_offset_hz") or 0.0) * scale_f
        half_bw = (obw / 2.0) * scale_f
        bx1 = margin_l + ((max(f_min, cfo - half_bw) - f_min) / (f_max - f_min)) * plot_w
        bx2 = margin_l + ((min(f_max, cfo + half_bw) - f_min) / (f_max - f_min)) * plot_w
        p.fillRect(QRectF(bx1, margin_t, bx2 - bx1, plot_h), QColor("#fef3c7"))

    # Grid
    p.setPen(QPen(QColor("#e2e8f0"), 1, Qt.PenStyle.DashLine))
    for gy in range(5):
        y_val = margin_t + gy * (plot_h / 4)
        p.drawLine(int(margin_l), int(y_val), int(margin_l + plot_w), int(y_val))
    for gx in range(6):
        x_val = margin_l + gx * (plot_w / 5)
        p.drawLine(int(x_val), int(margin_t), int(x_val), int(margin_t + plot_h))

    p.setPen(QPen(QColor("#94a3b8"), 1))
    p.drawRect(int(margin_l), int(margin_t), int(plot_w), int(plot_h))

    # Ticks & Labels
    font = QFont("sans-serif", 7)
    p.setFont(font)
    p.setPen(QPen(QColor("#475569"), 1))

    # Y Ticks
    for gy in range(5):
        y_val = margin_t + gy * (plot_h / 4)
        db_val = max_db - (gy / 4.0) * db_range
        p.drawText(QRectF(0, y_val - 5, margin_l - 4, 12), Qt.AlignmentFlag.AlignRight, f"{db_val:.0f}")

    # X Ticks
    for gx in range(6):
        x_val = margin_l + gx * (plot_w / 5)
        f_val = f_min + (gx / 5.0) * (f_max - f_min)
        p.drawText(QRectF(x_val - 25, margin_t + plot_h + 3, 50, 12), Qt.AlignmentFlag.AlignCenter, f"{f_val:.1f}")

    p.drawText(QRectF(margin_l, h - 13, plot_w, 12), Qt.AlignmentFlag.AlignCenter, f"Frequency ({unit_f})")

    # Curve with subtle fill
    poly_fill = QPolygonF()
    poly_fill.append(QPointF(margin_l, margin_t + plot_h))
    poly_line = QPolygonF()

    n_f = len(freqs)
    for i in range(n_f):
        px = margin_l + (i / (n_f - 1)) * plot_w
        norm_y = (float(psd_db[i]) - min_db) / db_range
        py = margin_t + plot_h - norm_y * plot_h
        poly_line.append(QPointF(px, py))
        poly_fill.append(QPointF(px, py))

    poly_fill.append(QPointF(margin_l + plot_w, margin_t + plot_h))

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor("#e0f2fe")))
    p.drawPolygon(poly_fill)

    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor("#0f766e"), 1.6))
    p.drawPolyline(poly_line)

    p.end()
    return _image_to_base64(img)


def _render_spectrogram_plot(results: dict[str, Any], w: int = 700, h: int = 105) -> str | None:
    """Render a clean STFT spectrogram waterfall image with frequency axis and colorbar."""
    sp = results.get("spectral")
    if not sp or "spec_db" not in sp or "spec_times" not in sp or "spec_freqs" not in sp:
        return None
    spec_db = np.asarray(sp["spec_db"], dtype=np.float32)
    spec_times = np.asarray(sp["spec_times"], dtype=np.float32)
    spec_freqs = np.asarray(sp["spec_freqs"], dtype=np.float32)
    if spec_db.size == 0 or len(spec_times) == 0 or len(spec_freqs) == 0:
        return None

    n_f, n_t = spec_db.shape
    min_db = float(np.min(spec_db))
    max_db = float(np.max(spec_db))
    db_range = max(1.0, max_db - min_db)

    # Convert spectrogram 2D matrix into QImage
    raw_img = QImage(n_t, n_f, QImage.Format.Format_RGB32)
    for r in range(n_f):
        row_idx = n_f - 1 - r
        for c in range(n_t):
            val = (float(spec_db[row_idx, c]) - min_db) / db_range
            val = max(0.0, min(1.0, val))
            # Viridis-inspired color mapping
            red = int(255 * (val * 0.9 + 0.1 * (1.0 - val))) if val > 0.6 else int(255 * val * 0.5)
            green = int(255 * (val if val <= 0.8 else 0.8))
            blue = int(255 * (1.0 - val * 0.7))
            raw_img.setPixelColor(c, r, QColor(red, green, blue))

    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(QColor("#ffffff"))

    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    margin_l, margin_r, margin_t, margin_b = 48, 50, 14, 20
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b

    p.drawImage(QRectF(margin_l, margin_t, plot_w, plot_h), raw_img)
    p.setPen(QPen(QColor("#94a3b8"), 1))
    p.drawRect(int(margin_l), int(margin_t), int(plot_w), int(plot_h))

    # Colorbar on right
    cb_x = w - 34
    cb_w = 10
    for cy in range(int(plot_h)):
        c_val = 1.0 - (cy / plot_h)
        red = int(255 * (c_val * 0.9 + 0.1 * (1.0 - c_val))) if c_val > 0.6 else int(255 * c_val * 0.5)
        green = int(255 * (c_val if c_val <= 0.8 else 0.8))
        blue = int(255 * (1.0 - c_val * 0.7))
        p.setPen(QPen(QColor(red, green, blue), 1))
        p.drawLine(int(cb_x), int(margin_t + cy), int(cb_x + cb_w), int(margin_t + cy))
    p.setPen(QPen(QColor("#94a3b8"), 1))
    p.drawRect(int(cb_x), int(margin_t), int(cb_w), int(plot_h))

    # Labels
    font = QFont("sans-serif", 6)
    p.setFont(font)
    p.setPen(QPen(QColor("#475569"), 1))

    max_sf = np.max(np.abs(spec_freqs))
    sf_scale, sf_unit = (1e-6, "MHz") if max_sf >= 1e6 else (1e-3, "kHz")
    p.drawText(QRectF(0, margin_t - 5, margin_l - 4, 12), Qt.AlignmentFlag.AlignRight, f"{float(spec_freqs[-1])*sf_scale:.1f}")
    p.drawText(QRectF(0, margin_t + plot_h - 7, margin_l - 4, 12), Qt.AlignmentFlag.AlignRight, f"{float(spec_freqs[0])*sf_scale:.1f}")

    t_start = float(spec_times[0])
    t_end = float(spec_times[-1])
    for gx in range(5):
        x_val = margin_l + gx * (plot_w / 4)
        t_val = t_start + (gx / 4.0) * (t_end - t_start)
        p.drawText(QRectF(x_val - 20, margin_t + plot_h + 2, 40, 12), Qt.AlignmentFlag.AlignCenter, f"{t_val:.2f}s")

    p.drawText(QRectF(margin_l, h - 10, plot_w, 12), Qt.AlignmentFlag.AlignCenter, "Time (s)")
    p.drawText(QRectF(cb_x - 8, margin_t - 13, 30, 10), Qt.AlignmentFlag.AlignCenter, "dB")

    p.end()
    return _image_to_base64(img)


def _render_constellation_plot(results: dict[str, Any], w: int = 260, h: int = 240) -> str | None:
    """
    Render a clean, calibrated IQ constellation scatter plot with RMS power normalization,
    ideal constellation target overlays, reference rings, and labeled I/Q axes.
    """
    iq_win = results.get("iq_window")
    if iq_win is None:
        return None
    iq_arr = np.asarray(iq_win, dtype=np.float32)
    if iq_arr.ndim != 2 or iq_arr.shape[1] != 2 or len(iq_arr) == 0:
        return None

    # Normalise power so scatter fills the unit circle consistently
    rms = float(np.sqrt(np.mean(iq_arr[:, 0] ** 2 + iq_arr[:, 1] ** 2)))
    if rms > 1e-9:
        iq_norm = iq_arr / rms
    else:
        iq_norm = iq_arr

    # Limit maximum scatter points rendered to keep diagram sharp and fast
    if len(iq_norm) > 1500:
        iq_norm = iq_norm[:1500]

    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(QColor("#ffffff"))

    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    margin_l, margin_r, margin_t, margin_b = 32, 16, 16, 26
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b
    plot_rect = QRectF(margin_l, margin_t, plot_w, plot_h)

    center_x = margin_l + plot_w / 2.0
    center_y = margin_t + plot_h / 2.0
    radius = min(plot_w, plot_h) * 0.42

    # Concentric reference circle (Unit circle r=1.0)
    p.setPen(QPen(QColor("#cbd5e1"), 1, Qt.PenStyle.DashLine))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(center_x, center_y), radius, radius)
    p.drawEllipse(QPointF(center_x, center_y), radius * 0.5, radius * 0.5)

    # Outer border
    p.setPen(QPen(QColor("#94a3b8"), 1))
    p.drawRect(int(margin_l), int(margin_t), int(plot_w), int(plot_h))

    # Crosshairs (I=0, Q=0)
    p.setPen(QPen(QColor("#64748b"), 1, Qt.PenStyle.DashLine))
    p.drawLine(int(margin_l), int(center_y), int(margin_l + plot_w), int(center_y))
    p.drawLine(int(center_x), int(margin_t), int(center_x), int(margin_t + plot_h))

    # Clip points to plot area
    p.setClipRect(plot_rect)

    # Received IQ scatter points
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(37, 99, 235, 140)))
    for pt in iq_norm:
        px = center_x + float(pt[0]) * radius
        py = center_y - float(pt[1]) * radius
        p.drawEllipse(QPointF(px, py), 2.2, 2.2)

    # Ideal constellation reference overlay
    mod_name = str(results.get("modulation", "")).upper()
    ideal_points: list[tuple[float, float]] = []
    if "BPSK" in mod_name:
        ideal_points = [(-1.0, 0.0), (1.0, 0.0)]
    elif "QPSK" in mod_name or "4QAM" in mod_name:
        ideal_points = [(0.707, 0.707), (-0.707, 0.707), (-0.707, -0.707), (0.707, -0.707)]
    elif "GMSK" in mod_name or "8PSK" in mod_name:
        ideal_points = [(1.0, 0.0), (-1.0, 0.0)]

    if ideal_points:
        p.setPen(QPen(QColor("#ea580c"), 1.8))
        p.setBrush(QBrush(QColor(234, 88, 12, 60)))
        for ix, iy in ideal_points:
            ipx = center_x + ix * radius
            ipy = center_y - iy * radius
            p.drawEllipse(QPointF(ipx, ipy), 5.5, 5.5)
            # Center marker cross
            p.drawLine(int(ipx - 3), int(ipy), int(ipx + 3), int(ipy))
            p.drawLine(int(ipx), int(ipy - 3), int(ipx), int(ipy + 3))

    # Disable clipping for axes text
    p.setClipping(False)

    # Ticks & labels
    font = QFont("sans-serif", 6)
    p.setFont(font)
    p.setPen(QPen(QColor("#475569"), 1))

    # Y-axis ticks
    p.drawText(QRectF(0, margin_t - 5, margin_l - 4, 12), Qt.AlignmentFlag.AlignRight, "+1.0")
    p.drawText(QRectF(0, center_y - 5, margin_l - 4, 12), Qt.AlignmentFlag.AlignRight, "0.0")
    p.drawText(QRectF(0, margin_t + plot_h - 7, margin_l - 4, 12), Qt.AlignmentFlag.AlignRight, "-1.0")

    # X-axis ticks
    p.drawText(QRectF(margin_l - 12, margin_t + plot_h + 2, 24, 12), Qt.AlignmentFlag.AlignCenter, "-1.0")
    p.drawText(QRectF(center_x - 12, margin_t + plot_h + 2, 24, 12), Qt.AlignmentFlag.AlignCenter, "0.0")
    p.drawText(QRectF(margin_l + plot_w - 12, margin_t + plot_h + 2, 24, 12), Qt.AlignmentFlag.AlignCenter, "+1.0")

    # Axis Titles
    p.drawText(QRectF(margin_l, h - 11, plot_w, 12), Qt.AlignmentFlag.AlignCenter, "In-Phase (I)")

    p.end()
    return _image_to_base64(img)


def _grab_widget_to_base64(widget: QWidget | None, min_width: int = 800, min_height: int = 340) -> str | None:
    """Grab a live widget canvas as fallback."""
    if widget is None:
        return None
    try:
        pixmap = widget.grab()
        if pixmap.isNull() or pixmap.width() <= 0 or pixmap.height() <= 0:
            return None
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        b64 = base64.b64encode(bytes(buffer.data())).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def capture_all_plots(results: dict[str, Any], panels: Mapping[str, Any] | None = None) -> dict[str, str | None]:
    """
    Render or capture all four core analysis diagrams with high-resolution white backgrounds.
    """
    images: dict[str, str | None] = {
        "waveform": _render_waveform_plot(results),
        "spectrum": _render_spectrum_plot(results),
        "spectrogram": _render_spectrogram_plot(results),
        "constellation": _render_constellation_plot(results),
    }

    # Fallback to widget pixels if numeric rendering was not possible
    if panels:
        wave_panel = panels.get("wave")
        if not images["waveform"] and wave_panel and hasattr(wave_panel, "_plot"):
            images["waveform"] = _grab_widget_to_base64(wave_panel._plot, 850, 320)

        spec_panel = panels.get("spec")
        if spec_panel:
            if not images["spectrogram"] and hasattr(spec_panel, "_waterfall_plot"):
                images["spectrogram"] = _grab_widget_to_base64(spec_panel._waterfall_plot, 850, 320)
            if not images["spectrum"] and hasattr(spec_panel, "_psd_plot"):
                images["spectrum"] = _grab_widget_to_base64(spec_panel._psd_plot, 850, 260)

        const_panel = panels.get("const")
        if not images["constellation"] and const_panel and hasattr(const_panel, "_plot"):
            images["constellation"] = _grab_widget_to_base64(const_panel._plot, 550, 500)

    return images


# ---------------------------------------------------------------------------
# Factual Technical Summary Synthesizer
# ---------------------------------------------------------------------------

def generate_factual_summary(results: dict[str, Any], file_path: str) -> str:
    """
    Generate a concise factual technical summary based strictly on actual pipeline telemetry.
    Never invents, estimates, or hard-codes values.
    """
    fname = os.path.basename(file_path) if file_path else "Unspecified capture"
    meta = results.get("meta", {})
    sr = results.get("sample_rate") or meta.get("sample_rate")
    total_samples = meta.get("total_samples")

    sentences = []

    # 1. Capture description
    if sr and total_samples:
        duration_s = total_samples / sr
        sentences.append(
            f"RF capture file '{fname}' was processed, containing {total_samples:,} samples recorded at a sample rate of {_hz_fmt(sr)} "
            f"(effective duration: {_time_fmt(duration_s)})."
        )
    else:
        sentences.append(f"RF capture file '{fname}' was loaded and evaluated through the automated analysis pipeline.")

    # 2. Modulation classification
    mod = results.get("modulation")
    conf = results.get("confidence")
    if mod:
        conf_str = f"{conf:.2%}" if conf is not None else "Not available"
        sentences.append(
            f"The neural modulation classifier identified the transmission format as {mod} "
            f"with a classification confidence of {conf_str}."
        )

    # 3. Spectral parameters
    obw = results.get("occupied_bandwidth_hz")
    cfo = results.get("center_freq_offset_hz")
    if obw is not None or cfo is not None:
        spectral_parts = []
        if obw is not None:
            spectral_parts.append(f"an occupied bandwidth of {_hz_fmt(obw)}")
        if cfo is not None:
            spectral_parts.append(f"a carrier frequency offset of {_hz_fmt(cfo)}")
        sentences.append(f"Spectral analysis determined {' with '.join(spectral_parts)}.")

    # 4. Demodulation & Frame Sync
    decoded_bits = results.get("decoded_bits_raw")
    preamble = results.get("preamble", {})
    h_off = results.get("header_offset")
    p_off = results.get("payload_offset")

    if decoded_bits is not None and len(decoded_bits) > 0:
        sync_desc = ""
        if preamble.get("found"):
            p_name = preamble.get("pattern_name", "Sync Marker")
            p_conf = preamble.get("confidence", 0.0)
            sync_desc = f" Frame synchronization locked to {p_name} ({p_conf:.0%} match) with header boundary at bit {h_off} and payload commencing at bit {p_off}."
        sentences.append(f"Demodulation recovered {len(decoded_bits):,} raw bit transitions.{sync_desc}")

    # 5. Coding & Protocol
    di = results.get("deinterleaver", "none")
    fec = results.get("fec_scheme", "none")
    n_bytes = results.get("decoded_bytes", 0)

    fec_parts = []
    if str(di).lower() not in ("none", "", "—"):
        fec_parts.append(f"a {di} de-interleaver")
    else:
        fec_parts.append("direct (no) interleaving")

    if str(fec).lower() not in ("none", "", "—"):
        fec_parts.append(f"a {fec} FEC scheme")
    else:
        fec_parts.append("raw uncoded bit framing")

    sentences.append(
        f"Channel decoding detected {', and '.join(fec_parts)}, successfully assembling {n_bytes:,} decoded payload bytes."
    )

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# HTML Report Builder (White A4 Print-Ready Design)
# ---------------------------------------------------------------------------

def build_html_report(
    results: dict[str, Any],
    file_path: str,
    plot_images: dict[str, str | None],
    is_pdf: bool = False,
) -> str:
    """Build a complete, standalone, white A4 print-ready HTML engineering report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname = os.path.basename(file_path) if file_path else "Not available"
    file_size = f"{os.path.getsize(file_path):,} bytes" if (file_path and os.path.exists(file_path)) else "Not available"
    ext = os.path.splitext(file_path)[1].upper() if file_path else "UNKNOWN"
    file_format = f"WAV Audio / IQ ({ext})" if ext == ".WAV" else f"Raw IQ Binary ({ext})"

    meta = results.get("meta", {})
    sr = results.get("sample_rate") or meta.get("sample_rate")
    total_samples = meta.get("total_samples")
    duration = (total_samples / sr) if (sr and total_samples) else None

    mod = results.get("modulation", "Not available")
    conf = results.get("confidence")
    conf_str = f"{conf:.2%}" if conf is not None else "Not available"
    all_probs = results.get("all_probs", {})

    obw = results.get("occupied_bandwidth_hz")
    cfo = results.get("center_freq_offset_hz")
    sps = results.get("sps")
    symbol_rate_str = f"{sr / sps:,.1f} symbols/sec" if (sr and sps and sps > 0) else "Not available"

    di = results.get("deinterleaver", "None")
    fec = results.get("fec_scheme", "None")
    fec_score = results.get("fec_score")
    if fec_score is not None and fec_score > 0:
        fec_display = f"{fec} (Score: {fec_score:.1%})"
    else:
        fec_display = str(fec)

    decoded_bytes_count = results.get("decoded_bytes", 0)
    h_off = results.get("header_offset")
    p_off = results.get("payload_offset")
    preamble = results.get("preamble", {})
    sync_status = "Locked" if preamble.get("found") else "Unlocked / Direct"

    # Bitstream telemetry
    final_bits = results.get("final_bits", np.array([], dtype=np.uint8))
    final_bytes = results.get("final_bytes", b"")
    bits_stat = _format_bits_summary(final_bits)
    hexdump_text = _bytes_to_hexdump(final_bytes)
    ascii_text = _extract_ascii(final_bytes)

    # NTRO Logo
    ntro_logo_uri = _load_ntro_logo_b64()
    if ntro_logo_uri:
        logo_html = f'<img src="{ntro_logo_uri}" width="78" style="display:block;" />'
    else:
        logo_html = '<div style="width:70px; height:45px; border:1px solid #cbd5e1; text-align:center; line-height:45px; font-weight:bold; color:#0f2a4a;">NTRO</div>'

    # Modulation Probability Table rows (Clean 2-column: Modulation & Probability)
    prob_rows = []
    if all_probs:
        for cname, pval in sorted(all_probs.items(), key=lambda x: -x[1]):
            pct = pval * 100.0
            prob_rows.append(f"""
                <tr>
                    <td style="font-weight:bold; color:#0f2a4a; border:1px solid #cbd5e1; padding:3px 8px;">{cname}</td>
                    <td style="text-align:right; font-family:monospace; font-weight:bold; border:1px solid #cbd5e1; padding:3px 8px;">{pct:.2f}%</td>
                </tr>
            """)
    prob_table_html = "".join(prob_rows) if prob_rows else "<tr><td colspan='2' style='text-align:center;'>Not available</td></tr>"

    # Factual Summary
    summary_text = generate_factual_summary(results, file_path)

    # Plot image tag generator
    def _render_plot_tag(key: str, caption: str, img_w: int = 680) -> str:
        data_uri = plot_images.get(key)
        if data_uri:
            return f"""
            <div style="text-align:center; margin-bottom:6px;">
                <img src="{data_uri}" width="{img_w}" style="display:block; margin:0 auto; border:1px solid #e2e8f0; border-radius:3px;" />
                <div style="font-size:8pt; font-weight:bold; color:#475569; margin-top:2px;">{caption}</div>
            </div>
            """
        return f'<div style="text-align:center; padding:10px; color:#94a3b8; font-style:italic;">{caption} &mdash; Telemetry not available</div>'

    html_footer = ""
    if not is_pdf:
        html_footer = f"""
        <!-- ── HTML VIEW FOOTER ── -->
        <div style="border-top: 1px solid #cbd5e1; margin-top: 16px; padding-top: 6px; font-size: 7pt; color: #64748b;">
            <table style="width: 100%; border-collapse: collapse; border: none;">
                <tr>
                    <td style="border: none; text-align: left; color: #64748b; font-size: 7pt;">SIH Signal Analyzer &bull; Technical Report</td>
                    <td style="border: none; text-align: center; color: #64748b; font-size: 7pt;">National Technical Research Organisation (NTRO)</td>
                    <td style="border: none; text-align: right; color: #64748b; font-size: 7pt;">Generated: {timestamp}</td>
                </tr>
            </table>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Signal Analysis Report &mdash; {fname}</title>
<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        background-color: #ffffff;
        color: #1e293b;
        margin: 0;
        padding: 6px 10px;
        font-size: 8.5pt;
        line-height: 1.3;
    }}
    .sec-title {{
        font-size: 9.5pt;
        font-weight: bold;
        color: #0f2a4a;
        border-bottom: 1.5px solid #0f2a4a;
        padding-bottom: 1px;
        margin-top: 6px;
        margin-bottom: 3px;
        text-transform: uppercase;
    }}
    .sec-num {{
        color: #2563eb;
        margin-right: 3px;
    }}
    table.data-tbl {{
        width: 100%;
        border-collapse: collapse;
        font-size: 7.5pt;
        margin-bottom: 3px;
    }}
    table.data-tbl th, table.data-tbl td {{
        border: 1px solid #cbd5e1;
        padding: 2px 4px;
        text-align: left;
    }}
    table.data-tbl th {{
        background-color: #f1f5f9;
        color: #0f2a4a;
        font-weight: bold;
    }}
    table.data-tbl td.lbl {{
        background-color: #f8fafc;
        color: #334155;
        font-weight: bold;
        width: 38%;
    }}
    table.data-tbl td.val {{
        color: #0f172a;
    }}
    .badge {{
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 1px 4px;
        font-weight: bold;
        border-radius: 2px;
    }}
    pre.code-dump {{
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        padding: 3px 5px;
        font-family: Consolas, monospace;
        font-size: 7pt;
        color: #0f172a;
        line-height: 1.25;
        margin: 2px 0;
    }}
    .summary-box {{
        background-color: #f8fafc;
        border-left: 3.5px solid #2563eb;
        border-top: 1px solid #cbd5e1;
        border-right: 1px solid #cbd5e1;
        border-bottom: 1px solid #cbd5e1;
        padding: 4px 8px;
        font-size: 8pt;
        color: #1e293b;
        line-height: 1.35;
    }}
</style>
</head>
<body>

    <!-- ════════════════════ PAGE 1 ════════════════════ -->

    <!-- ── HEADER TABLE ── -->
    <table style="width: 100%; border-bottom: 2px solid #0f2a4a; padding-bottom: 4px; margin-bottom: 8px;">
        <tr>
            <td style="width: 85px; vertical-align: middle; border: none;">
                {logo_html}
            </td>
            <td style="vertical-align: middle; padding-left: 8px; border: none;">
                <div style="font-size: 14pt; font-weight: bold; color: #0f2a4a; text-transform: uppercase; margin: 0;">SIGNAL ANALYSIS REPORT</div>
                <div style="font-size: 9pt; font-weight: bold; color: #334155; margin-top: 1px;">Automated Signal Analysis and Parameter Extraction</div>
                <div style="font-size: 7.5pt; color: #64748b; margin-top: 1px;">SIH Signal Analyzer &bull; National Technical Research Organisation</div>
            </td>
            <td style="text-align: right; vertical-align: middle; font-size: 7.5pt; color: #475569; line-height: 1.3; border: none;">
                <div><b>Date / Time:</b> {timestamp}</div>
                <div><b>Input File:</b> {fname}</div>
                <div><b>Format:</b> {file_format}</div>
            </td>
        </tr>
    </table>

    <!-- ── SECTIONS 1 & 2 (TWO-COLUMN TABLE) ── -->
    <table style="width: 100%; border-collapse: collapse; border: none; margin-bottom: 4px;">
        <tr>
            <!-- Section 1 -->
            <td style="width: 49%; vertical-align: top; padding: 0; border: none;">
                <div class="sec-title"><span class="sec-num">1.</span> FILE / SIGNAL INFORMATION</div>
                <table class="data-tbl">
                    <tr><td class="lbl">File Name</td><td class="val">{fname}</td></tr>
                    <tr><td class="lbl">File Format</td><td class="val">{file_format}</td></tr>
                    <tr><td class="lbl">File Size</td><td class="val">{file_size}</td></tr>
                    <tr><td class="lbl">Signal Duration</td><td class="val">{_time_fmt(duration)}</td></tr>
                    <tr><td class="lbl">Sample Rate (Fs)</td><td class="val">{_hz_fmt(sr)}</td></tr>
                    <tr><td class="lbl">Center Freq Offset</td><td class="val">{_hz_fmt(cfo)}</td></tr>
                    <tr><td class="lbl">Total Samples</td><td class="val">{f'{total_samples:,}' if total_samples is not None else 'Not available'}</td></tr>
                </table>
            </td>
            <td style="width: 2%; border: none;"></td>
            <!-- Section 2 -->
            <td style="width: 49%; vertical-align: top; padding: 0; border: none;">
                <div class="sec-title"><span class="sec-num">2.</span> DETECTED SIGNAL PARAMETERS</div>
                <table class="data-tbl">
                    <tr><td class="lbl">Modulation</td><td class="val"><span class="badge">{mod}</span></td></tr>
                    <tr><td class="lbl">Modulation Confidence</td><td class="val"><b>{conf_str}</b></td></tr>
                    <tr><td class="lbl">Occupied Bandwidth</td><td class="val">{_hz_fmt(obw)}</td></tr>
                    <tr><td class="lbl">Carrier Freq Offset (CFO)</td><td class="val">{_hz_fmt(cfo)}</td></tr>
                    <tr><td class="lbl">Symbol Rate</td><td class="val">{symbol_rate_str}</td></tr>
                    <tr><td class="lbl">Samples Per Symbol</td><td class="val">{f'{sps} sps' if sps is not None else 'Not available'}</td></tr>
                    <tr><td class="lbl">Frame Sync Status</td><td class="val">{sync_status}</td></tr>
                </table>
            </td>
        </tr>
    </table>

    <!-- ── SECTION 3: MODULATION CLASSIFICATION ── -->
    <div class="sec-title"><span class="sec-num">3.</span> MODULATION CLASSIFICATION</div>
    <table class="data-tbl" style="max-width: 320px; margin-bottom: 4px;">
        <thead>
            <tr>
                <th style="width: 60%;">Modulation Scheme</th>
                <th style="width: 40%; text-align: right;">Probability</th>
            </tr>
        </thead>
        <tbody>
            {prob_table_html}
        </tbody>
    </table>

    <!-- ── SECTION 4: TIME-DOMAIN WAVEFORM ── -->
    <div class="sec-title"><span class="sec-num">4.</span> TIME-DOMAIN WAVEFORM</div>
    {_render_plot_tag("waveform", "Figure 1: Time-Domain Waveform (In-Phase I & Quadrature Q Components)", img_w=680)}

    <!-- ── SECTION 5: FREQUENCY SPECTRUM ── -->
    <div class="sec-title"><span class="sec-num">5.</span> FREQUENCY SPECTRUM</div>
    {_render_plot_tag("spectrum", "Figure 2: Power Spectral Density (PSD) and Occupied Bandwidth", img_w=680)}

    <!-- ════════════════════ PAGE 2 (PAGE BREAK) ════════════════════ -->
    <div style="page-break-before: always;"></div>

    <!-- ── SECTION 6: SPECTROGRAM / WATERFALL ── -->
    <div class="sec-title"><span class="sec-num">6.</span> SPECTROGRAM / WATERFALL</div>
    {_render_plot_tag("spectrogram", "Figure 3: Time-Frequency STFT Spectrogram Waterfall", img_w=680)}

    <!-- ── SECTIONS 7 & 8 (TWO-COLUMN TABLE) ── -->
    <table style="width: 100%; border-collapse: collapse; border: none; margin-bottom: 4px;">
        <tr>
            <!-- Section 7 -->
            <td style="width: 45%; vertical-align: top; padding: 0; border: none;">
                <div class="sec-title"><span class="sec-num">7.</span> CONSTELLATION DIAGRAM</div>
                {_render_plot_tag("constellation", "Figure 4: IQ Constellation", img_w=200)}
            </td>
            <td style="width: 3%; border: none;"></td>
            <!-- Section 8 -->
            <td style="width: 52%; vertical-align: top; padding: 0; border: none;">
                <div class="sec-title"><span class="sec-num">8.</span> SYNCHRONIZATION / FRAME ANALYSIS</div>
                <table class="data-tbl">
                    <tr>
                        <td class="lbl" style="width: 44%;">Frame Sync Status</td>
                        <td class="val">{sync_status}</td>
                    </tr>
                    <tr>
                        <td class="lbl">Header Offset</td>
                        <td class="val">{f'{h_off} bits' if h_off is not None else 'Not available'}</td>
                    </tr>
                    <tr>
                        <td class="lbl">Payload Offset</td>
                        <td class="val">{f'{p_off} bits' if p_off is not None else 'Not available'}</td>
                    </tr>
                    <tr>
                        <td class="lbl">Sync Pattern Locked</td>
                        <td class="val">{preamble.get('pattern_name', 'None detected') if preamble.get('found') else 'Direct framing'}</td>
                    </tr>
                    <tr>
                        <td class="lbl">Correlation Sync Match</td>
                        <td class="val">{f"{preamble.get('confidence', 0):.1%}" if preamble.get('found') else 'Not available'}</td>
                    </tr>
                    <tr>
                        <td class="lbl">Phase Ambiguity Resolved</td>
                        <td class="val">{'Yes (Costas ambiguity resolved)' if results.get('phase_inverted') else 'No (Direct polarity)'}</td>
                    </tr>
                    <tr>
                        <td class="lbl">Demodulated Bits</td>
                        <td class="val">{f"{len(results.get('decoded_bits_raw', [])):,} bits" if results.get('decoded_bits_raw') is not None else 'Not available'}</td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>

    <!-- ── SECTION 9: BITSTREAM / DECODING ── -->
    <div class="sec-title"><span class="sec-num">9.</span> BITSTREAM / DECODING</div>
    <table class="data-tbl">
        <tr>
            <td class="lbl" style="width: 22%;">Total Decoded Bits</td>
            <td class="val" style="width: 28%;">{bits_stat['total_bits']} bits</td>
            <td class="lbl" style="width: 22%;">Decoded Bytes</td>
            <td class="val" style="width: 28%;"><b>{decoded_bytes_count:,} bytes</b></td>
        </tr>
        <tr>
            <td class="lbl">Bit Density (1s / 0s)</td>
            <td class="val">{bits_stat['ones_count']} / {bits_stat['zeros_count']}</td>
            <td class="lbl">ASCII Text Preview</td>
            <td class="val" style="font-family: Consolas, monospace; color: #1e40af;">{ascii_text}</td>
        </tr>
        <tr>
            <td class="lbl">Bitstream Sample</td>
            <td class="val" colspan="3" style="font-family: Consolas, monospace; font-size: 7pt; word-break: break-all;">{bits_stat['preview']}</td>
        </tr>
    </table>
    <div style="font-size: 7.5pt; font-weight: bold; color: #334155; margin-top: 3px;">Decoded Bytes Hex Dump:</div>
    <pre class="code-dump">{hexdump_text}</pre>

    <!-- ── SECTION 10: FEC / DE-INTERLEAVER ANALYSIS ── -->
    <div class="sec-title"><span class="sec-num">10.</span> FEC / DE-INTERLEAVER ANALYSIS</div>
    <table class="data-tbl">
        <thead>
            <tr>
                <th style="width: 25%;">Component</th>
                <th style="width: 45%;">Detected Configuration</th>
                <th style="width: 30%;">Status / Confidence</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="lbl">De-interleaver</td>
                <td class="val"><b>{di}</b></td>
                <td class="val">{'Applied' if str(di).lower() not in ('none', '', '—') else 'Bypassed'}</td>
            </tr>
            <tr>
                <td class="lbl">Forward Error Correction (FEC)</td>
                <td class="val"><b>{fec}</b></td>
                <td class="val">{f"Score: {fec_score:.1%}" if (fec_score is not None and fec_score > 0) else ('Locked' if str(fec).lower() not in ('none', '', '—') else 'Uncoded')}</td>
            </tr>
            <tr>
                <td class="lbl">Channel Decoder</td>
                <td class="val">Automated Multi-Scheme Pipeline</td>
                <td class="val">{f"{decoded_bytes_count:,} bytes recovered" if decoded_bytes_count > 0 else 'No payload'}</td>
            </tr>
        </tbody>
    </table>

    <!-- ── SECTION 11: FINAL ANALYSIS SUMMARY ── -->
    <div class="sec-title"><span class="sec-num">11.</span> FINAL ANALYSIS SUMMARY</div>
    <div class="summary-box">
        {summary_text}
    </div>

    {html_footer}

</body>
</html>
"""
    return html


# ---------------------------------------------------------------------------
# High-Quality PDF Exporter (A4 Portrait via PyQt6)
# ---------------------------------------------------------------------------

def export_to_pdf(html_content: str, output_pdf_path: str) -> None:
    """
    Compile HTML report directly to an A4 PDF document using PyQt6's
    QTextDocument, QPdfWriter, and QPainter with standard print typography,
    margins, running headers, and page numbers.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)

    doc = QTextDocument()
    doc.setHtml(html_content)

    writer = QPdfWriter(output_pdf_path)
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
    printable_h = h - 22
    doc.setPageSize(QSizeF(w, printable_h))

    page_count = doc.pageCount()
    painter = QPainter(writer)

    for page_idx in range(page_count):
        if page_idx > 0:
            writer.newPage()

        # Render document content for current page
        painter.save()
        painter.translate(0, -page_idx * printable_h)
        doc.drawContents(painter, QRectF(0, page_idx * printable_h, w, printable_h))
        painter.restore()

        # Render running footer with calibrated page number
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


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def generate_report(
    results: dict[str, Any],
    file_path: str,
    output_path: str,
    panels: Mapping[str, Any] | None = None,
) -> str:
    """
    Generate and save a signal analysis report (HTML or PDF).

    Parameters
    ----------
    results : dict
        The complete analysis dictionary produced by AnalyzerWorker.
    file_path : str
        Source capture file path.
    output_path : str
        Destination path (.html or .pdf).
    panels : dict, optional
        Active GUI panels mapping ('wave', 'spec', 'const') for diagram snapshots.

    Returns
    -------
    str : Path to the generated report file.
    """
    if not results:
        raise ValueError("Cannot generate report: No analysis results available.")

    # 1. Render print-friendly plots (using clean PyQt6 native offscreen rendering with fallback)
    plot_images = capture_all_plots(results, panels)

    # 2. Check destination format
    ext = os.path.splitext(output_path)[1].lower()
    is_pdf = (ext == ".pdf")

    # 3. Build complete standalone HTML report
    html_content = build_html_report(results, file_path, plot_images, is_pdf=is_pdf)

    # 4. Save as requested format
    if is_pdf:
        export_to_pdf(html_content, output_path)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return output_path
