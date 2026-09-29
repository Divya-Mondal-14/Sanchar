"""
gui/bitstream_panel.py

PyQt6 decoded-bitstream display panel for the SIH signal analyzer.

Shows the decoded bit stream in two synchronized views:
  - Binary string (e.g., 01110010...)
  - Hex dump (e.g., 72 65 63 ...)

Header bytes (up to payload_start) are highlighted in amber.
Payload bytes are shown in the default color.

Public API
----------
    BitstreamPanel(QWidget)
        .set_data(bits, header_end_bit)
            bits           : np.ndarray uint8, shape (N,) — 0/1 bit stream
            header_end_bit : int — bit index where payload starts (bits before
                             this index are highlighted as header)
        .clear()
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QLabel,
    QSplitter, QSizePolicy, QPushButton,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCharFormat, QColor, QTextCursor


_HEADER_BG   = QColor("#3d2b00")   # dark amber background
_HEADER_FG   = QColor("#ffa657")   # amber text
_PAYLOAD_BG  = QColor("#0d1117")   # normal dark
_PAYLOAD_FG  = QColor("#e6edf3")   # normal light
_DIM_FG      = QColor("#484f58")

_MAX_BITS_DISPLAYED = 8192          # cap to keep Qt text engine fast


def _bits_to_hex_dump(bits: np.ndarray) -> list[str]:
    """Pack bits MSB-first into bytes and return hex dump lines (16 bytes/line)."""
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (8 - len(bits) % 8) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    raw_bytes = np.packbits(bits, bitorder='big')

    lines = []
    for offset in range(0, len(raw_bytes), 16):
        chunk = raw_bytes[offset : offset + 16]
        hex_part = " ".join(f"{b:02X}" for b in chunk)
        asc_part = "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in chunk)
        lines.append(f"{offset:06X}  {hex_part:<47}  |{asc_part}|")

    return lines


class BitstreamPanel(QWidget):
    """Decoded bitstream viewer with header/payload highlighting."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._bits: np.ndarray | None = None
        self._header_end_bit: int = 0
        self._theme: str = "dark"
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Header row ───────────────────────────────────────────────
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(8)

        self._title = QLabel("Decoded Bitstream")
        self._title.setFont(QFont("Inter", 10, QFont.Weight.DemiBold))
        self._title.setStyleSheet("color: #e6edf3;")
        hdr_row.addWidget(self._title)

        self._legend_header = QLabel("  Header  ")
        self._legend_header.setStyleSheet(
            "background: #3d2b00; color: #ffa657; border-radius: 3px; "
            "padding: 1px 5px; font-size: 8pt;"
        )
        hdr_row.addWidget(self._legend_header)

        self._legend_payload = QLabel("  Payload  ")
        self._legend_payload.setStyleSheet(
            "background: #161b22; color: #e6edf3; border-radius: 3px; "
            "padding: 1px 5px; font-size: 8pt;"
        )
        hdr_row.addWidget(self._legend_payload)

        hdr_row.addStretch(1)

        self._copy_btn = QPushButton("Copy Hex")
        self._copy_btn.setFixedHeight(24)
        self._copy_btn.setStyleSheet("""
            QPushButton {
                background: #21262d; color: #8b949e;
                border: 1px solid #30363d; border-radius: 4px;
                padding: 0 8px; font-size: 8pt;
            }
            QPushButton:hover { background: #30363d; color: #e6edf3; }
        """)
        self._copy_btn.clicked.connect(self._copy_hex)
        hdr_row.addWidget(self._copy_btn)

        root.addLayout(hdr_row)

        # ── Splitter: binary | hex dump ──────────────────────────────
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(4)
        self._splitter.setStyleSheet("QSplitter::handle { background: #21262d; }")
        splitter = self._splitter

        _text_style = """
            QPlainTextEdit {
                background-color: #0d1117;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 4px;
                selection-background-color: #264f78;
            }
        """
        mono = QFont("Courier New", 8)
        mono.setFixedPitch(True)

        # Binary view
        self._binary_view = QPlainTextEdit()
        self._binary_view.setReadOnly(True)
        self._binary_view.setFont(mono)
        self._binary_view.setStyleSheet(_text_style)
        self._binary_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._binary_view.setMinimumHeight(120)
        splitter.addWidget(self._binary_view)

        # Hex dump view
        self._hex_view = QPlainTextEdit()
        self._hex_view.setReadOnly(True)
        self._hex_view.setFont(mono)
        self._hex_view.setStyleSheet(_text_style)
        self._hex_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._hex_view.setMinimumHeight(120)
        splitter.addWidget(self._hex_view)

        splitter.setSizes([300, 500])
        root.addWidget(splitter)

        # ── Status ───────────────────────────────────────────────────
        self._status = QLabel("No data loaded.")
        self._status.setFont(QFont("Inter", 8))
        self._status.setStyleSheet("color: #484f58; padding: 2px 4px;")
        root.addWidget(self._status)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_data(self, bits: np.ndarray, header_end_bit: int = 0):
        """
        Display decoded bits with header highlight.

        Parameters
        ----------
        bits           : np.ndarray uint8, shape (N,)
        header_end_bit : int — first payload bit index (header = bits[:header_end_bit])
        """
        self._bits = np.asarray(bits, dtype=np.uint8)
        self._header_end_bit = max(0, header_end_bit)
        self._refresh()

    def clear(self):
        self._bits = None
        self._header_end_bit = 0
        self._binary_view.clear()
        self._hex_view.clear()
        self._status.setText("No data loaded.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh(self):
        if self._bits is None:
            return

        bits = self._bits
        n_total = len(bits)
        truncated = n_total > _MAX_BITS_DISPLAYED
        display_bits = bits[:_MAX_BITS_DISPLAYED]
        header_end = min(self._header_end_bit, len(display_bits))

        # ── Binary view ───────────────────────────────────────────────
        # Build the text, then apply character formats with QTextCursor
        # (bit groups of 8 separated by space for readability)
        chunks = [
            "".join(str(b) for b in display_bits[i : i + 8])
            for i in range(0, len(display_bits), 8)
        ]
        binary_text = " ".join(chunks)

        self._binary_view.setPlainText(binary_text)

        # Apply header highlight
        if header_end > 0:
            h_bg = QColor("#fff8c5") if self._theme == "light" else QColor(_HEADER_BG)
            h_fg = QColor("#9a6700") if self._theme == "light" else QColor(_HEADER_FG)

            # Convert bit position to character position:
            # Every 8 bits = 9 chars (8 digits + 1 space), except maybe last
            header_chars = (header_end // 8) * 9 + (header_end % 8)
            cursor = self._binary_view.textCursor()
            cursor.setPosition(0)
            cursor.setPosition(min(header_chars, len(binary_text)),
                               QTextCursor.MoveMode.KeepAnchor)
            fmt = QTextCharFormat()
            fmt.setBackground(h_bg)
            fmt.setForeground(h_fg)
            cursor.mergeCharFormat(fmt)

        # ── Hex dump view ─────────────────────────────────────────────
        header_bytes = header_end // 8
        hex_lines = _bits_to_hex_dump(display_bits)
        self._hex_view.setPlainText("\n".join(hex_lines))

        # Highlight header lines in hex view
        if header_bytes > 0:
            h_bg = QColor("#fff8c5") if self._theme == "light" else QColor(_HEADER_BG)
            h_fg = QColor("#9a6700") if self._theme == "light" else QColor(_HEADER_FG)

            cursor = self._hex_view.textCursor()
            cursor.setPosition(0)
            # Each line is 16 bytes; header spans first ceil(header_bytes/16) lines
            n_header_lines = int(np.ceil(header_bytes / 16))
            for _ in range(n_header_lines):
                cursor.movePosition(QTextCursor.MoveOperation.EndOfLine,
                                    QTextCursor.MoveMode.KeepAnchor)
            fmt = QTextCharFormat()
            fmt.setBackground(h_bg)
            fmt.setForeground(h_fg)
            cursor.mergeCharFormat(fmt)

        trunc_note = f"  (showing first {_MAX_BITS_DISPLAYED})" if truncated else ""
        self._status.setText(
            f"{n_total:,} bits  |  header: {self._header_end_bit} bits  |  "
            f"payload: {max(0, n_total - self._header_end_bit):,} bits{trunc_note}"
        )

    def _copy_hex(self):
        """Copy hex dump to clipboard."""
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._hex_view.toPlainText())

    # ------------------------------------------------------------------
    # Theme Support
    # ------------------------------------------------------------------

    def set_theme(self, theme: str = "dark"):
        """Dynamically apply light or dark theme styling."""
        self._theme = theme
        if theme == "light":
            self._title.setStyleSheet("color: #1f2328;")
            self._legend_header.setStyleSheet(
                "background: #fff8c5; color: #9a6700; border: 1px solid #d4a72c; border-radius: 3px; "
                "padding: 1px 5px; font-size: 8pt;"
            )
            self._legend_payload.setStyleSheet(
                "background: #f6f8fa; color: #1f2328; border: 1px solid #d0d7de; border-radius: 3px; "
                "padding: 1px 5px; font-size: 8pt;"
            )
            self._copy_btn.setStyleSheet("""
                QPushButton {
                    background: #ffffff; color: #1f2328;
                    border: 1px solid #d0d7de; border-radius: 4px;
                    padding: 0 8px; font-size: 8pt;
                }
                QPushButton:hover { background: #f3f4f6; color: #0969da; border-color: #0969da; }
            """)
            self._splitter.setStyleSheet("QSplitter::handle { background: #d0d7de; }")
            text_style = """
                QPlainTextEdit {
                    background-color: #ffffff;
                    color: #1f2328;
                    border: 1px solid #d0d7de;
                    border-radius: 4px;
                    selection-background-color: #b6d3fe;
                }
            """
            self._binary_view.setStyleSheet(text_style)
            self._hex_view.setStyleSheet(text_style)
            self._status.setStyleSheet("color: #656d76; padding: 2px 4px;")
        else:
            self._title.setStyleSheet("color: #e6edf3;")
            self._legend_header.setStyleSheet(
                "background: #3d2b00; color: #ffa657; border-radius: 3px; "
                "padding: 1px 5px; font-size: 8pt;"
            )
            self._legend_payload.setStyleSheet(
                "background: #161b22; color: #e6edf3; border-radius: 3px; "
                "padding: 1px 5px; font-size: 8pt;"
            )
            self._copy_btn.setStyleSheet("""
                QPushButton {
                    background: #21262d; color: #8b949e;
                    border: 1px solid #30363d; border-radius: 4px;
                    padding: 0 8px; font-size: 8pt;
                }
                QPushButton:hover { background: #30363d; color: #e6edf3; }
            """)
            self._splitter.setStyleSheet("QSplitter::handle { background: #21262d; }")
            text_style = """
                QPlainTextEdit {
                    background-color: #0d1117;
                    color: #e6edf3;
                    border: 1px solid #30363d;
                    border-radius: 4px;
                    selection-background-color: #264f78;
                }
            """
            self._binary_view.setStyleSheet(text_style)
            self._hex_view.setStyleSheet(text_style)
            self._status.setStyleSheet("color: #484f58; padding: 2px 4px;")

        if self._bits is not None:
            self.set_data(self._bits, self._header_end_bit)

