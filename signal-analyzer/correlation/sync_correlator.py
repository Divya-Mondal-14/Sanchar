"""
correlation/sync_correlator.py

Bitstream correlator: cross-correlates a decoded bit stream against known
sync/preamble patterns to identify header/payload boundaries.

This is the final step before the GUI's bitstream panel — it takes the raw
decoded bits from fec_decode.py and finds where the frame header starts,
allowing the bitstream panel to highlight the header region.

Common preamble/sync patterns included
---------------------------------------
    HDLC / AX.25 / APRS  : 0x7E  = 01111110
    CCSDS ASM             : 0x1ACFFC1D (4 bytes)
    DVB-S2 SOF            : 0x018D2E82 (partial, 3 bytes, for demo)
    Generic all-ones      : 0xFF
    Generic alternating   : 0xAA = 10101010 (carrier tone / AGC preamble)
    Zadoff-Chu style      : 10-bit Barker code 1110010000 (BPSK)

Public API
----------
    correlate_sync(bits, pattern_bits)    -> dict
    find_preamble(bits, known_patterns)   -> dict
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Known sync-word patterns (bit arrays, MSB first)
# ---------------------------------------------------------------------------

def _hex_to_bits(hex_str: str) -> np.ndarray:
    """Convert a hex string like '7E' or '1ACFFC1D' to a uint8 bit array."""
    raw = bytes.fromhex(hex_str)
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='big').astype(np.uint8)


# Barker codes (classic radar / comms preambles)
_BARKER_13 = np.array([1,1,1,1,1,0,0,1,1,0,1,0,1], dtype=np.uint8)
_BARKER_7  = np.array([1,1,1,0,0,1,0], dtype=np.uint8)
_BARKER_11 = np.array([1,1,1,0,0,0,1,0,0,1,0], dtype=np.uint8)

KNOWN_PATTERNS: dict[str, np.ndarray] = {
    "HDLC_FLAG":    _hex_to_bits("7E"),          # 01111110
    "AX25_SYNC":    _hex_to_bits("7E7E7E7E"),    # 4x HDLC flags
    "CCSDS_ASM":    _hex_to_bits("1ACFFC1D"),    # CCSDS Attached Sync Marker
    "DVB_SOF":      _hex_to_bits("018D2E"),       # DVB-S2 SOF (first 3 bytes)
    "PREAMBLE_FF":  _hex_to_bits("FF"),           # all-ones
    "PREAMBLE_AA":  _hex_to_bits("AA"),           # alternating 10101010
    "BARKER_13":    _BARKER_13,
    "BARKER_7":     _BARKER_7,
    "BARKER_11":    _BARKER_11,
}


# ---------------------------------------------------------------------------
# Core correlator
# ---------------------------------------------------------------------------

def correlate_sync(
    bits: np.ndarray,
    pattern_bits: np.ndarray,
) -> dict:
    """
    Cross-correlate a received bit stream against a known sync pattern.

    Uses bipolar {-1, +1} cross-correlation (equivalent to computing
    the number of bit-agreements minus disagreements at each lag).

    Parameters
    ----------
    bits         : np.ndarray, 1-D, dtype uint8 (values 0/1)
    pattern_bits : np.ndarray, 1-D, dtype uint8 (values 0/1)

    Returns
    -------
    dict with keys:
        'peak_offset'     : int   — sample offset of the highest correlation peak
        'peak_score'      : float — normalised correlation [-1, 1] at the peak
        'confidence'      : float — peak / max-possible in [0, 1]
        'all_peaks'       : np.ndarray — full normalised correlation array
        'header_start'    : int   — estimated start of header (== peak_offset)
        'payload_start'   : int   — estimated start of payload
                                    (header_start + len(pattern_bits))
        'pattern_length'  : int   — number of bits in the sync pattern
    """
    bits = np.asarray(bits, dtype=np.float32)
    pattern = np.asarray(pattern_bits, dtype=np.float32)

    # Convert to bipolar
    b_bipolar = 2.0 * bits - 1.0
    p_bipolar = 2.0 * pattern - 1.0

    n = len(b_bipolar)
    m = len(p_bipolar)

    if n < m:
        return {
            "peak_offset": 0,
            "peak_score": 0.0,
            "confidence": 0.0,
            "all_peaks": np.zeros(1, dtype=np.float32),
            "header_start": 0,
            "payload_start": m,
            "pattern_length": m,
        }

    # Sliding dot product via numpy correlate
    # 'valid' mode: output length = n - m + 1
    corr = np.correlate(b_bipolar, p_bipolar, mode='valid').astype(np.float32)
    corr_norm = corr / float(m)   # normalise to [-1, 1]

    peak_idx = int(np.argmax(corr_norm))
    peak_score = float(corr_norm[peak_idx])
    confidence = float((peak_score + 1.0) / 2.0)  # map [-1,1] -> [0,1]

    # Calculate statistical significance (z-score = (peak - mean) / std)
    mean_corr = float(np.mean(corr_norm))
    std_corr = float(np.std(corr_norm))
    z_score = float((peak_score - mean_corr) / max(std_corr, 1e-6))

    return {
        "peak_offset": peak_idx,
        "peak_score": peak_score,
        "confidence": confidence,
        "z_score": z_score,
        "all_peaks": corr_norm,
        "header_start": peak_idx,
        "payload_start": peak_idx + m,
        "pattern_length": m,
    }


# ---------------------------------------------------------------------------
# Multi-pattern preamble finder
# ---------------------------------------------------------------------------

def find_preamble(
    bits: np.ndarray,
    known_patterns: dict[str, np.ndarray] | None = None,
    confidence_threshold: float = 0.70,
    z_score_threshold: float = 3.0,
) -> dict:
    """
    Scan the decoded bit stream against all known sync patterns and return
    the best match.

    Parameters
    ----------
    bits : np.ndarray, 1-D, dtype uint8
    known_patterns : dict of {name: bit_array} or None (uses built-in KNOWN_PATTERNS)
    confidence_threshold : float
        Minimum confidence in [0, 1] to consider a match valid.
    z_score_threshold : float
        Minimum statistical peak significance (peak vs background sidelobes) in sigmas.

    Returns
    -------
    dict with keys:
        'found'           : bool  — True if any pattern exceeded both thresholds
        'pattern_name'    : str   — name of the best matching pattern, or 'none'
        'pattern_bits'    : np.ndarray uint8 — the winning pattern
        'confidence'      : float — confidence of the best match
        'z_score'         : float — peak z-score of the winning match
        'header_start'    : int   — bit offset of header start
        'payload_start'   : int   — bit offset of payload start
        'pattern_length'  : int   — length of sync pattern in bits
        'all_results'     : list of (name, confidence, peak_offset) sorted desc
    """
    if known_patterns is None:
        known_patterns = KNOWN_PATTERNS

    all_results = []
    best_significance = -1.0
    best_confidence = -1.0
    best_z_score = 0.0
    best_name = "none"
    best_result: dict = {}
    best_pattern = np.array([], dtype=np.uint8)

    # Evaluate patterns; rank by significance = confidence * sqrt(length)
    for name, pattern in sorted(known_patterns.items(), key=lambda x: -len(x[1])):
        r = correlate_sync(bits, pattern)
        conf = r["confidence"]
        z = r.get("z_score", 0.0)
        sig = conf * np.sqrt(len(pattern))
        all_results.append((name, conf, r["peak_offset"]))

        if conf >= confidence_threshold and z >= z_score_threshold and sig > best_significance:
            best_significance = sig
            best_confidence = conf
            best_z_score = z
            best_name = name
            best_result = r
            best_pattern = pattern
        elif best_significance < 0 and conf > best_confidence:
            best_confidence = conf
            best_z_score = z
            best_name = name
            best_result = r
            best_pattern = pattern

    all_results.sort(key=lambda x: x[1], reverse=True)
    found = best_name != "none" and best_confidence >= confidence_threshold and best_z_score >= z_score_threshold

    return {
        "found": found,
        "pattern_name": best_name if found else "none",
        "pattern_bits": best_pattern,
        "confidence": best_confidence,
        "z_score": best_z_score,
        "header_start": best_result.get("header_start", 0),
        "payload_start": best_result.get("payload_start", 0),
        "pattern_length": best_result.get("pattern_length", 0),
        "all_results": all_results,
    }


# ---------------------------------------------------------------------------
# Repeated-pattern detector (for long preambles like 0xAA chains)
# ---------------------------------------------------------------------------

def find_repetition_period(
    bits: np.ndarray,
    max_period: int = 32,
) -> int | None:
    """
    Detect the dominant repetition period of a bit stream (useful for
    identifying preamble patterns like 0xAA repeated N times).

    Returns the period (in bits) if a clear periodicity is found, else None.
    """
    if len(bits) < 4:
        return None

    b = (2.0 * bits.astype(np.float64) - 1.0)
    acf = np.correlate(b, b, mode='full')
    acf = acf[len(b) - 1:]   # keep lag >= 0
    acf = acf / (acf[0] + 1e-12)

    # Find first local maximum after lag=1
    for lag in range(2, min(max_period + 1, len(acf) - 1)):
        if acf[lag] > acf[lag - 1] and acf[lag] > acf[lag + 1] and acf[lag] > 0.5:
            return lag

    return None


# ---------------------------------------------------------------------------
# Self-test (python correlation/sync_correlator.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("sync_correlator.py -- self-test")

    rng = np.random.default_rng(7)

    # --- Test 1: inject a known HDLC flag into random bits and detect it ---
    PATTERN = KNOWN_PATTERNS["HDLC_FLAG"]
    PATTERN_LEN = len(PATTERN)
    INJECT_AT = 150

    random_bits = rng.integers(0, 2, size=300, dtype=np.uint8)
    random_bits[INJECT_AT : INJECT_AT + PATTERN_LEN] = PATTERN

    result = correlate_sync(random_bits, PATTERN)
    print(f"\n  correlate_sync (HDLC flag injected at {INJECT_AT}):")
    print(f"    peak_offset  = {result['peak_offset']}  (expected {INJECT_AT})")
    print(f"    peak_score   = {result['peak_score']:.4f}")
    print(f"    confidence   = {result['confidence']:.4f}")
    print(f"    payload_start= {result['payload_start']}")
    assert result["peak_offset"] == INJECT_AT, \
        f"Peak at wrong offset: {result['peak_offset']} vs {INJECT_AT}"
    assert result["confidence"] > 0.9, \
        f"Low confidence for exact match: {result['confidence']}"
    print("  correlate_sync: PASSED")

    # --- Test 2: find_preamble ---
    CCSDS = KNOWN_PATTERNS["CCSDS_ASM"]
    ASM_AT = 64
    bits2 = rng.integers(0, 2, size=512, dtype=np.uint8)
    bits2[ASM_AT : ASM_AT + len(CCSDS)] = CCSDS

    preamble_result = find_preamble(bits2, known_patterns={"CCSDS_ASM": KNOWN_PATTERNS["CCSDS_ASM"]})
    print(f"\n  find_preamble (CCSDS_ASM injected at {ASM_AT}, searched against CCSDS_ASM only):")
    print(f"    found          = {preamble_result['found']}")
    print(f"    pattern_name   = {preamble_result['pattern_name']}")
    print(f"    confidence     = {preamble_result['confidence']:.4f}")
    print(f"    header_start   = {preamble_result['header_start']}")
    print(f"    payload_start  = {preamble_result['payload_start']}")
    assert preamble_result["found"], "find_preamble: no match found"
    assert preamble_result["header_start"] == ASM_AT, \
        f"Wrong header_start: {preamble_result['header_start']} vs {ASM_AT}"
    print("  find_preamble: PASSED")

    # --- Test 3: repetition period ---
    # Construct a stream with period-8 AA preamble (10101010)
    period_bits = np.tile(np.array([1,0,1,0,1,0,1,0], dtype=np.uint8), 32)
    period = find_repetition_period(period_bits, max_period=16)
    print(f"\n  find_repetition_period on 0xAA: {period}  (expected 2)")
    assert period == 2, f"Wrong period: {period}"
    print("  find_repetition_period: PASSED")

    print("\nAll sync_correlator self-tests PASSED.")
    sys.exit(0)
