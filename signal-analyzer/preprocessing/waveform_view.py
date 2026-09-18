"""
preprocessing/waveform_view.py

Raw-waveform data preparation for the GUI's signal view (URH-style layering).

Consumes complex-sample arrays produced by file_loader.py (shape: (N,) complex64)
and returns Python dicts ready to hand to WaveformPanel / SpectrogramPanel.

Public API
----------
    extract_waveform(complex_samples)  -> dict
    downsample_for_display(data, max_points=8192) -> np.ndarray

extract_waveform() keys
------------------------
    'i'           : np.ndarray float32 — in-phase component
    'q'           : np.ndarray float32 — quadrature component
    'amplitude'   : np.ndarray float32 — instantaneous envelope  |I + jQ|
    'phase_rad'   : np.ndarray float32 — instantaneous phase  angle(I + jQ)  in radians
    'inst_freq'   : np.ndarray float32 — instantaneous frequency (finite-diff of unwrapped
                                         phase, normalised to [-0.5, 0.5] cycles/sample)
                                         length = N-1 (one sample shorter than the rest)
    'n_samples'   : int                — total number of samples in the capture
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_waveform(complex_samples: np.ndarray) -> dict:
    """
    Decompose a complex-sample array into waveform components for display.

    Parameters
    ----------
    complex_samples : np.ndarray, shape (N,), dtype complex64 or complex128
        Full (unwindowed) IQ capture as returned by file_loader.load_wav /
        file_loader.load_iq_raw.

    Returns
    -------
    dict with keys: 'i', 'q', 'amplitude', 'phase_rad', 'inst_freq', 'n_samples'
    """
    if complex_samples.ndim != 1:
        raise ValueError(
            f"Expected 1-D complex array, got shape {complex_samples.shape}. "
            "Pass the raw complex_samples from load_wav/load_iq_raw, not the "
            "windowed array from window_samples()."
        )

    cs = complex_samples.astype(np.complex64)

    i = cs.real.copy()
    q = cs.imag.copy()

    amplitude = np.abs(cs)

    phase_rad = np.angle(cs)  # range [-π, π]

    # Instantaneous frequency: normalised rate-of-phase-change
    # unwrap removes ±π discontinuities, diff gives sample-to-sample delta
    unwrapped = np.unwrap(phase_rad.astype(np.float64))
    inst_freq = (np.diff(unwrapped) / (2.0 * np.pi)).astype(np.float32)

    return {
        "i": i.astype(np.float32),
        "q": q.astype(np.float32),
        "amplitude": amplitude.astype(np.float32),
        "phase_rad": phase_rad.astype(np.float32),
        "inst_freq": inst_freq,            # length N-1
        "n_samples": int(cs.shape[0]),
    }


# ---------------------------------------------------------------------------
# Display decimation
# ---------------------------------------------------------------------------

def downsample_for_display(
    data: np.ndarray,
    max_points: int = 8192,
) -> np.ndarray:
    """
    Reduce `data` to at most `max_points` samples using power-of-2 decimation
    (takes every 2^k-th sample so the result index-maps cleanly back to time).

    This is the pre-decimation step; pyqtgraph's autoDownsample / clipToView
    will then handle further dynamic reduction while the user pans/zooms.

    Parameters
    ----------
    data : np.ndarray, 1-D
    max_points : int (default 8192)

    Returns
    -------
    np.ndarray — decimated copy (view if no decimation needed)
    """
    n = data.shape[0]
    if n <= max_points:
        return data

    # Smallest power-of-2 factor that brings n down to <= max_points
    factor = 1
    while n // factor > max_points:
        factor *= 2

    return data[::factor].copy()


# ---------------------------------------------------------------------------
# Convenience: build display-ready dict (decimated) for the GUI
# ---------------------------------------------------------------------------

def waveform_display_data(
    complex_samples: np.ndarray,
    max_points: int = 32768,
) -> dict:
    """
    Extract waveform components and decimate each array to max_points for
    efficient GUI rendering.

    Returns the same keys as extract_waveform() plus:
        'decimation_factor' : int — the power-of-2 factor applied
        'display_n_samples' : int — number of points in each (decimated) array
                                    ('inst_freq' is 1 shorter than the rest)
    """
    wv = extract_waveform(complex_samples)
    n = wv["n_samples"]

    # Compute common factor from the longest arrays
    factor = 1
    while n // factor > max_points:
        factor *= 2

    def _decimate(arr: np.ndarray) -> np.ndarray:
        return arr[::factor].copy() if len(arr) > max_points else arr

    return {
        "i":                _decimate(wv["i"]),
        "q":                _decimate(wv["q"]),
        "amplitude":        _decimate(wv["amplitude"]),
        "phase_rad":        _decimate(wv["phase_rad"]),
        "inst_freq":        _decimate(wv["inst_freq"]),
        "n_samples":        n,
        "decimation_factor": factor,
        "display_n_samples": int(np.ceil(n / factor)),
    }


# ---------------------------------------------------------------------------
# Self-test (run directly: python preprocessing/waveform_view.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("waveform_view.py — self-test with synthetic IQ signal")

    # --- Generate a synthetic BPSK-like signal ---
    rng = np.random.default_rng(42)
    N = 4096
    fs = 1_000_000  # 1 MHz sample rate (notional)
    t = np.arange(N) / fs
    fc = 100_000    # 100 kHz carrier

    # BPSK: random bits modulate carrier phase (0 or π)
    bits = rng.integers(0, 2, size=N // 8)
    phases = np.repeat(bits * np.pi, 8)           # 8 samples/symbol
    noise = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * 0.1
    iq_complex = (np.exp(1j * (2 * np.pi * fc * t + phases)) + noise).astype(np.complex64)

    # --- Test extract_waveform ---
    wv = extract_waveform(iq_complex)
    print(f"\nextract_waveform() keys: {list(wv.keys())}")
    print(f"  i shape:          {wv['i'].shape},  dtype: {wv['i'].dtype}")
    print(f"  q shape:          {wv['q'].shape},  dtype: {wv['q'].dtype}")
    print(f"  amplitude shape:  {wv['amplitude'].shape}")
    print(f"  phase_rad shape:  {wv['phase_rad'].shape}")
    print(f"  inst_freq shape:  {wv['inst_freq'].shape}  (should be N-1 = {N-1})")
    print(f"  n_samples:        {wv['n_samples']}  (should be {N})")

    assert wv["i"].shape == (N,),          "i shape mismatch"
    assert wv["q"].shape == (N,),          "q shape mismatch"
    assert wv["amplitude"].shape == (N,),  "amplitude shape mismatch"
    assert wv["phase_rad"].shape == (N,),  "phase_rad shape mismatch"
    assert wv["inst_freq"].shape == (N-1,),"inst_freq should be length N-1"
    assert wv["n_samples"] == N,           "n_samples mismatch"

    # Amplitude should be close to 1 (carrier) with small noise
    mean_amp = float(np.mean(wv["amplitude"]))
    assert 0.8 < mean_amp < 1.2, f"Unexpected mean amplitude: {mean_amp}"
    print(f"  mean amplitude ~ {mean_amp:.4f}  OK")

    # --- Test downsample_for_display ---
    dec = downsample_for_display(wv["i"], max_points=512)
    print(f"\ndownsample_for_display(max_points=512): {wv['i'].shape[0]} -> {dec.shape[0]}")
    assert dec.shape[0] <= 512, "Decimation did not reduce below max_points"
    print("  Decimation check OK")

    # --- Test waveform_display_data ---
    disp = waveform_display_data(iq_complex, max_points=1024)
    print(f"\nwaveform_display_data(max_points=1024):")
    print(f"  decimation_factor:  {disp['decimation_factor']}")
    print(f"  display_n_samples:  {disp['display_n_samples']}")
    assert disp["i"].shape[0] <= 1024, "Display i exceeds max_points"
    print("  waveform_display_data OK")

    print("\nAll assertions passed. waveform_view.py is working correctly.")
    sys.exit(0)
