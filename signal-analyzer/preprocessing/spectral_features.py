"""
preprocessing/spectral_features.py

PSD, spectrogram, and occupied-bandwidth estimation for a loaded IQ capture.

Scope note: true blind *sample-rate* estimation from raw IQ alone isn't
generally feasible without symbol-timing / cyclostationary analysis (that's
a separate, harder problem — see the roadmap item for symbol-rate via
cyclostationary features). What IS achievable here, and what this module
provides, is: given a *known* sample rate (from the WAV header via
file_loader.load_wav, or user-supplied for raw .iq captures), estimate the
signal's *occupied bandwidth* and produce PSD/spectrogram data for the
GUI's spectrogram panel and params panel.

No scipy dependency — implemented with numpy only (Welch's method and STFT
done manually), to match the project's existing dependency list
(torch numpy h5py pyqt6 pyqtgraph).

Usage
-----
    from file_loader import load_wav
    from spectral_features import analyze

    complex_samples, sample_rate = load_wav("capture.wav")
    features = analyze(complex_samples, sample_rate)
    # features = {
    #     'freqs': np.ndarray,            PSD frequency bins (Hz, centered at 0)
    #     'psd_db': np.ndarray,            PSD in dB, same length as freqs
    #     'spec_times': np.ndarray,        spectrogram time bins (s)
    #     'spec_freqs': np.ndarray,        spectrogram frequency bins (Hz)
    #     'spec_db': np.ndarray,           spectrogram magnitude in dB, shape (n_freqs, n_times)
    #     'occupied_bandwidth_hz': float,  estimated occupied bandwidth
    #     'center_freq_offset_hz': float,  estimated offset of energy centroid from 0 Hz
    # }
"""

from __future__ import annotations

import numpy as np

DEFAULT_NPERSEG_PSD = 1024
DEFAULT_NPERSEG_SPEC = 256
DEFAULT_OVERLAP_FRAC = 0.5
DB_FLOOR = -120.0  # clamp for log(0) safety


def _hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def _to_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 10 ** (DB_FLOOR / 10.0)))


def compute_psd(
    complex_samples: np.ndarray,
    sample_rate: float,
    nperseg: int = DEFAULT_NPERSEG_PSD,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Welch's method PSD estimate (averaged, windowed periodograms).

    Returns (freqs, psd_db):
        freqs   — frequency bins in Hz, centered at 0 (fftshift'd), length nperseg
        psd_db  — power spectral density in dB, same length as freqs
    """
    n = complex_samples.shape[0]
    nperseg = min(nperseg, n)
    step = max(1, int(nperseg * (1 - overlap_frac)))
    window = _hann(nperseg)
    win_power = np.sum(window ** 2)

    accum = np.zeros(nperseg, dtype=np.float64)
    n_segments = 0

    for start in range(0, n - nperseg + 1, step):
        seg = complex_samples[start : start + nperseg] * window
        spec = np.fft.fftshift(np.fft.fft(seg))
        accum += (np.abs(spec) ** 2) / (win_power * sample_rate)
        n_segments += 1

    if n_segments == 0:
        # signal shorter than nperseg — fall back to a single zero-padded segment
        seg = np.zeros(nperseg, dtype=np.complex64)
        seg[:n] = complex_samples * _hann(n)
        spec = np.fft.fftshift(np.fft.fft(seg))
        accum = (np.abs(spec) ** 2) / (win_power * sample_rate)
        n_segments = 1

    psd = accum / n_segments
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / sample_rate))
    return freqs.astype(np.float32), _to_db(psd).astype(np.float32)


def compute_spectrogram(
    complex_samples: np.ndarray,
    sample_rate: float,
    nperseg: int = DEFAULT_NPERSEG_SPEC,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Short-time Fourier transform spectrogram (magnitude, in dB).

    Returns (times, freqs, spec_db):
        times    — time bin centers in seconds, length n_time_bins
        freqs    — frequency bins in Hz, centered at 0, length nperseg
        spec_db  — shape (nperseg, n_time_bins), dB magnitude, ready for a
                   waterfall/heatmap plot (e.g. pyqtgraph ImageItem)
    """
    n = complex_samples.shape[0]
    nperseg = min(nperseg, n)
    step = max(1, int(nperseg * (1 - overlap_frac)))
    window = _hann(nperseg)

    starts = list(range(0, n - nperseg + 1, step))
    if not starts:
        starts = [0]
        seg_pad = np.zeros(nperseg, dtype=np.complex64)
        seg_pad[:n] = complex_samples
        complex_samples = seg_pad
        n = nperseg

    cols = []
    for start in starts:
        seg = complex_samples[start : start + nperseg] * window
        spec = np.fft.fftshift(np.fft.fft(seg))
        cols.append(_to_db(np.abs(spec) ** 2))

    spec_db = np.stack(cols, axis=1)  # (nperseg, n_time_bins)
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / sample_rate)).astype(
        np.float32
    )
    times = (np.array(starts, dtype=np.float64) + nperseg / 2) / sample_rate
    return times.astype(np.float32), freqs, spec_db.astype(np.float32)


def estimate_occupied_bandwidth(
    freqs: np.ndarray, psd_db: np.ndarray, threshold_db: float = -20.0
) -> tuple[float, float]:
    """
    Estimate occupied bandwidth and center-frequency offset from a PSD.

    A sample is counted as "occupied" if it's within threshold_db of the
    peak. Returns (occupied_bandwidth_hz, center_freq_offset_hz), where the
    offset is the energy centroid's distance from 0 Hz (useful for spotting
    a frequency-offset capture, e.g. mistuned SDR center frequency).

    threshold_db=-20 is a reasonable default for a clean capture; noisy /
    low-SNR captures may need a shallower threshold (e.g. -10 dB) to avoid
    the estimate being dominated by noise floor ripple.
    """
    peak_db = np.max(psd_db)
    occupied_mask = psd_db >= (peak_db + threshold_db)

    if not np.any(occupied_mask):
        return 0.0, 0.0

    occupied_freqs = freqs[occupied_mask]
    bandwidth = float(occupied_freqs.max() - occupied_freqs.min())

    # Energy-weighted centroid (linear power, not dB) for the offset estimate
    linear_power = 10 ** (psd_db / 10.0)
    total_power = np.sum(linear_power[occupied_mask])
    centroid = (
        float(np.sum(freqs[occupied_mask] * linear_power[occupied_mask]) / total_power)
        if total_power > 0
        else 0.0
    )

    return bandwidth, centroid


def analyze(
    complex_samples: np.ndarray,
    sample_rate: float,
    psd_nperseg: int = DEFAULT_NPERSEG_PSD,
    spec_nperseg: int = DEFAULT_NPERSEG_SPEC,
    bandwidth_threshold_db: float = -20.0,
) -> dict:
    """
    Convenience wrapper: compute PSD, spectrogram, and occupied-bandwidth
    estimate in one call. This is the function the GUI's params/spectrogram
    panels should call.
    """
    freqs, psd_db = compute_psd(complex_samples, sample_rate, nperseg=psd_nperseg)
    spec_times, spec_freqs, spec_db = compute_spectrogram(
        complex_samples, sample_rate, nperseg=spec_nperseg
    )
    bandwidth_hz, offset_hz = estimate_occupied_bandwidth(
        freqs, psd_db, threshold_db=bandwidth_threshold_db
    )

    return {
        "freqs": freqs,
        "psd_db": psd_db,
        "spec_times": spec_times,
        "spec_freqs": spec_freqs,
        "spec_db": spec_db,
        "occupied_bandwidth_hz": bandwidth_hz,
        "center_freq_offset_hz": offset_hz,
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    try:
        from file_loader import load_file
    except ImportError:
        print("Run this from inside preprocessing/, next to file_loader.py")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python spectral_features.py <path-to-.wav-or-.iq-file>")
        sys.exit(1)

    windows, meta = load_file(sys.argv[1])
    if meta["sample_rate"] is None:
        print("No sample rate available (raw .iq with none supplied) — "
              "pass raw_iq_sample_rate to load_file() first.")
        sys.exit(1)

    # Re-load full (unwindowed) signal for a whole-capture spectral view
    ext = sys.argv[1].lower()
    if ext.endswith(".wav"):
        from file_loader import load_wav
        complex_samples, sample_rate = load_wav(sys.argv[1])
    else:
        from file_loader import load_iq_raw
        complex_samples, sample_rate = load_iq_raw(
            sys.argv[1], sample_rate=meta["sample_rate"]
        )

    features = analyze(complex_samples, sample_rate)
    print(f"sample_rate:            {sample_rate} Hz")
    print(f"occupied_bandwidth_hz:  {features['occupied_bandwidth_hz']:.1f}")
    print(f"center_freq_offset_hz:  {features['center_freq_offset_hz']:.1f}")
    print(f"psd shape:              {features['psd_db'].shape}")
    print(f"spectrogram shape:      {features['spec_db'].shape} "
          f"(freqs x times: {features['spec_freqs'].shape[0]} x {features['spec_times'].shape[0]})")