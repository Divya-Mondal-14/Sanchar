"""
preprocessing/file_loader.py

Parses raw RF capture files (.wav or .iq) into windowed complex-sample
arrays shaped (1024, 2) — [I, Q] per timestep — ready to hand to
models/inference.py::predict_modulation().

Supported inputs
-----------------
.wav   Stereo WAV = IQ capture (left channel = I, right channel = Q).
       This is the convention used by SDR#, GQRX, and most SDR WAV dumps.
       Mono WAV is treated as a real-valued signal (Q filled with zeros).

.iq    Headerless raw binary, interleaved I,Q,I,Q,... samples.
       No self-describing sample rate or dtype, so both must be supplied
       by the caller (or left at the defaults below and corrected once
       you know your capture format). Common SDR raw formats:
         - 'uint8'   : RTL-SDR raw capture (offset binary, unsigned, 0-255)
         - 'int16'   : GNU Radio short-complex dumps
         - 'float32' : GNU Radio gr_complex / most generic raw IQ dumps

If your project's actual capture format differs from these assumptions
(e.g. I/Q channel order is swapped, or a different WAV convention is
used), adjust CHANNEL_ORDER / the dtype default below — everything else
in this file is independent of that choice.

Usage
-----
    from file_loader import load_file

    windows, meta = load_file("capture.wav")
    # windows: np.ndarray, shape (num_windows, 1024, 2), dtype float32
    # meta: dict with sample_rate, source_path, num_windows, dropped_samples

    for w in windows:
        result = predict_modulation(w)   # from models/inference.py
"""

from __future__ import annotations

import os
import wave
import numpy as np

# ---------------------------------------------------------------------------
# Config / assumptions (adjust here if your capture convention differs)
# ---------------------------------------------------------------------------

WINDOW_SIZE = 1024          # samples per window, matches classifier input length
CHANNEL_ORDER = ("I", "Q")  # left=I, right=Q for stereo WAV
RAW_IQ_DEFAULT_DTYPE = "float32"
RAW_IQ_DEFAULT_SAMPLE_RATE = None  # must be supplied by caller for .iq files

_WAV_DTYPE_BY_SAMPWIDTH = {
    1: np.uint8,   # 8-bit WAV is unsigned
    2: np.int16,
    4: np.int32,
}

_RAW_DTYPE_MAP = {
    "uint8": np.uint8,
    "int8": np.int8,
    "int16": np.int16,
    "float32": np.float32,
}


# ---------------------------------------------------------------------------
# WAV loading
# ---------------------------------------------------------------------------

def load_wav(path: str) -> tuple[np.ndarray, int]:
    """
    Read a .wav file and return (complex_samples, sample_rate).

    Stereo -> complex I/Q (left=I, right=Q), normalized to float32 in [-1, 1].
    Mono   -> real signal placed in the I channel, Q = 0.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if samp_width not in _WAV_DTYPE_BY_SAMPWIDTH:
        raise ValueError(
            f"Unsupported WAV sample width: {samp_width} bytes "
            f"(supported: 1, 2, 4 bytes/sample)"
        )

    dtype = _WAV_DTYPE_BY_SAMPWIDTH[samp_width]
    data = np.frombuffer(raw, dtype=dtype)

    # Normalize to float32 in [-1, 1]
    if dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        max_val = float(np.iinfo(dtype).max)
        data = data.astype(np.float32) / max_val

    if n_channels == 2:
        data = data.reshape(-1, 2)
        i_samples = data[:, 0]
        q_samples = data[:, 1]
    elif n_channels == 1:
        i_samples = data
        q_samples = np.zeros_like(data)
    else:
        raise ValueError(
            f"Unsupported channel count: {n_channels} "
            f"(expected mono=1 or IQ-stereo=2)"
        )

    complex_samples = i_samples + 1j * q_samples
    return complex_samples.astype(np.complex64), sample_rate


# ---------------------------------------------------------------------------
# Raw .iq loading
# ---------------------------------------------------------------------------

def load_iq_raw(
    path: str,
    dtype: str = RAW_IQ_DEFAULT_DTYPE,
    sample_rate: int | None = RAW_IQ_DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, int | None]:
    """
    Read a headerless raw .iq file (interleaved I,Q,I,Q,... samples).

    dtype must match how the capture was recorded — there's no header to
    infer it from. sample_rate is likewise unknown from the file itself
    and should be supplied if you have it (e.g. from your SDR capture
    settings); it's passed through in the returned metadata but is not
    required to produce correct windows.
    """
    if dtype not in _RAW_DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Supported: {list(_RAW_DTYPE_MAP)}"
        )

    np_dtype = _RAW_DTYPE_MAP[dtype]
    data = np.fromfile(path, dtype=np_dtype)

    if data.size % 2 != 0:
        # Odd trailing byte/sample — drop it rather than fail
        data = data[:-1]

    data = data.reshape(-1, 2)

    # Normalize to float32 in [-1, 1] where the dtype is fixed-point
    if np_dtype == np.uint8:
        iq = (data.astype(np.float32) - 128.0) / 128.0
    elif np_dtype == np.int8:
        iq = data.astype(np.float32) / 128.0
    elif np_dtype == np.int16:
        iq = data.astype(np.float32) / 32768.0
    else:  # float32 raw dumps are assumed already scaled
        iq = data.astype(np.float32)

    complex_samples = iq[:, 0] + 1j * iq[:, 1]
    return complex_samples.astype(np.complex64), sample_rate


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def window_samples(
    complex_samples: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
    drop_last: bool = True,
) -> np.ndarray:
    """
    Slice a 1D complex sample array into non-overlapping (or strided)
    windows shaped (num_windows, window_size, 2) as [I, Q] float32 —
    the format predict_modulation() expects per window.

    stride=None -> non-overlapping windows (stride == window_size).
    drop_last=True -> discard any trailing partial window that doesn't
    fill window_size samples (rather than zero-padding it), since a
    padded window could otherwise be misread as a valid low-amplitude
    signal segment by the classifier.
    """
    if stride is None:
        stride = window_size

    n_samples = complex_samples.shape[0]
    if n_samples < window_size:
        return np.empty((0, window_size, 2), dtype=np.float32)

    starts = range(0, n_samples - window_size + 1, stride)
    windows = np.stack(
        [complex_samples[s : s + window_size] for s in starts], axis=0
    )

    iq_windows = np.stack([windows.real, windows.imag], axis=-1).astype(np.float32)

    if not drop_last:
        remainder_start = (
            starts[-1] + stride if len(starts) else 0  # type: ignore[index]
        )
        leftover = complex_samples[remainder_start:]
        if leftover.size > 0:
            pad = np.zeros(window_size - leftover.size, dtype=np.complex64)
            padded = np.concatenate([leftover, pad])
            padded_iq = np.stack([padded.real, padded.imag], axis=-1).astype(
                np.float32
            )
            iq_windows = np.concatenate([iq_windows, padded_iq[None, ...]], axis=0)

    return iq_windows


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def load_file(
    path: str,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
    drop_last: bool = True,
    raw_iq_dtype: str = RAW_IQ_DEFAULT_DTYPE,
    raw_iq_sample_rate: int | None = RAW_IQ_DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, dict]:
    """
    Load a .wav or .iq capture file and return (windows, meta).

    windows: np.ndarray, shape (num_windows, window_size, 2), dtype float32
    meta: {
        'source_path': str,
        'sample_rate': int | None,
        'total_samples': int,
        'num_windows': int,
        'window_size': int,
    }
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".wav":
        complex_samples, sample_rate = load_wav(path)
    elif ext == ".iq":
        complex_samples, sample_rate = load_iq_raw(
            path, dtype=raw_iq_dtype, sample_rate=raw_iq_sample_rate
        )
    else:
        raise ValueError(f"Unsupported file extension '{ext}' (expected .wav or .iq)")

    windows = window_samples(
        complex_samples, window_size=window_size, stride=stride, drop_last=drop_last
    )

    meta = {
        "source_path": path,
        "sample_rate": sample_rate,
        "total_samples": int(complex_samples.shape[0]),
        "num_windows": int(windows.shape[0]),
        "window_size": window_size,
    }
    return windows, meta


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python file_loader.py <path-to-.wav-or-.iq-file>")
        sys.exit(1)

    windows, meta = load_file(sys.argv[1])
    print(f"Loaded: {meta}")
    print(f"Windows array shape: {windows.shape}, dtype: {windows.dtype}")
    if windows.shape[0] > 0:
        print(f"First window sample (I, Q) at t=0: {windows[0, 0]}")