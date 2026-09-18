"""
gnuradio_pipeline/demod.py

Demodulates IQ samples using GNU Radio's Python API (gr-digital blocks).
Routes to the correct demodulator based on the modulation class string
returned by models/inference.py::predict_modulation().

Supported classes (matching label_map.json):
    'BPSK'  -> gr.digital.bpsk_demod (via constellation decoder)
    'QPSK'  -> gr.digital.qpsk_demod (via constellation decoder)
    'GMSK'  -> gr.digital.gmsk_demod  (GMSK stands in for FSK in the
               RadioML2018.01A dataset used for training)

GNU Radio is used directly via its Python bindings — NOT via GRC canvas.
Each demodulation is run as an ad-hoc gr.top_block with:
    vector_source -> demodulator block -> vector_sink

Usage
-----
    from gnuradio_pipeline.demod import demodulate

    bits = demodulate(
        iq_samples,                # np.ndarray (N, 2) float32 [I, Q]
        modulation_class='QPSK',   # from predict_modulation()['class']
        sample_rate=2_000_000,     # Hz
        samples_per_symbol=8,      # must be >= 2
    )
    # bits: np.ndarray, dtype uint8, values 0/1

Public API
----------
    demodulate(iq_samples, modulation_class, sample_rate, samples_per_symbol) -> np.ndarray
    estimate_samples_per_symbol(complex_samples, sample_rate) -> int
"""

from __future__ import annotations

import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# GNU Radio path auto-discovery
# ---------------------------------------------------------------------------
# When the app is launched from the project venv but GNU Radio lives in the
# signal_analyzer conda env, the bindings won't be on sys.path.
# This block probes several well-known locations and injects them before
# attempting the import so the user never needs to set PYTHONPATH manually.

def _inject_gnuradio_paths() -> None:
    """Find the signal_analyzer conda env and add its paths to the process."""
    # If gnuradio.gr is already fully functional, nothing to do
    try:
        import gnuradio.gr  # noqa: F401
        return
    except (ImportError, OSError):
        pass

    home = os.path.expanduser("~")

    # Search candidates (in priority order)
    candidates: list[str] = []

    # 1. Active conda env (CONDA_PREFIX set by `conda activate`)
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidates.append(conda_prefix)


    # 3. Common conda root patterns
    for conda_root in [
        os.path.join(home, "radioconda"),
        os.path.join(home, "miniconda3"),
        os.path.join(home, "anaconda3"),
        r"C:\ProgramData\miniconda3",
        r"C:\ProgramData\anaconda3",
    ]:
        candidates.append(os.path.join(conda_root, "envs", "signal_analyzer"))

    for env_root in candidates:
        if not os.path.isdir(env_root):
            continue

        site_pkgs = os.path.join(env_root, "Lib", "site-packages")
        dll_dirs = [
            os.path.join(env_root, "Library", "bin"),
            os.path.join(env_root, "Library", "lib"),
            os.path.join(env_root, "bin"),
        ]

        for dll_dir in dll_dirs:
            if os.path.isdir(dll_dir):
                os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(dll_dir)
                    except (AttributeError, OSError):
                        pass

        if os.path.isdir(site_pkgs) and site_pkgs not in sys.path:
            sys.path.insert(0, site_pkgs)

        try:
            import gnuradio.gr  # noqa: F401
            return  # success — this candidate worked
        except (ImportError, OSError):
            if site_pkgs in sys.path:
                sys.path.remove(site_pkgs)


_inject_gnuradio_paths()


# GNU Radio imports — hard requirement (no numpy fallback per project spec)
try:
    import gnuradio.gr as gr
    import gnuradio.digital as digital
    import gnuradio.blocks as blocks
    from gnuradio import analog
    _GR_AVAILABLE = True
except ImportError as _gr_err:
    raise ImportError(
        "GNU Radio Python bindings could not be found.\n\n"
        "Tried: sys.path, CONDA_PREFIX, "
        r"C:\Users\3055\radioconda\envs\signal_analyzer, "
        "and common conda root locations.\n\n"
        "Fix options:\n"
        "  A) conda activate signal_analyzer  then  python main.py\n"
        "  B) conda run -n signal_analyzer python main.py\n\n"
        f"Original error: {_gr_err}"
    ) from _gr_err


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iq_to_gr_complex(iq_samples: np.ndarray) -> np.ndarray:
    """Convert (N, 2) float32 [I, Q] to GNU Radio's interleaved complex64."""
    return (iq_samples[:, 0] + 1j * iq_samples[:, 1]).astype(np.complex64)


def estimate_samples_per_symbol(
    complex_samples: np.ndarray,
    sample_rate: float,
    baud_rate_hint: float | None = None,
) -> int:
    """
    Estimate samples-per-symbol (sps) from a complex-sample array.

    If baud_rate_hint is supplied (Hz), sps = round(sample_rate / baud_rate_hint),
    clamped to [2, 64].

    Without a hint: uses Oerder-Meyr cyclostationary clock recovery (Fourier transform
    of the squared envelope |s(t)|^2) to identify the discrete symbol clock tone.

    Returns an integer in [2, 64].
    """
    if baud_rate_hint is not None and baud_rate_hint > 0:
        sps = max(2, min(64, round(sample_rate / baud_rate_hint)))
        return int(sps)

    if len(complex_samples) < 16:
        return 8

    # Oerder-Meyr / non-linear envelope clock recovery:
    # The squared envelope of a pulse-shaped PSK signal exhibits a spectral line at baud_rate
    cs = complex_samples[:16384] if len(complex_samples) > 16384 else complex_samples
    sq = np.abs(cs).astype(np.float64) ** 2
    sq -= np.mean(sq)

    n = len(sq)
    F = np.abs(np.fft.rfft(sq))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    # Search for clock tone corresponding to SPS in [2, 64]
    # Frequency range: sample_rate / 64 to sample_rate / 2
    min_freq = sample_rate / 64.0
    max_freq = sample_rate / 2.0
    mask = (freqs >= min_freq) & (freqs <= max_freq)

    if np.any(mask) and np.max(F[mask]) > 0:
        peak_freq = freqs[mask][np.argmax(F[mask])]
        if peak_freq > 0:
            sps_float = sample_rate / peak_freq
            sps = int(round(sps_float))
            return max(2, min(64, sps))

    return 8


# ---------------------------------------------------------------------------
# GNU Radio flowgraph builders
# ---------------------------------------------------------------------------

class _BpskDemodFlowgraph(gr.top_block):
    """
    BPSK demodulator flowgraph.
        vector_source(complex) -> digital.bpsk_demod -> vector_sink(bytes)
    """
    def __init__(self, iq_complex: np.ndarray, samples_per_symbol: int):
        super().__init__()

        constellation = digital.constellation_bpsk().base()

        self._src = blocks.vector_source_c(
            iq_complex.tolist(), repeat=False
        )
        self._demod = digital.generic_demod(
            constellation=constellation,
            differential=False,
            samples_per_symbol=samples_per_symbol,
            pre_diff_code=True,
            freq_bw=2 * np.pi / 100.0,
            timing_bw=2 * np.pi / 100.0,
            verbose=False,
            log=False,
        )
        self._sink = blocks.vector_sink_b()

        self.connect(self._src, self._demod, self._sink)

    def get_bits(self) -> np.ndarray:
        return np.array(self._sink.data(), dtype=np.uint8)


class _QpskDemodFlowgraph(gr.top_block):
    """
    QPSK demodulator flowgraph.
        vector_source(complex) -> digital.qpsk_demod -> vector_sink(bytes)
    """
    def __init__(self, iq_complex: np.ndarray, samples_per_symbol: int):
        super().__init__()

        constellation = digital.constellation_qpsk().base()

        self._src = blocks.vector_source_c(
            iq_complex.tolist(), repeat=False
        )
        self._demod = digital.generic_demod(
            constellation=constellation,
            differential=False,
            samples_per_symbol=samples_per_symbol,
            pre_diff_code=True,
            freq_bw=2 * np.pi / 100.0,
            timing_bw=2 * np.pi / 100.0,
            verbose=False,
            log=False,
        )
        self._sink = blocks.vector_sink_b()

        self.connect(self._src, self._demod, self._sink)

    def get_bits(self) -> np.ndarray:
        return np.array(self._sink.data(), dtype=np.uint8)


class _GmskDemodFlowgraph(gr.top_block):
    """
    GMSK demodulator flowgraph (used for the 'GMSK' / FSK class).
        vector_source(complex) -> digital.gmsk_demod -> vector_sink(bytes)
    """
    def __init__(
        self,
        iq_complex: np.ndarray,
        samples_per_symbol: int,
        bt: float = 0.35,
    ):
        super().__init__()

        self._src = blocks.vector_source_c(
            iq_complex.tolist(), repeat=False
        )
        self._demod = digital.gmsk_demod(
            samples_per_symbol=samples_per_symbol,
            gain_mu=0.175,
            mu=0.5,
            omega_relative_limit=0.005,
            freq_error=0.0,
            verbose=False,
            log=False,
        )
        self._sink = blocks.vector_sink_b()

        self.connect(self._src, self._demod, self._sink)

    def get_bits(self) -> np.ndarray:
        return np.array(self._sink.data(), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Public demodulate() function
# ---------------------------------------------------------------------------

_SUPPORTED_CLASSES = {"BPSK", "QPSK", "GMSK"}


def demodulate(
    iq_samples: np.ndarray,
    modulation_class: str,
    sample_rate: float,
    samples_per_symbol: int | None = None,
    baud_rate_hint: float | None = None,
) -> np.ndarray:
    """
    Demodulate IQ samples using the appropriate GNU Radio flowgraph.

    Parameters
    ----------
    iq_samples : np.ndarray, shape (N, 2), dtype float32
        [I, Q] samples as produced by file_loader.window_samples or any
        (N, 2) slice of the windowed array concatenated back to a stream.
        You can also pass the full complex_samples array reshaped:
            iq = np.stack([cs.real, cs.imag], axis=-1)
    modulation_class : str
        One of 'BPSK', 'QPSK', 'GMSK' — as returned by predict_modulation()['class'].
    sample_rate : float
        Capture sample rate in Hz (from file metadata or user input).
    samples_per_symbol : int or None
        If None, estimated automatically from the signal. Minimum 2.
    baud_rate_hint : float or None
        Optional baud-rate hint (Hz) used only when samples_per_symbol is None.

    Returns
    -------
    np.ndarray, dtype uint8, shape (M,) — demodulated bit stream (0/1 values).
    """
    mod = modulation_class.upper()
    if mod not in _SUPPORTED_CLASSES:
        raise ValueError(
            f"Unsupported modulation class '{modulation_class}'. "
            f"Supported: {_SUPPORTED_CLASSES}"
        )

    if iq_samples.ndim != 2 or iq_samples.shape[1] != 2:
        raise ValueError(
            f"iq_samples must have shape (N, 2), got {iq_samples.shape}"
        )

    iq_complex = _iq_to_gr_complex(iq_samples)

    # --- Estimate samples-per-symbol if not provided ---
    if samples_per_symbol is None:
        samples_per_symbol = estimate_samples_per_symbol(
            iq_complex, sample_rate, baud_rate_hint=baud_rate_hint
        )

    samples_per_symbol = max(2, int(samples_per_symbol))

    # --- Build and run the flowgraph ---
    if mod == "BPSK":
        fg = _BpskDemodFlowgraph(iq_complex, samples_per_symbol)
    elif mod == "QPSK":
        fg = _QpskDemodFlowgraph(iq_complex, samples_per_symbol)
    else:  # GMSK
        fg = _GmskDemodFlowgraph(iq_complex, samples_per_symbol)

    fg.run()
    bits = fg.get_bits()
    fg.stop()
    fg.wait()

    return bits


# ---------------------------------------------------------------------------
# Self-test (python gnuradio_pipeline/demod.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("demod.py — GNU Radio demodulation self-test")
    print(f"GNU Radio available: {_GR_AVAILABLE}")

    rng = np.random.default_rng(0)
    N_SYMS = 128
    SPS = 8
    N = N_SYMS * SPS
    fs = 1_000_000.0
    noise_std = 0.05

    for mod in ("BPSK", "QPSK", "GMSK"):
        print(f"\n--- {mod} ---")
        t = np.arange(N) / fs

        if mod == "BPSK":
            tx_bits = rng.integers(0, 2, size=N_SYMS)
            phases = np.repeat(tx_bits * np.pi, SPS).astype(np.float64)
            cs = np.exp(1j * phases).astype(np.complex64)
        elif mod == "QPSK":
            tx_bits = rng.integers(0, 4, size=N_SYMS)
            phases = np.repeat(tx_bits * np.pi / 2.0, SPS).astype(np.float64)
            cs = np.exp(1j * phases).astype(np.complex64)
        else:  # GMSK — FM-modulated NRZ
            tx_bits = rng.integers(0, 2, size=N_SYMS)
            nrz = np.repeat((tx_bits * 2 - 1).astype(np.float64), SPS)
            phase = np.cumsum(nrz) * (np.pi / SPS)
            cs = np.exp(1j * phase).astype(np.complex64)

        noise = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * noise_std
        cs = (cs + noise).astype(np.complex64)
        iq = np.stack([cs.real, cs.imag], axis=-1).astype(np.float32)

        try:
            bits_out = demodulate(iq, mod, sample_rate=fs, samples_per_symbol=SPS)
            print(f"  Demodulated {len(bits_out)} bits (input {N} samples, sps={SPS})")
            if len(bits_out) == 0:
                print("  WARNING: demodulator produced zero output bits")
            else:
                print(f"  First 16 output bits: {bits_out[:16].tolist()}")
                print("  Demodulation ran successfully (bit-accuracy depends on phase lock).")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            sys.exit(1)

    print("\nSelf-test complete.")
    sys.exit(0)
