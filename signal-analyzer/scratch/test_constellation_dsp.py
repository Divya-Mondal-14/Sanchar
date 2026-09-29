"""
scratch/test_constellation_dsp.py
Test RRC filtering, Costas loop, and symbol timing recovery for QPSK and BPSK.
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
]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from file_loader import load_file, load_wav
from spectral_features import analyze as spectral_analyze
from demod import estimate_samples_per_symbol

def rrc_taps(sps: int, span: int = 8, alpha: float = 0.35) -> np.ndarray:
    n_taps = span * sps + 1
    t = np.arange(n_taps, dtype=np.float64) - n_taps // 2
    t_norm = t / sps
    eps = 1e-8
    h = np.zeros(n_taps, dtype=np.float64)
    for i, tn in enumerate(t_norm):
        if abs(tn) < eps:
            h[i] = 1.0 + alpha * (4.0 / np.pi - 1.0)
        elif abs(abs(tn) - 1.0 / (4.0 * alpha)) < eps:
            h[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            num = np.sin(np.pi * tn * (1.0 - alpha)) + 4.0 * alpha * tn * np.cos(np.pi * tn * (1.0 + alpha))
            den = np.pi * tn * (1.0 - (4.0 * alpha * tn) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h.astype(np.float32)

def recover_constellation_symbols(
    complex_samples: np.ndarray,
    sample_rate: float,
    sps: int = 8,
    mod_class: str = "QPSK",
    cfo_hz: float | None = None,
    loop_bw: float = 0.02,
    damping: float = 0.707,
) -> np.ndarray:
    """
    Apply coarse CFO correction, matched filtering (RRC), symbol timing decimation,
    and Costas carrier phase tracking to recover tightly-clustered constellation symbols.
    """
    cs = np.asarray(complex_samples, dtype=np.complex64)
    if len(cs) < 32:
        return np.stack([cs.real, cs.imag], axis=-1)

    # 1. Coarse CFO Correction if CFO is known
    fs = float(sample_rate) if sample_rate and sample_rate > 0 else 1_000_000.0
    if cfo_hz is not None and abs(cfo_hz) > 0.1:
        t = np.arange(len(cs), dtype=np.float64) / fs
        cs = cs * np.exp(-1j * 2.0 * np.pi * cfo_hz * t).astype(np.complex64)

    # 2. Matched Filter (RRC / LPF)
    sps = max(2, int(sps))
    h = rrc_taps(sps, span=8, alpha=0.35)
    cs_filtered = np.convolve(cs, h, mode="same")

    # 3. Fine Carrier Tracking (Sample-Rate or Symbol-Rate Costas Loop)
    # Apply Costas loop with proportional-integral tracking
    mod_upper = mod_class.upper()
    is_qpsk = "QPSK" in mod_upper or "4QAM" in mod_upper
    is_bpsk = "BPSK" in mod_upper

    # Symbol timing: find optimal sampling instant with maximum eye opening
    n_syms = len(cs_filtered) // sps
    if n_syms > 4:
        # Search over sps possible strobe phases
        phase_metrics = []
        for p_idx in range(sps):
            sub = cs_filtered[p_idx : p_idx + n_syms * sps : sps]
            # Kurtosis / dispersion metric: minimize dispersion around unit circle
            r2 = np.abs(sub) ** 2
            mean_r2 = np.mean(r2) + 1e-12
            dispersion = np.mean((r2 - mean_r2) ** 2) / (mean_r2 ** 2)
            # We want minimum dispersion (or maximum eye opening)
            phase_metrics.append(-dispersion)
        best_phase = int(np.argmax(phase_metrics))
    else:
        best_phase = 0

    sym_samples = cs_filtered[best_phase : best_phase + n_syms * sps : sps]

    # 4. Symbol-Rate 2nd-Order Costas Loop (Carrier Phase & Residual Frequency Tracking)
    theta = loop_bw / (damping + 0.25 / damping)
    denom = 1.0 + 2.0 * damping * theta + theta * theta
    alpha_gain = (4.0 * damping * theta) / denom
    beta_gain = (4.0 * theta * theta) / denom

    phase = 0.0
    freq = 0.0
    recovered_syms = np.zeros(len(sym_samples), dtype=np.complex64)

    for k, s in enumerate(sym_samples):
        s_rot = s * np.exp(-1j * phase)
        recovered_syms[k] = s_rot

        I_val = s_rot.real
        Q_val = s_rot.imag
        if is_qpsk:
            # 4th-order QPSK Costas PED: sign(I)*Q - sign(Q)*I
            e = np.sign(I_val) * Q_val - np.sign(Q_val) * I_val
        elif is_bpsk:
            # BPSK Costas PED: sign(I)*Q
            e = np.sign(I_val) * Q_val
        else:
            # General PSK
            e = np.sign(I_val) * Q_val

        e = np.clip(e, -1.5, 1.5)

        freq += beta_gain * e
        freq = np.clip(freq, -0.1, 0.1)
        phase += freq + alpha_gain * e

    # Discard initial acquisition / settling transients (first 40 symbols)
    settle_count = min(50, len(recovered_syms) // 4)
    settled = recovered_syms[settle_count:] if len(recovered_syms) > settle_count + 16 else recovered_syms

    # 5. Constellation Phase Alignment to Standard Grid
    if is_qpsk and len(settled) > 0:
        # Standard QPSK targets are (+-0.707, +-0.707), i.e. angles at pi/4, 3pi/4, -3pi/4, -pi/4
        # Compute 4th power phase to estimate grid rotation mod pi/2
        z4 = np.mean(settled ** 4)
        if abs(z4) > 1e-6:
            # For ideal QPSK (exp(j*(pi/4 + k*pi/2))), z^4 = exp(j*pi) = -1 = exp(j*pi)
            angle_error = (np.angle(z4) - np.pi) / 4.0
            settled = settled * np.exp(-1j * angle_error)
    elif is_bpsk and len(settled) > 0:
        # Standard BPSK targets are (+-1, 0), i.e. angles 0, pi
        # Compute 2nd power phase to estimate grid rotation mod pi
        z2 = np.mean(settled ** 2)
        if abs(z2) > 1e-6:
            angle_error = np.angle(z2) / 2.0
            settled = settled * np.exp(-1j * angle_error)

    # 6. Normalize RMS Power to 1.0 (Unit Circle)
    rms = float(np.sqrt(np.mean(settled.real ** 2 + settled.imag ** 2)))
    if rms > 1e-9:
        settled = settled / rms

    return np.stack([settled.real, settled.imag], axis=-1).astype(np.float32)

def test_qpsk():
    qpsk_wav = r"c:\Sanchar\signal-analyzer\synthetic_QPSK_1000kHz_SNR20dB.wav"
    complex_samples, sr = load_wav(qpsk_wav)
    print(f"\nLoaded QPSK: {len(complex_samples)} samples, sr={sr}")
    
    spectral = spectral_analyze(complex_samples, sr)
    obw = spectral["occupied_bandwidth_hz"]
    cfo = spectral["center_freq_offset_hz"]
    print(f"OBW: {obw}, CFO: {cfo}")
    
    baud_hint = (obw / 1.35) if (obw is not None and obw > 1000) else None
    sps = estimate_samples_per_symbol(complex_samples, float(sr), baud_rate_hint=baud_hint)
    print(f"Estimated SPS: {sps}")

    syms = recover_constellation_symbols(complex_samples, sr, sps, "QPSK", cfo_hz=cfo, loop_bw=0.035)
    print(f"Recovered {len(syms)} constellation symbols. Shape: {syms.shape}")
    print(f"Mean I: {np.mean(syms[:, 0]):.4f}, Std I: {np.std(syms[:, 0]):.4f}")
    print(f"Mean Q: {np.mean(syms[:, 1]):.4f}, Std Q: {np.std(syms[:, 1]):.4f}")
    
    # Calculate cluster tightness around ideal points (+-0.707, +-0.707)
    ideal_qpsk = np.array([[0.7071, 0.7071], [-0.7071, 0.7071], [-0.7071, -0.7071], [0.7071, -0.7071]])
    dists = []
    for pt in syms:
        d = np.min(np.sqrt(np.sum((ideal_qpsk - pt) ** 2, axis=1)))
        dists.append(d)
    print(f"Average distance to nearest ideal constellation point: {np.mean(dists):.4f} (tightness metric)")

def test_bpsk():
    bpsk_wav = r"c:\Sanchar\signal-analyzer\synthetic_BPSK_1000kHz_SNR20dB.wav"
    complex_samples, sr = load_wav(bpsk_wav)
    print(f"\nLoaded BPSK: {len(complex_samples)} samples, sr={sr}")
    
    spectral = spectral_analyze(complex_samples, sr)
    obw = spectral["occupied_bandwidth_hz"]
    cfo = spectral["center_freq_offset_hz"]
    print(f"OBW: {obw}, CFO: {cfo}")
    
    baud_hint = (obw / 1.35) if (obw is not None and obw > 1000) else None
    sps = estimate_samples_per_symbol(complex_samples, float(sr), baud_rate_hint=baud_hint)
    print(f"Estimated SPS: {sps}")

    syms = recover_constellation_symbols(complex_samples, sr, sps, "BPSK", cfo_hz=cfo, loop_bw=0.035)
    print(f"Recovered {len(syms)} constellation symbols. Shape: {syms.shape}")
    print(f"Mean I: {np.mean(syms[:, 0]):.4f}, Std I: {np.std(syms[:, 0]):.4f}")
    print(f"Mean Q: {np.mean(syms[:, 1]):.4f}, Std Q: {np.std(syms[:, 1]):.4f}")
    
    # Calculate cluster tightness around ideal points (+-1.0, 0.0)
    ideal_bpsk = np.array([[1.0, 0.0], [-1.0, 0.0]])
    dists = []
    for pt in syms:
        d = np.min(np.sqrt(np.sum((ideal_bpsk - pt) ** 2, axis=1)))
        dists.append(d)
    print(f"Average distance to nearest ideal constellation point: {np.mean(dists):.4f} (tightness metric)")

if __name__ == "__main__":
    test_qpsk()
    test_bpsk()


