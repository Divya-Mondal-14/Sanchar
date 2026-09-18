"""
preprocessing/synthetic_iq.py

Synthetic IQ signal generator for the SIH Signal Analyzer.

Generates realistic BPSK, QPSK, or GMSK baseband IQ captures — useful for
pipeline testing without a real SDR capture.

Each generated signal includes:
  - Root-raised-cosine pulse shaping (via a Hann-windowed sinc approximation)
  - Additive white Gaussian noise at a configurable SNR
  - Optional carrier frequency offset (to test CFO estimation)
  - Output as .wav (stereo IQ, left=I, right=Q) or raw .iq (float32 interleaved)

Public API
----------
    generate(modulation, n_symbols, sample_rate, samples_per_symbol,
             snr_db, carrier_offset_hz, seed) -> (complex_samples, tx_bits, meta)

    save_wav(complex_samples, sample_rate, path) -> path
    save_iq_raw(complex_samples, path) -> path
    generate_and_save(...)  -> (path, tx_bits, meta)
"""

from __future__ import annotations

import os
import struct
import wave

import numpy as np


# ---------------------------------------------------------------------------
# Pulse shaping (simple raised-cosine via Hann window)
# ---------------------------------------------------------------------------

def _rrc_filter(sps: int, span: int = 8, rolloff: float = 0.35) -> np.ndarray:
    """
    Approximate root-raised cosine filter (Hann-windowed sinc).
    span  — filter length in symbols
    rolloff — excess bandwidth factor (0–1)
    """
    n_taps = span * sps + 1
    t = np.arange(n_taps, dtype=np.float64) - n_taps // 2
    t_norm = t / sps

    # RRC impulse response
    eps = 1e-9
    h = np.zeros(n_taps)
    for i, tn in enumerate(t_norm):
        if abs(tn) < eps:
            h[i] = 1.0 + rolloff * (4.0 / np.pi - 1.0)
        elif abs(abs(tn) - 1.0 / (2.0 * rolloff)) < eps:
            h[i] = (rolloff / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * rolloff))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * rolloff))
            )
        else:
            num = np.sin(np.pi * tn * (1.0 - rolloff)) + 4.0 * rolloff * tn * np.cos(np.pi * tn * (1.0 + rolloff))
            den = np.pi * tn * (1.0 - (4.0 * rolloff * tn) ** 2)
            h[i] = num / den

    h *= np.hanning(n_taps)
    h /= np.sqrt(np.sum(h ** 2))
    return h.astype(np.float32)


# ---------------------------------------------------------------------------
# Modulator functions
# ---------------------------------------------------------------------------

def _modulate_bpsk(bits: np.ndarray, sps: int) -> np.ndarray:
    """BPSK: bit 0 → +1, bit 1 → -1, pulse-shaped."""
    syms = (1.0 - 2.0 * bits.astype(np.float64))  # {+1, -1}
    upsampled = np.zeros(len(syms) * sps, dtype=np.float64)
    upsampled[::sps] = syms
    h = _rrc_filter(sps)
    shaped = np.convolve(upsampled, h, mode='same')
    return (shaped + 0j).astype(np.complex64)


def _modulate_qpsk(bits: np.ndarray, sps: int) -> np.ndarray:
    """QPSK: groups of 2 bits → Gray-coded constellation point."""
    # Pad to even length
    if len(bits) % 2:
        bits = np.concatenate([bits, [0]])
    pairs = bits.reshape(-1, 2)

    # Gray coded: 00→+1+j, 01→-1+j, 11→-1-j, 10→+1-j  (normalized)
    gray_map = {
        (0, 0): (1.0 + 1.0j),
        (0, 1): (-1.0 + 1.0j),
        (1, 1): (-1.0 - 1.0j),
        (1, 0): (1.0 - 1.0j),
    }
    syms = np.array([gray_map[(int(b0), int(b1))] for b0, b1 in pairs]) / np.sqrt(2.0)
    I = np.zeros(len(syms) * sps)
    Q = np.zeros(len(syms) * sps)
    I[::sps] = syms.real
    Q[::sps] = syms.imag
    h = _rrc_filter(sps)
    I_shaped = np.convolve(I, h, mode='same')
    Q_shaped = np.convolve(Q, h, mode='same')
    return (I_shaped + 1j * Q_shaped).astype(np.complex64)


def _modulate_gmsk(bits: np.ndarray, sps: int, bt: float = 0.35) -> np.ndarray:
    """
    GMSK modulator: NRZ → Gaussian filter → FM integrate.
    bt — bandwidth-time product (0.35 is GSM standard).
    """
    nrz = (1.0 - 2.0 * bits.astype(np.float64))  # {+1, -1}
    # Upsample NRZ
    upsampled = np.repeat(nrz, sps)

    # Gaussian filter
    n_taps = 4 * sps + 1
    t = (np.arange(n_taps) - n_taps // 2) / float(sps)
    sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * bt)
    g = np.exp(-t ** 2 / (2.0 * sigma ** 2))
    g /= g.sum()

    smoothed = np.convolve(upsampled, g, mode='same')

    # FM: integrate phase, modulation index h=0.5 for GMSK
    h_mod = 0.5
    phase = np.cumsum(smoothed) * (np.pi * h_mod / sps)
    iq = np.exp(1j * phase).astype(np.complex64)
    return iq


# ---------------------------------------------------------------------------
# Noise addition
# ---------------------------------------------------------------------------

def _add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN to achieve the requested SNR (dB)."""
    sig_power = float(np.mean(np.abs(signal) ** 2))
    if sig_power < 1e-12:
        return signal
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    noise = noise * np.sqrt(noise_power / 2.0)
    return (signal + noise).astype(np.complex64)


def _add_cfo(signal: np.ndarray, carrier_offset_hz: float, sample_rate: float) -> np.ndarray:
    """Apply a carrier-frequency offset."""
    if abs(carrier_offset_hz) < 1.0:
        return signal
    n = np.arange(len(signal))
    return (signal * np.exp(1j * 2.0 * np.pi * carrier_offset_hz / sample_rate * n)).astype(np.complex64)


# ---------------------------------------------------------------------------
# Core generate() function
# ---------------------------------------------------------------------------

_MODULATORS = {
    "BPSK": (_modulate_bpsk, 1),   # bits per symbol
    "QPSK": (_modulate_qpsk, 2),
    "GMSK": (_modulate_gmsk, 1),
}


# ---------------------------------------------------------------------------
# Coding & Framing helpers for synthetic generation
# ---------------------------------------------------------------------------

def _encode_fec(bits: np.ndarray, fec_scheme: str) -> tuple[np.ndarray, dict]:
    """Encode payload bits with the specified FEC scheme."""
    fec = fec_scheme.lower().strip()
    if fec in ("none", "", "no", "false"):
        return bits, {"fec": "none"}

    if fec in ("viterbi_k7", "k7", "viterbi", "convolutional_k7"):
        from commpy import Trellis, ConvolutionalEncoder
        trellis = Trellis(7, [0o171, 0o133])
        enc = ConvolutionalEncoder(trellis)
        coded, _ = enc.encode(bits.astype(int))
        return coded.astype(np.uint8), {"fec": "viterbi_k7", "constraint": 7, "generators": [0o171, 0o133]}

    if fec in ("viterbi_k3", "k3", "convolutional_k3"):
        from commpy import Trellis, ConvolutionalEncoder
        trellis = Trellis(3, [0o7, 0o5])
        enc = ConvolutionalEncoder(trellis)
        coded, _ = enc.encode(bits.astype(int))
        return coded.astype(np.uint8), {"fec": "viterbi_k3", "constraint": 3, "generators": [0o7, 0o5]}

    if fec in ("reed_solomon_8", "rs_8", "rs", "reedsolo"):
        import reedsolo
        # Pack bits to bytes, encode, unpack back
        pad = (8 - len(bits) % 8) % 8
        b = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)]) if pad else bits
        raw_bytes = np.packbits(b, bitorder='big').tobytes()
        codec = reedsolo.RSCodec(8)
        coded_bytes = bytes(codec.encode(bytearray(raw_bytes)))
        coded_bits = np.unpackbits(np.frombuffer(coded_bytes, dtype=np.uint8), bitorder='big')
        return coded_bits.astype(np.uint8), {"fec": "rs_8", "nsym": 8}

    return bits, {"fec": "none"}


def _apply_interleaver(bits: np.ndarray, interleaver: str) -> tuple[np.ndarray, dict]:
    """Apply interleaving to coded bits."""
    il = interleaver.lower().strip()
    if il in ("none", "", "no", "false"):
        return bits, {"interleaver": "none"}

    if il in ("block_8x8", "block", "8x8"):
        rows, cols = 8, 8
        blen = rows * cols
        n = (len(bits) // blen) * blen
        if n == 0:
            return bits, {"interleaver": "none"}
        trimmed = bits[:n]
        out = np.empty(n, dtype=np.uint8)
        for s in range(0, n, blen):
            blk = trimmed[s:s+blen].reshape(rows, cols, order='C')
            out[s:s+blen] = blk.T.reshape(-1, order='C')
        return out, {"interleaver": "block_8x8", "rows": 8, "cols": 8}

    if il in ("block_4x8", "4x8", "block_4_8"):
        rows, cols = 8, 4
        blen = rows * cols
        n = (len(bits) // blen) * blen
        if n == 0:
            return bits, {"interleaver": "none"}
        trimmed = bits[:n]
        out = np.empty(n, dtype=np.uint8)
        for s in range(0, n, blen):
            blk = trimmed[s:s+blen].reshape(rows, cols, order='C')
            out[s:s+blen] = blk.T.reshape(-1, order='C')
        return out, {"interleaver": "block_4x8", "rows": 4, "cols": 8}

    if il in ("conv_depth4", "conv", "depth4", "convolutional"):
        depth = 4
        span = 4
        n = len(bits)
        out = np.zeros(n, dtype=np.uint8)
        for k in range(depth):
            delay = k * span * depth
            for idx in range(k, n, depth):
                src = idx - delay
                if src >= 0:
                    out[idx] = bits[src]
        return out, {"interleaver": "conv_depth4", "depth": 4, "span": 4}

    return bits, {"interleaver": "none"}


def _get_preamble_bits(preamble: str, modulation: str = "BPSK") -> np.ndarray:
    """Return preamble bits + carrier lock lead-in sequence."""
    p = preamble.lower().strip()
    if p in ("none", "", "no", "false"):
        return np.array([], dtype=np.uint8)

    # Balanced lead-in allows Costas loop & clock recovery to settle
    mod = modulation.upper().strip()
    if mod == "QPSK":
        # Alternates diagonal constellation points (+1+j) and (-1-j)
        lead_in = np.array([0, 0, 1, 1] * 24, dtype=np.uint8)
    else:
        # Alternating 01 lead-in (+1 and -1)
        lead_in = np.array([0, 1] * 48, dtype=np.uint8)

    if p in ("ccsds", "ccsds_asm", "asm", "ccsds_0x1acffc1d"):
        asm = np.array([
            0,0,0,1,1,0,1,0, 1,1,0,0,1,1,1,1,
            1,1,1,1,1,1,0,0, 0,0,0,1,1,1,0,1
        ], dtype=np.uint8)
        return np.concatenate([lead_in, asm])

    if p in ("barker13", "barker", "barker_13"):
        barker = np.array([1,1,1,1,1,0,0,1,1,0,1,0,1], dtype=np.uint8)
        return np.concatenate([lead_in, barker])

    return np.array([], dtype=np.uint8)


def generate(
    modulation: str = "BPSK",
    n_symbols: int = 1024,
    sample_rate: float = 1_000_000,
    samples_per_symbol: int = 8,
    snr_db: float = 20.0,
    carrier_offset_hz: float = 0.0,
    seed: int | None = None,
    fec_scheme: str = "none",
    interleaver: str = "none",
    preamble: str = "none",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate a synthetic IQ signal with optional FEC, interleaving, and preamble framing.

    Parameters
    ----------
    modulation        : 'BPSK', 'QPSK', or 'GMSK'
    n_symbols         : approximate target number of modulated symbols
    sample_rate       : Hz — used for CFO and metadata
    samples_per_symbol: samples per symbol (>= 2)
    snr_db            : signal-to-noise ratio in dB
    carrier_offset_hz : optional carrier frequency offset (Hz)
    seed              : RNG seed for reproducibility
    fec_scheme        : 'none', 'viterbi_k7', 'viterbi_k3', or 'rs_8'
    interleaver       : 'none', 'block_8x8', or 'conv_depth4'
    preamble          : 'none', 'ccsds', or 'barker13'

    Returns
    -------
    (complex_samples, tx_bits, meta)
        complex_samples : np.ndarray complex64, shape (N,)
        tx_bits         : np.ndarray uint8 — transmitted payload bits (ground truth)
        meta            : dict with signal parameters
    """
    mod = modulation.upper()
    if mod not in _MODULATORS:
        raise ValueError(f"Unknown modulation '{modulation}'. Choose from: {list(_MODULATORS)}")

    rng = np.random.default_rng(seed)
    modulate_fn, bps = _MODULATORS[mod]

    # Compute payload size based on rate-1/2 FEC or RS expansion
    has_fec = fec_scheme.lower().strip() not in ("none", "", "no", "false")
    raw_payload_len = n_symbols * bps // 2 if has_fec else n_symbols * bps
    raw_payload_len = max(64, raw_payload_len)

    # For block interleaver, align to 64-bit blocks
    if interleaver.lower().strip() in ("block_8x8", "block", "8x8"):
        raw_payload_len = (raw_payload_len // 64) * 64
        if raw_payload_len == 0:
            raw_payload_len = 64

    tx_bits = rng.integers(0, 2, size=raw_payload_len, dtype=np.uint8)

    # 1. FEC encode
    coded_bits, fec_meta = _encode_fec(tx_bits, fec_scheme)

    # 2. Interleave
    interleaved_bits, il_meta = _apply_interleaver(coded_bits, interleaver)

    # 3. Framing / Preamble
    preamble_bits = _get_preamble_bits(preamble, modulation=mod)
    tx_stream = np.concatenate([preamble_bits, interleaved_bits]) if len(preamble_bits) > 0 else interleaved_bits

    # 4. Modulate
    iq = modulate_fn(tx_stream, samples_per_symbol)
    iq = _add_awgn(iq, snr_db, rng)
    iq = _add_cfo(iq, carrier_offset_hz, sample_rate)

    meta = {
        "modulation": mod,
        "n_symbols": len(tx_stream) // bps,
        "n_bits": len(tx_stream),
        "n_payload_bits": len(tx_bits),
        "sample_rate": sample_rate,
        "samples_per_symbol": samples_per_symbol,
        "snr_db": snr_db,
        "carrier_offset_hz": carrier_offset_hz,
        "total_samples": len(iq),
        "seed": seed,
        "fec_scheme": fec_scheme,
        "interleaver": interleaver,
        "preamble": preamble,
        "preamble_bits_len": len(preamble_bits),
    }

    return iq.astype(np.complex64), tx_bits, meta


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_wav(
    complex_samples: np.ndarray,
    sample_rate: float,
    path: str,
) -> str:
    """
    Save complex IQ as a stereo 16-bit PCM WAV file (left=I, right=Q).
    Normalises to ±32767 based on the peak amplitude.
    """
    I = complex_samples.real.astype(np.float64)
    Q = complex_samples.imag.astype(np.float64)

    peak = max(float(np.max(np.abs(I))), float(np.max(np.abs(Q))), 1e-9)
    I_int = np.clip((I / peak) * 32767, -32768, 32767).astype(np.int16)
    Q_int = np.clip((Q / peak) * 32767, -32768, 32767).astype(np.int16)

    # Interleave L, R
    stereo = np.empty(len(I_int) * 2, dtype=np.int16)
    stereo[0::2] = I_int
    stereo[1::2] = Q_int

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(int(sample_rate))
        wf.writeframes(stereo.tobytes())

    return path


def save_iq_raw(
    complex_samples: np.ndarray,
    path: str,
) -> str:
    """
    Save complex IQ as a raw float32 interleaved binary file (I,Q,I,Q,...).
    Compatible with file_loader.load_iq_raw(dtype='float32').
    """
    I = complex_samples.real.astype(np.float32)
    Q = complex_samples.imag.astype(np.float32)
    interleaved = np.empty(len(I) * 2, dtype=np.float32)
    interleaved[0::2] = I
    interleaved[1::2] = Q
    interleaved.tofile(path)
    return path


def generate_and_save(
    modulation: str = "BPSK",
    n_symbols: int = 1024,
    sample_rate: float = 1_000_000,
    samples_per_symbol: int = 8,
    snr_db: float = 20.0,
    carrier_offset_hz: float = 0.0,
    seed: int | None = None,
    output_dir: str = ".",
    fmt: str = "wav",   # 'wav' or 'iq'
    fec_scheme: str = "none",
    interleaver: str = "none",
    preamble: str = "none",
) -> tuple[str, np.ndarray, dict]:
    """
    Generate and save a synthetic IQ file.

    Returns (file_path, tx_bits, meta).
    """
    iq, tx_bits, meta = generate(
        modulation=modulation,
        n_symbols=n_symbols,
        sample_rate=sample_rate,
        samples_per_symbol=samples_per_symbol,
        snr_db=snr_db,
        carrier_offset_hz=carrier_offset_hz,
        seed=seed,
        fec_scheme=fec_scheme,
        interleaver=interleaver,
        preamble=preamble,
    )

    fmt = fmt.lower().lstrip(".")
    tag_parts = [modulation.upper()]
    if fec_scheme not in ("none", "", "no"):
        tag_parts.append(fec_scheme)
    if interleaver not in ("none", "", "no"):
        tag_parts.append(interleaver)
    tag_parts.extend([f"{int(sample_rate/1e3)}kHz", f"SNR{snr_db:.0f}dB"])

    fname = f"synthetic_{'_'.join(tag_parts)}.{fmt}"
    path = os.path.join(output_dir, fname)

    if fmt == "wav":
        save_wav(iq, sample_rate, path)
    elif fmt == "iq":
        save_iq_raw(iq, path)
    else:
        raise ValueError(f"Unknown format '{fmt}'. Use 'wav' or 'iq'.")

    return path, tx_bits, meta


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, tempfile

    print("synthetic_iq.py -- self-test")

    with tempfile.TemporaryDirectory() as tmpdir:
        for mod in ("BPSK", "QPSK", "GMSK"):
            iq, bits, meta = generate(
                modulation=mod, n_symbols=512, sample_rate=1e6,
                samples_per_symbol=8, snr_db=20.0, seed=42,
            )
            print(f"\n  {mod}: {len(iq)} samples, {len(bits)} bits, "
                  f"mean_amp={float(np.mean(np.abs(iq))):.4f}")
            assert len(iq) > 0, "Empty IQ"
            assert len(bits) > 0, "Empty bits"

            # Save/load round-trip for WAV
            wav_path, _, _ = generate_and_save(
                modulation=mod, n_symbols=256, output_dir=tmpdir, fmt="wav", seed=0,
            )
            assert os.path.exists(wav_path), "WAV file not created"
            print(f"  Saved WAV: {os.path.basename(wav_path)}  ({os.path.getsize(wav_path)} bytes)")

            # Save/load round-trip for .iq
            iq_path, _, _ = generate_and_save(
                modulation=mod, n_symbols=256, output_dir=tmpdir, fmt="iq", seed=0,
            )
            assert os.path.exists(iq_path), ".iq file not created"
            print(f"  Saved .iq: {os.path.basename(iq_path)}  ({os.path.getsize(iq_path)} bytes)")

    print("\nAll synthetic_iq self-tests PASSED.")
    sys.exit(0)
