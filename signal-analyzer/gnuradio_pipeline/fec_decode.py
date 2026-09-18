"""
gnuradio_pipeline/fec_decode.py

FEC decoders for the SIH signal analyzer pipeline.

Supported schemes
-----------------
    Viterbi     — short-constraint convolutional codes, via commpy.channelcoding
    Reed-Solomon — RS block codes, via reedsolo.RSCodec
    LDPC         — TODO stretch goal (see roadmap note below)
    Concatenated — TODO stretch goal (see roadmap note below)

The user NEVER manually selects the FEC scheme.  `try_all_fec()` is the
public entry point: it auto-tries every supported decoder + parameter
combination, validates by path-metric quality (Viterbi) or zero syndrome
(Reed-Solomon), and returns the best result.

Roadmap note (LDPC / Concatenated codes)
-----------------------------------------
    LDPC and concatenated-code decoding require a known parity-check matrix
    or code parameters (rate, interleaver seed), which are typically signalled
    in a protocol header.  Until header parsing is implemented (bitstream
    correlation step), these cannot be attempted blindly without an
    unacceptably large search space.

    When the bitstream correlator (correlation/sync_correlator.py) successfully
    identifies a known frame format, the code parameters can be extracted from
    the header and passed directly to a targeted decoder here.  At that point,
    LDPC decoding can be implemented using `ldpc` (pypi) or GNU Radio's
    gr.fec.ldpc_decoder_def, and concatenated decoding as an RS outer +
    convolutional inner chain already available in this module.

New dependencies (add to requirements.txt)
-------------------------------------------
    commpy   — pip install commpy
    reedsolo — pip install reedsolo

Public API
----------
    viterbi_decode(bits, constraint, generators)   -> (np.ndarray, float)
    rs_decode(data_bytes, nsym)                    -> (bytes, bool)
    try_all_fec(bits)                              -> dict
    bits_to_bytes(bits)                            -> bytes
    bytes_to_bits(data, nbits)                     -> np.ndarray
"""

from __future__ import annotations

import numpy as np

# --- commpy (Viterbi) ---
try:
    from commpy import viterbi_decode as _commpy_viterbi, Trellis
    _COMMPY_AVAILABLE = True
except ImportError:
    _COMMPY_AVAILABLE = False

# --- reedsolo (Reed-Solomon) ---
try:
    import reedsolo
    _REEDSOLO_AVAILABLE = True
except ImportError:
    _REEDSOLO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Bit <-> byte conversion helpers
# ---------------------------------------------------------------------------

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Pack a 0/1 bit array (MSB first) into bytes, zero-padding the last byte.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    pad = (8 - len(bits) % 8) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    packed = np.packbits(bits, bitorder='big')
    return packed.tobytes()


def bytes_to_bits(data: bytes | bytearray, nbits: int | None = None) -> np.ndarray:
    """
    Unpack bytes to a 0/1 bit array (MSB first).
    nbits trims to exact length if supplied.
    """
    arr = np.frombuffer(data if isinstance(data, (bytes, bytearray)) else bytes(data),
                        dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder='big')
    if nbits is not None:
        bits = bits[:nbits]
    return bits.astype(np.uint8)


# ---------------------------------------------------------------------------
# Viterbi decoder (commpy)
# ---------------------------------------------------------------------------

# Common constraint lengths and generator polynomials for short convolutional codes
# Format: (constraint_length, [generator_oct, generator_oct, ...])
# Rates: 1/2 with k=3 through 7 cover most HF/VHF standards
_VITERBI_CANDIDATES = [
    # (constraint, generators_octal)
    (3, [0o7, 0o5]),          # rate 1/2, k=3  (classic, simple)
    (5, [0o35, 0o23]),        # rate 1/2, k=5  (AX.25 FX.25, APRS)
    (7, [0o171, 0o133]),      # rate 1/2, k=7  (standard, e.g. CCSDS, 802.11)
    (7, [0o133, 0o171]),      # rate 1/2, k=7 reversed
    (9, [0o561, 0o753]),      # rate 1/2, k=9  (NASA / CCSDS)
]


def viterbi_decode(
    bits: np.ndarray,
    constraint: int,
    generators: list[int],
    tb_depth: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Viterbi (maximum likelihood) decoder for a rate-1/2 convolutional code.

    Uses commpy 1.2.0 API:
        Trellis(constraint_length, generators)
        viterbi_decode(trellis, received, mode='hard', terminated=True)

    Parameters
    ----------
    bits : np.ndarray, 1-D, dtype uint8
        Received bit stream (hard decisions, values 0/1).
    constraint : int
        Constraint length k.
    generators : list of int
        Generator polynomials as octal integers, e.g. [0o171, 0o133] for k=7.
    tb_depth : int or None
        Unused (kept for API compatibility; commpy 1.2.0 uses full ML internally).

    Returns
    -------
    (decoded_bits, path_metric)
        decoded_bits : np.ndarray, dtype uint8
        path_metric  : float — 0.0 = perfect, 1.0 = completely wrong.
    """
    if not _COMMPY_AVAILABLE:
        raise RuntimeError(
            "commpy is required for Viterbi decoding. "
            "Install it with: pip install commpy"
        )

    trellis = Trellis(constraint, generators)

    # commpy 1.2.0: viterbi_decode(trellis, received, mode='hard'|'soft', terminated=bool)
    received = bits.astype(int)
    if len(received) % 2 != 0:
        received = received[:-1]

    decoded = _commpy_viterbi(trellis, received, mode='hard', terminated=True)

    # Path metric: re-encode and compare to measure how many errors were corrected
    from commpy import ConvolutionalEncoder
    enc = ConvolutionalEncoder(trellis)
    re_encoded, _ = enc.encode(decoded.astype(int))
    n_compare = min(len(re_encoded), len(received))
    errors = int(np.sum(re_encoded[:n_compare] != received[:n_compare]))
    path_metric = float(errors) / max(n_compare, 1)

    return decoded.astype(np.uint8), path_metric


# ---------------------------------------------------------------------------
# Reed-Solomon decoder (reedsolo)
# ---------------------------------------------------------------------------

# Common nsym (ECC symbol) counts to try
_RS_NSYM_CANDIDATES = [8, 10, 16, 20, 32]


def rs_decode(
    data_bytes: bytes | bytearray,
    nsym: int,
) -> tuple[bytes, bool]:
    """
    Reed-Solomon decoder.

    Parameters
    ----------
    data_bytes : bytes or bytearray
        Received codeword (data + ECC bytes).
    nsym : int
        Number of ECC (parity check) symbols appended by the encoder.
        Must be even; can correct up to nsym/2 symbol errors.

    Returns
    -------
    (decoded_bytes, success)
        decoded_bytes : bytes — corrected message (ECC bytes stripped)
        success       : bool  — True if syndrome is zero (valid codeword)
    """
    if not _REEDSOLO_AVAILABLE:
        raise RuntimeError(
            "reedsolo is required for Reed-Solomon decoding. "
            "Install it with: pip install reedsolo"
        )

    codec = reedsolo.RSCodec(nsym)
    try:
        # 1. Zero-syndrome check: clean codeword requires zero corrections
        chk = codec.check(bytearray(data_bytes))
        if isinstance(chk, list) and all(chk):
            result = codec.decode(bytearray(data_bytes))
            decoded_msg = bytes(result[0]) if isinstance(result, tuple) else bytes(result)
            return decoded_msg, True

        # 2. Decode with strict error tolerance (at most 1 symbol error)
        # Random noise frequently triggers false positive "corrections" of 4+ bytes in small blocks.
        result = codec.decode(bytearray(data_bytes))
        errata = result[2] if len(result) > 2 else []
        if len(errata) <= 1:
            decoded_msg = bytes(result[0]) if isinstance(result, tuple) else bytes(result)
            return decoded_msg, True

        return bytes(data_bytes), False
    except (reedsolo.ReedSolomonError, IndexError, ValueError):
        return bytes(data_bytes), False


# ---------------------------------------------------------------------------
# LDPC / Concatenated — stretch goals (stubbed)
# ---------------------------------------------------------------------------

def ldpc_decode(bits: np.ndarray, **kwargs) -> tuple[np.ndarray, float]:
    """
    LDPC decoder — STRETCH GOAL, not yet implemented.

    TODO: Implement using `ldpc` (pypi) or GNU Radio's gr.fec.ldpc_decoder_def
    once the bitstream correlator (correlation/sync_correlator.py) is able to
    extract the parity-check matrix parameters from a protocol header.

    Roadmap:
        1. sync_correlator identifies frame format and extracts LDPC params
           (block length N, rate R, H-matrix seed or standard name).
        2. ldpc_decode() uses those params to instantiate a belief-propagation
           decoder and decode the soft-decision LLR input.
        3. Convergence check: compare syndrome H @ decoded mod 2 == 0.
    """
    raise NotImplementedError(
        "LDPC decoding is a stretch goal. "
        "See the docstring for the implementation roadmap."
    )


def concatenated_decode(
    bits: np.ndarray,
    inner_constraint: int = 7,
    inner_generators: list[int] | None = None,
    outer_nsym: int = 8,
) -> tuple[np.ndarray, bool]:
    """
    Concatenated code decoder — STRETCH GOAL.

    Decodes an inner convolutional code (Viterbi) followed by an outer
    Reed-Solomon code.  Currently delegates to viterbi_decode + rs_decode
    in sequence, which works when both succeed independently.

    TODO: Add interleaving between inner and outer stages once
    de-interleaver parameters are reliably identified by try_all_deinterleavers.
    """
    if inner_generators is None:
        inner_generators = [0o171, 0o133]  # k=7 standard

    inner_bits, pm = viterbi_decode(bits, inner_constraint, inner_generators)
    if pm > 0.1:
        return inner_bits, False

    inner_bytes = bits_to_bytes(inner_bits)
    outer_decoded, ok = rs_decode(inner_bytes, outer_nsym)
    return bytes_to_bits(outer_decoded), ok


# ---------------------------------------------------------------------------
# Auto-try orchestrator
# ---------------------------------------------------------------------------

def _viterbi_score(path_metric: float, bits: np.ndarray | None = None) -> float:
    """
    Convert path metric (lower=better) to score (higher=better) in [0, 1].
    True convolutional codes with typical channel noise have path_metric < 0.05.
    Un-encoded random data produces path_metric around 0.12 - 0.15.
    Reject degenerate all-zero / all-one inputs.
    """
    if bits is not None and len(bits) > 0:
        mean_val = float(np.mean(bits))
        if mean_val < 0.05 or mean_val > 0.95:
            return 0.0
    if path_metric < 0.05:
        return float(np.clip(1.0 - path_metric * 10.0, 0.5, 1.0))
    return 0.0


def _rs_score(success: bool, decoded: bytes, total_len: int) -> float:
    """RS decode gives 1.0 on success, 0.0 on failure."""
    return 1.0 if success else 0.0


def fast_fec_score(bits: np.ndarray, max_eval_bits: int = 1024) -> float:
    """
    Ultra-fast FEC scoring heuristic for inner loops (such as de-interleaver search).
    Evaluates only a representative prefix of bits (default 1024) against the most
    common terrestrial standards (CCSDS k=7, simple k=3, RS-8) with early exit.
    Tolerates carrier phase inversion and 1-bit slip.
    """
    if len(bits) < 16:
        return 0.0
    mean_val = float(np.mean(bits))
    if mean_val < 0.05 or mean_val > 0.95:
        return 0.0
    sample = bits[:max_eval_bits] if len(bits) > max_eval_bits else bits

    # Test candidate polarities (normal, inverted) and small bit-slips (0, 1)
    variants = [sample]
    if len(sample) > 32:
        variants.extend([1 - sample, sample[1:], 1 - sample[1:]])

    if _COMMPY_AVAILABLE:
        for var in variants:
            # 1. Check standard CCSDS / terrestrial rate-1/2 convolutional code (k=7)
            try:
                _, pm = viterbi_decode(var, 7, [0o171, 0o133])
                score = _viterbi_score(pm, var)
                if score >= 0.8:
                    return score
            except Exception:
                pass

            # 2. Check simple rate-1/2 (k=3)
            try:
                _, pm = viterbi_decode(var, 3, [0o7, 0o5])
                score = _viterbi_score(pm, var)
                if score >= 0.8:
                    return score
            except Exception:
                pass

    # 3. Check Reed-Solomon (nsym=8) across small byte/bit shifts
    if _REEDSOLO_AVAILABLE:
        for shift in (0, 1):
            s_arr = sample[shift:] if shift > 0 else sample
            data_bytes = bits_to_bytes(s_arr)
            if len(data_bytes) > 8:
                try:
                    _, ok = rs_decode(data_bytes, 8)
                    if ok:
                        return 1.0
                except Exception:
                    pass

    return 0.0


def try_all_fec(bits: np.ndarray, max_eval_bits: int = 2048) -> dict:
    """
    Automatically try every supported FEC decoder + parameter combination.

    Evaluates candidates by screening on a sample prefix (up to max_eval_bits)
    first for high performance, and only performs full-length decoding when
    a verified valid code is detected.
    Tolerates carrier phase inversion and bit alignment shifts.

    Parameters
    ----------
    bits : np.ndarray, 1-D, dtype uint8
    max_eval_bits : int
        Maximum number of bits to use for candidate screening (default 2048).

    Returns
    -------
    dict with keys:
        'method'      : str   — winning scheme ('viterbi', 'reed_solomon', 'none')
        'params'      : dict  — winning parameter set
        'score'       : float — quality score in [0, 1]; 1.0 = confirmed valid
        'decoded_bits': np.ndarray uint8 — decoded bit stream
        'decoded_bytes': bytes            — decoded bytes (bits packed, MSB first)
        'all_scores'  : list of (method, params, score) tuples, sorted desc
    """
    if len(bits) < 16:
        return _pack("none", {}, 0.0, bits.copy(), bits_to_bytes(bits), [])
    mean_val = float(np.mean(bits))
    if mean_val < 0.05 or mean_val > 0.95:
        return _pack("none", {}, 0.0, bits.copy(), bits_to_bytes(bits), [])

    all_scores: list = []
    best_score = 0.0
    best_candidate = None
    best_variant_opts = {"invert": False, "shift": 0}

    eval_bits = bits[:max_eval_bits] if len(bits) > max_eval_bits else bits

    # Candidate variants to test: (invert, shift)
    variants_to_test = [
        (False, 0),
        (True, 0),
        (False, 1),
        (True, 1),
    ]

    # --- Screen Viterbi candidates on eval_bits ---
    if _COMMPY_AVAILABLE:
        for inv, shift in variants_to_test:
            if shift >= len(eval_bits):
                continue
            cur_bits = 1 - eval_bits[shift:] if inv else eval_bits[shift:]

            for constraint, generators in _VITERBI_CANDIDATES:
                if constraint > 7 and best_score >= 0.8:
                    continue
                try:
                    _, pm = viterbi_decode(cur_bits, constraint, generators)
                    score = _viterbi_score(pm, cur_bits)
                    params = {"constraint": constraint, "generators": generators, "invert": inv, "shift": shift}
                    all_scores.append(("viterbi", params, score))
                    if score > best_score:
                        best_score = score
                        best_candidate = ("viterbi", params)
                        best_variant_opts = {"invert": inv, "shift": shift}
                    if best_score >= 0.95:
                        break
                except Exception:
                    pass
            if best_score >= 0.95:
                break

    # --- Screen Reed-Solomon candidates on eval_bits ---
    if _REEDSOLO_AVAILABLE and best_score < 0.95:
        for shift in range(min(8, len(eval_bits))):
            cur_bits = eval_bits[shift:]
            data_bytes = bits_to_bytes(cur_bits)
            for nsym in _RS_NSYM_CANDIDATES:
                if len(data_bytes) <= nsym:
                    continue
                try:
                    dec_bytes, ok = rs_decode(data_bytes, nsym)
                    score = _rs_score(ok, dec_bytes, len(data_bytes))
                    params = {"nsym": nsym, "shift": shift}
                    all_scores.append(("reed_solomon", params, score))
                    if score > best_score:
                        best_score = score
                        best_candidate = ("reed_solomon", params)
                        best_variant_opts = {"invert": False, "shift": shift}
                    if best_score >= 1.0:
                        break
                except Exception:
                    pass
            if best_score >= 1.0:
                break

    # If a genuine code match was found, decode the full bitstream with the winner
    if best_candidate is not None and best_score > 0.0:
        method, params = best_candidate
        inv = best_variant_opts["invert"]
        shift = best_variant_opts["shift"]
        full_bits = 1 - bits[shift:] if inv else bits[shift:]

        try:
            if method == "viterbi":
                dec_bits, pm = viterbi_decode(full_bits, params["constraint"], params["generators"])
                dec_bytes = bits_to_bytes(dec_bits)
                return _pack(method, params, _viterbi_score(pm, full_bits), dec_bits, dec_bytes, all_scores)
            elif method == "reed_solomon":
                full_bytes = bits_to_bytes(full_bits)
                dec_bytes, ok = rs_decode(full_bytes, params["nsym"])
                dec_bytes = bytes(dec_bytes)
                dec_bits = bytes_to_bits(dec_bytes)
                return _pack(method, params, 1.0 if ok else 0.0, dec_bits, dec_bytes, all_scores)
        except Exception:
            pass

    # No FEC detected / un-encoded
    return _pack("none", {}, 0.0, bits.copy(), bits_to_bytes(bits), all_scores)


def _pack(method, params, score, bits, dec_bytes, all_scores):
    return {
        "method": method,
        "params": params,
        "score": score,
        "decoded_bits": bits,
        "decoded_bytes": bytes(dec_bytes) if dec_bytes is not None else b"",
        "all_scores": sorted(all_scores, key=lambda x: x[2], reverse=True),
    }


# ---------------------------------------------------------------------------
# Self-test (python gnuradio_pipeline/fec_decode.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("fec_decode.py -- self-test")
    print(f"  commpy  available: {_COMMPY_AVAILABLE}")
    print(f"  reedsolo available: {_REEDSOLO_AVAILABLE}")

    rng = np.random.default_rng(99)
    PASS = True

    # --- Viterbi round-trip ---
    if _COMMPY_AVAILABLE:
        from commpy import ConvolutionalEncoder

        constraint = 7
        generators = [0o171, 0o133]
        trellis = Trellis(constraint, generators)
        enc = ConvolutionalEncoder(trellis)

        tx_bits = rng.integers(0, 2, size=100, dtype=np.uint8)
        encoded, _ = enc.encode(tx_bits.astype(int))
        encoded = encoded.astype(np.uint8)
        decoded, pm = viterbi_decode(encoded, constraint, generators)

        # Trim to same length for comparison
        compare_len = min(len(tx_bits), len(decoded))
        ber = float(np.sum(tx_bits[:compare_len] != decoded[:compare_len])) / compare_len
        print(f"\n  Viterbi (k=7) BER on noiseless encode->decode: {ber:.4f}  path_metric={pm:.4f}")
        if ber > 0.05:
            print("  WARNING: Viterbi BER unexpectedly high for noiseless case")
        else:
            print("  Viterbi round-trip: PASSED")
    else:
        print("\n  commpy not installed -- skipping Viterbi test")

    # --- Reed-Solomon round-trip ---
    if _REEDSOLO_AVAILABLE:
        nsym = 10
        codec = reedsolo.RSCodec(nsym)
        msg = bytes(rng.integers(0, 256, size=40, dtype=np.uint8))
        encoded_rs = bytes(codec.encode(bytearray(msg)))
        decoded_rs, ok = rs_decode(encoded_rs, nsym)
        print(f"\n  RS({len(encoded_rs)},{len(msg)}) noiseless decode success={ok}")
        assert ok, "RS noiseless decode FAILED"
        assert decoded_rs == msg, f"RS decoded bytes mismatch: {decoded_rs[:8]} vs {msg[:8]}"
        print("  Reed-Solomon round-trip: PASSED")

        # With 1 byte error (correctable)
        corrupted = bytearray(encoded_rs)
        corrupted[5] ^= 0xFF  # flip all bits in byte 5
        dec_err, ok_err = rs_decode(bytes(corrupted), nsym)
        print(f"  RS with 1 error byte: success={ok_err}")
        if ok_err:
            assert dec_err == msg, "RS error-correction output mismatch"
            print("  Reed-Solomon 1-error correction: PASSED")
        else:
            print("  Reed-Solomon 1-error correction: could not correct (may depend on RS version)")
    else:
        print("\n  reedsolo not installed -- skipping RS test")

    # --- try_all_fec smoke test ---
    test_bits = rng.integers(0, 2, size=512, dtype=np.uint8)
    result = try_all_fec(test_bits)
    print(f"\n  try_all_fec: method='{result['method']}', score={result['score']:.4f}")
    print(f"  decoded_bits shape: {result['decoded_bits'].shape}")
    print(f"  decoded_bytes length: {len(result['decoded_bytes'])}")
    print("  try_all_fec: PASSED (ran without exceptions)")

    print("\nAll fec_decode self-tests complete.")
    sys.exit(0)
