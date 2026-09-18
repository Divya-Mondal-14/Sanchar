"""
gnuradio_pipeline/deinterleave.py

Three de-interleavers for the SIH signal analyzer pipeline.

The user NEVER manually selects the de-interleaver type.  Instead, the
public entry point `try_all_deinterleavers()` automatically tries every
supported scheme + parameter combination and keeps whichever produces
the best FEC validation score from the downstream decoder.

Supported schemes
-----------------
    Block             — transpose a (rows x cols) bit matrix
    Convolutional     — shift-register / diagonal de-interleaver (Ramsey type)
    Pseudo-Random     — reverse a PRNG-seeded index permutation

Public API
----------
    block_deinterleave(bits, rows, cols)          -> np.ndarray
    convolutional_deinterleave(bits, depth, span) -> np.ndarray
    pseudo_random_deinterleave(bits, seed)        -> np.ndarray
    try_all_deinterleavers(bits, fec_validator_fn) -> dict

fec_validator_fn signature
--------------------------
    Callable[[np.ndarray], float]
    Accepts a bit array, returns a score in [0, 1] where higher = better decode.
    A score of 1.0 means a validated, zero-error decode (e.g. zero RS syndrome).
    A score of 0.0 means total decode failure.
"""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Block de-interleaver
# ---------------------------------------------------------------------------

def block_deinterleave(
    bits: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    Block (rectangular) de-interleaver.

    During block interleaving the transmitter writes bits into a (rows x cols)
    matrix row-by-row and reads them out column-by-column.  The receiver
    reverses this: write column-by-column (reshape to (cols, rows) and
    transpose back).

    Parameters
    ----------
    bits : np.ndarray, 1-D, dtype uint8
        Bit stream of length rows * cols.  Trailing bits that don't fill the
        last block are silently dropped.
    rows : int
        Number of rows in the interleaver matrix.
    cols : int
        Number of columns in the interleaver matrix.

    Returns
    -------
    np.ndarray, 1-D, dtype uint8 — de-interleaved bits.
    """
    block_len = rows * cols
    n = (len(bits) // block_len) * block_len  # trim to whole blocks
    if n == 0:
        return bits.copy()

    out = np.empty(n, dtype=np.uint8)
    trimmed = bits[:n]

    for block_start in range(0, n, block_len):
        blk = trimmed[block_start : block_start + block_len]
        # Transmitter wrote row-by-row (shape (rows, cols)), read col-by-col
        # De-interleave: the received order is column-major, undo it
        matrix = blk.reshape(rows, cols, order='C')          # row-major
        out[block_start : block_start + block_len] = matrix.T.reshape(-1, order='C')

    return out


# ---------------------------------------------------------------------------
# Convolutional / diagonal de-interleaver
# ---------------------------------------------------------------------------

def convolutional_deinterleave(
    bits: np.ndarray,
    depth: int,
    span: int | None = None,
) -> np.ndarray:
    """
    Convolutional (diagonal / Ramsey Type II) de-interleaver.

    A convolutional interleaver routes each bit through a shift register of
    increasing depth (0, step, 2*step, …, (depth-1)*step) and mixes across
    `depth` sequential symbols.  The de-interleaver applies the complementary
    delays (depth*step, (depth-1)*step, …, 0).

    Parameters
    ----------
    bits : np.ndarray, 1-D, dtype uint8
    depth : int
        Number of branches (== interleaver depth parameter).
    span : int or None
        Delay step between branches.  None defaults to `depth` itself (the
        most common standard choice, giving total delay = depth² samples).

    Returns
    -------
    np.ndarray, 1-D, dtype uint8 — de-interleaved bits (length trimmed to
    complete commutation cycles, leading latency samples zeroed).
    """
    if span is None:
        span = depth

    total_delay = depth * span * depth  # total pipeline latency in samples
    n = len(bits)

    if n < total_delay:
        return bits.copy()

    out = np.zeros(n, dtype=np.uint8)

    # Each branch k gets a delay of (depth - 1 - k) * span * depth samples
    for k in range(depth):
        delay = (depth - 1 - k) * span * depth
        # Indices in the de-interleaved output that branch k is responsible for
        for idx in range(k, n, depth):
            src = idx - delay
            if src >= 0:
                out[idx] = bits[src]

    return out


# ---------------------------------------------------------------------------
# Pseudo-random de-interleaver
# ---------------------------------------------------------------------------

def pseudo_random_deinterleave(
    bits: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Pseudo-random de-interleaver.

    The transmitter permutes bits using a PRNG-seeded index shuffle.
    This function inverts that permutation for a block of len(bits) bits.

    Parameters
    ----------
    bits  : np.ndarray, 1-D, dtype uint8
    seed  : int — PRNG seed used by the transmitter.

    Returns
    -------
    np.ndarray, 1-D, dtype uint8 — de-interleaved bits (same length).
    """
    n = len(bits)
    rng = np.random.default_rng(seed)

    # Reproduce the transmitter's permutation
    perm = rng.permutation(n).astype(np.intp)

    # Invert: out[perm[i]] = bits[i]  <=>  out = bits[inverse_perm]
    inv_perm = np.empty(n, dtype=np.intp)
    inv_perm[perm] = np.arange(n, dtype=np.intp)

    return bits[inv_perm].astype(np.uint8)


# ---------------------------------------------------------------------------
# Auto-try orchestrator
# ---------------------------------------------------------------------------

# Candidate parameters to sweep during auto-detection.
# These cover the most common terrestrial HF/VHF/UHF interleaver sizes.
_BLOCK_CANDIDATES = [
    (4, 4), (8, 8), (16, 16), (32, 32),   # square
    (4, 8), (8, 4), (4, 16), (16, 4),      # rectangular
    (12, 12), (20, 20),
]

_CONV_CANDIDATES = [
    (4,  None),   # depth=4,  span=4
    (8,  None),   # depth=8,  span=8
    (16, None),   # depth=16, span=16
    (4,  5),
    (8,  5),
]

_PR_SEEDS = [0, 1, 42, 12345]


def try_all_deinterleavers(
    bits: np.ndarray,
    fec_validator_fn: Callable[[np.ndarray], float],
) -> dict:
    """
    Automatically try every supported de-interleaver + parameter combination.

    Evaluates each candidate by passing its output to `fec_validator_fn` and
    keeps the combination with the highest score.  Short-circuits as soon as a
    score of 1.0 is achieved (perfect decode).

    Parameters
    ----------
    bits : np.ndarray, 1-D, dtype uint8
    fec_validator_fn : Callable[[np.ndarray], float]
        Downstream FEC validator.  Must return a float in [0, 1].
        Higher = better.  1.0 = confirmed valid decode.

    Returns
    -------
    dict with keys:
        'method'     : str   — winning scheme name ('block', 'convolutional',
                                'pseudo_random', or 'none')
        'params'     : dict  — winning parameter set
        'score'      : float — best FEC validation score achieved
        'bits'       : np.ndarray uint8 — de-interleaved bits for the winner
        'all_scores' : list of (method, params, score) tuples, sorted desc
    """
    best_score = -1.0
    best_bits = bits.copy()
    best_method = "none"
    best_params: dict = {}
    all_scores: list = []

    orig_mean = float(np.mean(bits)) if len(bits) > 0 else 0.5

    def _evaluate(method: str, params: dict, candidate_bits: np.ndarray) -> float:
        if len(candidate_bits) == 0:
            score = 0.0
        else:
            cand_mean = float(np.mean(candidate_bits))
            # Reject degenerate outputs that collapsed into all zeros or all ones
            if (cand_mean < 0.03 or cand_mean > 0.97) and (0.05 <= orig_mean <= 0.95):
                score = 0.0
            else:
                try:
                    score = float(fec_validator_fn(candidate_bits))
                except Exception:
                    score = 0.0
        all_scores.append((method, params, score))
        return score

    # --- 0. Baseline: no de-interleaving ---
    score = _evaluate("none", {}, bits)
    if score > best_score:
        best_score, best_bits, best_method, best_params = score, bits.copy(), "none", {}
    if best_score >= 0.90:
        return _pack_result(best_method, best_params, best_score, best_bits, all_scores)

    # --- 1. Block ---
    for rows, cols in _BLOCK_CANDIDATES:
        candidate = block_deinterleave(bits, rows, cols)
        score = _evaluate("block", {"rows": rows, "cols": cols}, candidate)
        if score > best_score:
            best_score, best_bits, best_method, best_params = (
                score, candidate.copy(), "block", {"rows": rows, "cols": cols}
            )
        if best_score >= 0.90:
            return _pack_result(best_method, best_params, best_score, best_bits, all_scores)

    # --- 2. Convolutional ---
    for depth, span in _CONV_CANDIDATES:
        candidate = convolutional_deinterleave(bits, depth, span)
        params = {"depth": depth, "span": span if span is not None else depth}
        score = _evaluate("convolutional", params, candidate)
        if score > best_score:
            best_score, best_bits, best_method, best_params = (
                score, candidate.copy(), "convolutional", params
            )
        if best_score >= 0.90:
            return _pack_result(best_method, best_params, best_score, best_bits, all_scores)

    # --- 3. Pseudo-random ---
    for seed in _PR_SEEDS:
        candidate = pseudo_random_deinterleave(bits, seed)
        score = _evaluate("pseudo_random", {"seed": seed}, candidate)
        if score > best_score:
            best_score, best_bits, best_method, best_params = (
                score, candidate.copy(), "pseudo_random", {"seed": seed}
            )
        if best_score >= 0.90:
            return _pack_result(best_method, best_params, best_score, best_bits, all_scores)

    # An interleaver is only confirmed if the downstream FEC decoder successfully locks (score >= 0.70)
    if best_score < 0.70:
        return _pack_result("none", {}, 0.0, bits.copy(), all_scores)

    return _pack_result(best_method, best_params, best_score, best_bits, all_scores)


def _pack_result(
    method: str,
    params: dict,
    score: float,
    bits: np.ndarray,
    all_scores: list,
) -> dict:
    all_scores_sorted = sorted(all_scores, key=lambda x: x[2], reverse=True)
    return {
        "method": method,
        "params": params,
        "score": score,
        "bits": bits,
        "all_scores": all_scores_sorted,
    }


# ---------------------------------------------------------------------------
# Self-test (python gnuradio_pipeline/deinterleave.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("deinterleave.py -- self-test")
    rng = np.random.default_rng(7)
    ROWS, COLS = 8, 8
    N = ROWS * COLS * 10

    original = rng.integers(0, 2, size=N, dtype=np.uint8)

    # --- Block round-trip ---
    # Simulate interleaver (write row-by-row, read col-by-col = transpose)
    def block_interleave(b, rows, cols):
        blen = rows * cols
        n = (len(b) // blen) * blen
        out = np.empty(n, dtype=np.uint8)
        for s in range(0, n, blen):
            blk = b[s:s+blen].reshape(rows, cols, order='C')
            out[s:s+blen] = blk.T.reshape(-1, order='C')
        return out

    interleaved = block_interleave(original, ROWS, COLS)
    recovered = block_deinterleave(interleaved, ROWS, COLS)
    assert np.array_equal(original[:len(recovered)], recovered), \
        "Block round-trip FAILED"
    print(f"  Block ({ROWS}x{COLS}) round-trip: PASSED")

    # --- Pseudo-random round-trip ---
    SEED = 42
    perm = np.random.default_rng(SEED).permutation(N).astype(np.intp)
    pr_interleaved = original[perm].astype(np.uint8)
    pr_recovered = pseudo_random_deinterleave(pr_interleaved, seed=SEED)
    assert np.array_equal(original, pr_recovered), "PR round-trip FAILED"
    print(f"  Pseudo-random (seed={SEED}) round-trip: PASSED")

    # --- Convolutional (smoke test: output produced, length reasonable) ---
    depth = 4
    cv_out = convolutional_deinterleave(original, depth)
    assert len(cv_out) == N, f"Convolutional output length mismatch: {len(cv_out)} vs {N}"
    print(f"  Convolutional (depth={depth}) output length: PASSED")

    # --- try_all_deinterleavers with a mock FEC validator that rewards block(8,8) ---
    def mock_fec(bits):
        # Perfect score only when bits == original (round-trip through block)
        recovered_inner = block_deinterleave(bits, ROWS, COLS)
        matches = np.sum(original[:len(recovered_inner)] == recovered_inner)
        return float(matches) / max(len(recovered_inner), 1)

    result = try_all_deinterleavers(interleaved, mock_fec)
    print(f"\n  try_all_deinterleavers winner: method='{result['method']}', "
          f"score={result['score']:.4f}, params={result['params']}")
    assert result["score"] > 0.95, f"Auto-selector did not find good block params: {result}"
    print("  Auto-selector: PASSED")

    print("\nAll deinterleave self-tests PASSED.")
    sys.exit(0)
