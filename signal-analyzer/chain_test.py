"""
Quick end-to-end sanity check: file_loader.py -> inference.py, on a real
loaded .wav/.iq file rather than pre-saved Kaggle test samples.

Run from the project root (so both `preprocessing` and `models` are
importable), or adjust the sys.path lines below to match your layout.

Usage:
    python chain_test.py path/to/capture.wav
    python chain_test.py path/to/capture.iq --dtype float32 --rate 2000000
"""

import argparse
import sys
import os

# Adjust these two if your project layout differs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "preprocessing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from file_loader import load_file          # noqa: E402
from inference import predict_modulation   # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="file_loader -> inference chain test")
    parser.add_argument("path", help="Path to a .wav or .iq capture file")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["uint8", "int8", "int16", "float32"],
        help="Sample dtype, only used for raw .iq files (default: float32)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=None,
        help="Sample rate in Hz, only used for raw .iq files (metadata only)",
    )
    args = parser.parse_args()

    windows, meta = load_file(
        args.path, raw_iq_dtype=args.dtype, raw_iq_sample_rate=args.rate
    )
    print(f"Loaded '{args.path}'")
    print(f"  sample_rate:   {meta['sample_rate']}")
    print(f"  total_samples: {meta['total_samples']}")
    print(f"  num_windows:   {meta['num_windows']}")
    print()

    if meta["num_windows"] == 0:
        print("No full windows available — file shorter than 1024 samples.")
        return

    for idx, window in enumerate(windows):
        result = predict_modulation(window)
        print(
            f"window {idx:>3}: {result['class']:<6} "
            f"(confidence={result['confidence']:.4f})  "
            f"all_probs={ {k: round(v, 3) for k, v in result['all_probs'].items()} }"
        )


if __name__ == "__main__":
    main()