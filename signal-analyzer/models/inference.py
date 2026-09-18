"""
Inference module for the modulation classifier.

Usage:
    from inference import predict_modulation
    result = predict_modulation(iq_samples)   # iq_samples: numpy array, shape (1024, 2)
    print(result)  # {'class': 'QPSK', 'confidence': 0.98, 'all_probs': {...}}

Loads the model once at import time so repeated calls (e.g. from the GUI's
file-load handler) don't reload weights from disk every time.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from classifier import ModClassifier

# --- Paths (adjust if your folder layout differs) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_THIS_DIR, "modclassifier_best.pth")
LABEL_MAP_PATH = os.path.join(_THIS_DIR, "label_map.json")

# --- Load label map ---
with open(LABEL_MAP_PATH, "r") as f:
    _label_info = json.load(f)

CLASSES = _label_info["classes"]          # e.g. ["BPSK", "QPSK", "GMSK"]
INPUT_LEN = _label_info["input_len"]      # e.g. 1024

# --- Load model once ---
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = ModClassifier(num_classes=len(CLASSES), input_len=INPUT_LEN)
_model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
_model.to(_device)
_model.eval()


def predict_modulation(iq_samples: np.ndarray) -> dict:
    """
    Predict the modulation type of an IQ capture.

    Args:
        iq_samples: numpy array of shape (1024, 2) or (num_windows, 1024, 2)
                    with I in column 0, Q in column 1.

    Returns:
        dict with keys:
            'class'      — predicted class name, e.g. 'QPSK'
            'confidence' — softmax probability of the predicted class (0-1)
            'all_probs'  — dict of class_name -> probability, for all classes
    """
    if iq_samples.ndim == 2:
        if iq_samples.shape != (INPUT_LEN, 2):
            raise ValueError(
                f"Expected IQ samples of shape ({INPUT_LEN}, 2), got {iq_samples.shape}. "
                "Slice or pad your capture to match the trained input length."
            )
        # Shape to (batch=1, channels=2, length=1024) to match training
        x = torch.from_numpy(iq_samples).float().permute(1, 0).unsqueeze(0)
        x = x.to(_device)

        with torch.no_grad():
            logits = _model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    elif iq_samples.ndim == 3:
        if iq_samples.shape[1:] != (INPUT_LEN, 2):
            raise ValueError(
                f"Expected windows of shape (N, {INPUT_LEN}, 2), got {iq_samples.shape}."
            )
        n = len(iq_samples)
        if n == 0:
            raise ValueError("Empty window array provided.")
        # If multiple windows are available, avoid startup edge transients by evaluating
        # on windows 1..min(n, 17), falling back to window 0 only if n <= 2.
        eval_batch = iq_samples[1:min(n, 17)] if n > 2 else iq_samples[:min(n, 16)]
        x = torch.from_numpy(eval_batch).float().permute(0, 2, 1)
        x = x.to(_device)

        with torch.no_grad():
            logits = _model(x)
            probs = F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()

    else:
        raise ValueError(f"Expected 2D or 3D array, got {iq_samples.ndim}D array.")

    pred_idx = int(np.argmax(probs))

    return {
        "class": CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
    }