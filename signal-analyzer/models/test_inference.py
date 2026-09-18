"""
Smoke test for models/inference.py using real RadioML samples from Kaggle.

Run with your venv active, from inside the models/ folder:
    python test_inference.py
"""

import json
import os

import numpy as np

from inference import predict_modulation

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SAMPLES_PATH = os.path.join(_THIS_DIR, "..", "test_data", "test_samples.json")

if __name__ == "__main__":
    with open(TEST_SAMPLES_PATH) as f:
        test_samples = json.load(f)

    print("Running inference on real RadioML validation samples...\n")

    for class_name, iq_list in test_samples.items():
        iq = np.array(iq_list, dtype=np.float32)
        result = predict_modulation(iq)
        status = "correct" if result["class"] == class_name else "WRONG"
        print(f"True: {class_name:6s} -> Predicted: {result['class']:6s} "
              f"(confidence {result['confidence']:.4f})  [{status}]")

    print("\nIf all show 'correct', your model loads and predicts correctly locally.")