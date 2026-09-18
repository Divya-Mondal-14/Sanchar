# SIH Signal Analyzer 🛰️📡

> **Autonomous Blind Signal Classification, Demodulation, Frame Synchronization, De-interleaving & FEC Recovery Pipeline.**

[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![GNU Radio](https://img.shields.io/badge/GNU%20Radio-DSP%20Demodulation-brightgreen.svg)](https://www.gnuradio.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6%20%26%20PyQtGraph-green.svg)](https://riverbankcomputing.com/software/pyqt/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Quick Setup for Teammates (Conda Environment)](#-quick-setup-for-teammates-conda-environment)
3. [Running the Application](#-running-the-application)
4. [Sanity Tests & Verification](#-sanity-tests--verification)
5. [Application Features & Usage](#-application-features--usage)
6. [Repository Structure & Git Readiness](#-repository-structure--git-readiness)
7. [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🎯 Project Overview

This repository provides an autonomous pipeline for blind RF signal intelligence:
- **Blind Modulation Classification**: 1D-CNN deep learning model trained on RadioML recognizing BPSK, QPSK, GMSK, etc.
- **Spectral Estimation**: Welch Power Spectral Density (PSD), 99% Occupied Bandwidth (OBW), and Oerder-Meyr symbol clock recovery.
- **DSP Demodulation**: Hardware-accelerated GNU Radio blocks (Costas Loop, Polyphase Clock Sync, Constellation Decoders).
- **Frame Synchronization**: Cross-correlation across phase ambiguity states (CCSDS ASM 32-bit, Barker 13-bit, HDLC flags).
- **Blind De-interleaving**: Matrix inversion search (4×8, 8×8, 16×16) and Ramsey-II convolutional de-interleavers.
- **Forward Error Correction (FEC)**: Viterbi decoder (k=7 CCSDS Rate 1/2, k=3) and Reed-Solomon (RS 128,120) syndrome validation.
- **Interactive PyQt6 GUI**: Real-time I/Q constellation diagram, Welch spectrogram, time-domain waveform, and bitstream analyzer with in-app synthetic signal generation.

---

## ⚡ Quick Setup for Teammates (Conda Environment)

> 💡 **For Teammates Pulling This Repository**:  
> Since you already have Conda installed/configured on your system, follow these steps to replicate the exact environment.

### 1. Open Terminal & Activate Your Conda Environment
Open **Anaconda Prompt**, **Miniconda Prompt**, or a terminal where conda is initialized:
```bash
conda activate <your_conda_env_name>
```
*(Recommended Python version: **Python 3.10 or 3.11** for seamless GNU Radio + PyTorch compatibility on Windows).*

---

### 2. Verify / Install GNU Radio (Essential Conda Requirement)
GNU Radio contains compiled C++ DSP binaries (`gnuradio-runtime`, `gnuradio-digital`) that **cannot** be installed via standard `pip`. It must be present in your Conda environment:

```bash
# Check if GNU Radio is already installed in your active environment:
python -c "import gnuradio.gr; print('GNU Radio is installed!')"
```

If not installed yet, run:
```bash
conda install -c conda-forge gnuradio -y
```

---

### 3. Install Python Dependencies
Navigate to the project root and install the required Python packages:

```bash
cd signal-analyzer
pip install -r requirements.txt
```

> **Packages installed via `requirements.txt`:**
> - `torch`: PyTorch runtime for running the 1D-CNN modulation classifier.
> - `numpy`, `h5py`: Numerical matrix processing & vector handling.
> - `PyQt6`, `pyqtgraph`: Dark-mode GUI and GPU-accelerated real-time plotting.
> - `commpy`: Viterbi convolutional decoding (`scikit-commpy`).
> - `reedsolo`: Reed-Solomon algebraic decoding & syndrome validation.

---

### 4. 10-Second Environment Sanity Check
Run this single command in your terminal to verify all core components are functional:
```bash
python -c "import gnuradio.gr, torch, PyQt6, pyqtgraph, commpy, reedsolo; print('>>> ENVIRONMENT FULLY VERIFIED! Ready to run.')"
```

---

## 🚀 Running the Application

Ensure your Conda environment is active:
```bash
cd signal-analyzer
```

### Option A: Using the Batch Runner (Windows)
Double click `run.bat` or run:
```cmd
run.bat
```
*(The launcher automatically checks for active Conda or virtual environments and keeps the terminal open if any error occurs).*

### Option B: Direct Python Launch
```bash
python main.py
```

The PyQt6 Dark-Mode Modern GUI will launch immediately.

---

## 🧪 Sanity Tests & Verification

Before making code changes, run these quick validation tests:

### Test 1: Deep Learning Classifier Smoke Test
Validates that the PyTorch 1D-CNN model architecture, pretrained weights (`modclassifier_best.pth`), and label dictionary load accurately:
```bash
cd signal-analyzer
python models/test_inference.py
```
*Expected output:*
```text
Running inference on real RadioML validation samples...
True: BPSK   -> Predicted: BPSK   (confidence 0.9998)  [correct]
True: QPSK   -> Predicted: QPSK   (confidence 0.9985)  [correct]
True: GMSK   -> Predicted: GMSK   (confidence 0.9991)  [correct]
If all show 'correct', your model loads and predicts correctly locally.
```

### Test 2: File Ingestion & Processing Chain Test
Validates file ingestion, window slicing (1024 complex samples), and inference on a bundled test capture:
```bash
cd signal-analyzer
python chain_test.py synthetic_QPSK_1000kHz_SNR20dB.wav
```
*Expected output:* Prints sample rate, sample count, window slices, predicted modulation, and confidence probabilities.

---

## 🖥️ Application Features & Usage

### 1. Ingesting RF Captures
Click **Open File** (or `File -> Open File...`).
- **WAV format (`.wav`)**: Stereo 16-bit PCM (Channel 1 = In-phase $I$, Channel 2 = Quadrature $Q$).
- **Raw IQ format (`.iq`)**: Interleaved binary I/Q streams (`float32`, `int16`, `uint8`).
- **Bundled Test Files Included**:
  - `synthetic_BPSK_1000kHz_SNR20dB.wav`
  - `synthetic_QPSK_1000kHz_SNR20dB.wav`
  - `synthetic_QPSK_viterbi_k7_conv_depth4_1000kHz_SNR20dB.wav`

### 2. Built-in Synthetic Signal Generator (No SDR Hardware Required)
Click **Generate Synthetic Signal** (or `Tools -> Synthetic Signal Generator`):
- **Modulation**: `BPSK`, `QPSK`, `GMSK`
- **Channel Impairments**: SNR slider ($-5\text{ dB}$ to $+30\text{ dB}$), Carrier Frequency Offset (CFO), Phase Offset.
- **Interleaving**: `None`, `Block (4x8, 8x8, 16x16)`, `Convolutional (Ramsey II)`
- **FEC Schemes**: `None`, `Viterbi (k=7 CCSDS Rate 1/2)`, `Viterbi (k=3 Rate 1/2)`, `Reed-Solomon (RS 128,120)`
- **Synchronization Preamble**: `CCSDS ASM (32-bit)`, `Barker (13-bit)`, `HDLC Flag`
- Click **Generate & Inject** to pipe directly into the live dashboard!

---

## 📁 Repository Structure & Git Readiness

```
sanchaarcopy/
├── .gitignore                     # Repository root gitignore (venv, cache, temporary files)
├── README.md                      # Primary project overview & teammate setup guide
└── signal-analyzer/
    ├── .gitignore                 # Directory-level gitignore safety shield
    ├── correlation/               # Frame sync detection & phase ambiguity resolution
    │   └── sync_correlator.py     # Cross-correlator (Barker, CCSDS, HDLC)
    ├── gnuradio_pipeline/         # GNU Radio DSP & algebraic channel decoders
    │   ├── demod.py               # GNU Radio demodulator & auto-path resolution
    │   ├── deinterleave.py        # Matrix & convolutional de-interleaving search
    │   └── fec_decode.py          # Viterbi and Reed-Solomon decoders
    ├── gui/                       # PyQt6 GUI panels & dashboard
    │   ├── main_window.py         # Main window & background worker thread
    │   ├── constellation_panel.py # IQ constellation scatter visualizer
    │   ├── spectrogram_panel.py   # Welch PSD & dynamic spectrogram
    │   ├── waveform_panel.py      # Time-domain I/Q envelope rendering
    │   ├── bitstream_panel.py     # Bitstream, sync lock, & payload viewer
    │   ├── params_panel.py        # Real-time telemetry, SNR, & class confidence
    │   └── synthetic_dialog.py    # Synthetic signal generator dialog
    ├── models/                    # Deep learning modulation recognition
    │   ├── classifier.py          # 1D-CNN PyTorch model
    │   ├── inference.py           # Model loading & inference logic
    │   ├── modclassifier_best.pth # Pretrained neural network weights (tracked)
    │   ├── label_map.json         # Modulation class index mapping
    │   └── test_inference.py      # Classifier verification script
    ├── preprocessing/             # Signal processing & synthesis
    │   ├── file_loader.py         # WAV and Raw IQ loader
    │   ├── spectral_features.py   # Welch PSD, OBW, and Oerder-Meyr clock recovery
    │   ├── synthetic_iq.py        # Mathematical RF generator & channel models
    │   └── waveform_view.py       # Time-series decimation & envelope calculations
    ├── test_data/                 # Ground-truth validation vectors
    │   └── test_samples.json      # RadioML sample test vectors
    ├── chain_test.py              # End-to-end integration test
    ├── main.py                    # Application launch script
    ├── requirements.txt           # Python pip requirements
    ├── run.bat                    # One-click Windows runner
    └── synthetic_*.wav            # Bundled demo capture files
```

### Git Commit Verification Checklist
- [x] **Virtual environments excluded**: `venv/`, `venv312/`, `.conda/` are strictly ignored by `.gitignore`.
- [x] **Caches & bytecode excluded**: `__pycache__/`, `*.pyc`, `.DS_Store`, `.idea/`, `.vscode/` ignored.
- [x] **Pretrained model weights preserved**: `models/modclassifier_best.pth` (~188 KB) is un-ignored so teammate receives the model.
- [x] **No hardcoded local machine paths**: Dynamic path resolution used for user directories and conda prefixes.
- [x] **Demo test files preserved**: Lightweight `.wav` test captures (~68 KB) are un-ignored for immediate offline testing.

---

## ❓ Troubleshooting & FAQs

### Q1: `ImportError: DLL load failed while importing _runtime_swig: The specified module could not be found`
- **Cause**: Windows cannot find GNU Radio DLL dependencies.
- **Fix**: Run the application from within your active Conda environment where `gnuradio` is installed (`conda activate <your_env>`). `demod.py` automatically injects the active `CONDA_PREFIX\Library\bin` into the Windows DLL search paths.

### Q2: `qt.qpa.plugin: Could not find the Qt platform plugin "windows"`
- **Cause**: Broken or conflicting Qt libraries.
- **Fix**: Reinstall PyQt6 cleanly in your conda environment:
  ```bash
  pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
  pip install PyQt6 pyqtgraph
  ```

### Q3: `FileNotFoundError: models/modclassifier_best.pth`
- **Cause**: Missing weights file.
- **Fix**: Ensure `models/modclassifier_best.pth` is present in your working copy (it is tracked in git and should be cloned with the repository).
