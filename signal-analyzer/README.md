# SIH Signal Analyzer 🛰️📡
> **Autonomous Blind Signal Classification, Demodulation, Frame Synchronization, De-interleaving & Forward Error Correction (FEC) Pipeline**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![GNU Radio](https://img.shields.io/badge/GNU%20Radio-DSP%20Demodulation-brightgreen.svg)](https://www.gnuradio.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6%20%26%20PyQtGraph-green.svg)](https://riverbankcomputing.com/software/pyqt/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()

---

## 📖 Table of Contents
1. [Overview & Operational Problem](#-overview--operational-problem)
2. [End-to-End Pipeline Architecture](#-end-to-end-pipeline-architecture)
3. [Prerequisites](#-prerequisites)
4. [Step-by-Step Setup Guide](#-step-by-step-setup-guide)
   - [Option A: Conda / Radioconda (Recommended)](#option-a-conda--radioconda-recommended)
   - [Option B: Python Virtual Environment (`venv`)](#option-b-python-virtual-environment-venv)
5. [Running the Application](#-running-the-application)
6. [Testing & Sanity Verification](#-testing--sanity-verification)
7. [GUI Features & How to Use](#-gui-features--how-to-use)
8. [Project Structure](#-project-structure)
9. [Troubleshooting & FAQs](#-troubleshooting--faqs)
10. [Git & Repository Guidelines](#-git--repository-guidelines)

---

## 🎯 Overview & Operational Problem

In signal surveillance, non-cooperative RF intelligence (ELINT/SIGINT), and SDR monitoring, intercepted RF transmissions arrive **blind** with no prior metadata. The receiver has zero prior knowledge of:
1. **Modulation Scheme** (BPSK, QPSK, GMSK, etc.)
2. **Symbol Rate / Samples Per Symbol (SPS)** and Baud Rate
3. **Carrier Frequency Offset (CFO)** and phase drift
4. **Frame Synchronization Markers** (Barker, CCSDS ASM, HDLC)
5. **Interleaver Structure** (Block interleavers, Convolutional/Ramsey-II)
6. **Forward Error Correction (FEC)** (Viterbi Convolutional, Reed-Solomon)

### The Solution: Hybrid DL + Classical DSP
- **Deep Learning (1D-CNN)**: Used where analytical models degrade—for Automatic Modulation Recognition (AMR) under low SNR and unknown channel conditions.
- **Classical DSP & Algebraic Decoding**: Used where exact mathematical guarantees are non-negotiable—GNU Radio constellation tracking, Costas Loops, Oerder-Meyr clock sync, blind de-interleaving search, and Viterbi / Reed-Solomon syndrome validation.

---

## 🏗️ End-to-End Pipeline Architecture

```
       RAW RF CAPTURE (.wav / .iq) or SYNTHETIC GENERATOR
                           │
                           ▼
               ┌───────────────────────┐
               │  File Ingestion &     │  Fixed-point to float32 [-1, 1]
               │  Windowing (1024x2)   │  Reconstruct I/Q orthogonal channels
               └───────────┬───────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
         ┌───────────────┐   ┌────────────────────────┐
         │ Deep Learning │   │ Spectral Analysis      │  Welch's PSD, 99% OBW,
         │ Modulation    │   │ & Symbol Rate (SPS)    │  Oerder-Meyr Non-Linear
         │ Classifier    │   │ Clock Recovery         │  Envelope Cyclostationarity
         └───────┬───────┘   └───────────┬────────────┘
                 │                       │
                 └─────────┬─────────────┘
                           ▼
               ┌───────────────────────┐
               │ GNU Radio Real-time   │  Constellation Decoders, Costas Loop,
               │ C++ DSP Demodulation  │  RRC Polyphase Clock Sync, FM Discriminator
               └───────────┬───────────┘
                           │  Raw demodulated bits
                           ▼
               ┌───────────────────────┐
               │ Frame Sync & Phase    │  Cross-correlation across 4 phase states
               │ Ambiguity Resolution  │  CCSDS ASM, Barker, HDLC; payload extraction
               └───────────┬───────────┘
                           │  Bit-aligned, phase-corrected stream
                           ▼
               ┌───────────────────────┐
               │ Blind De-Interleaver  │  Hypothesis testing: Block (4x8, 8x8, 16x16),
               │ Search & Inversion    │  Convolutional (Ramsey II)
               └───────────┬───────────┘
                           │  De-interleaved stream
                           ▼
               ┌───────────────────────┐
               │ Blind FEC Decoder     │  Viterbi (k=7 CCSDS, k=3),
               │ & Syndrome Validator  │  Reed-Solomon (RS 128,120) syndrome check
               └───────────┬───────────┘
                           │
                           ▼
                 RECOVERED PAYLOAD BITS / ASCII / HEX
```

---

## ⚙️ Prerequisites

1. **Operating System**: Windows 10/11 (or 64-bit Linux).
2. **Python**: Version **3.10, 3.11, or 3.12** (Python 3.11 is strongly recommended).
3. **Conda Package Manager** (Miniconda, Anaconda, or [Radioconda](https://github.com/ryanvolz/radioconda)).
   > ⚠️ **Important:** GNU Radio has compiled C++ DSP acceleration binaries (`gnuradio-runtime`, `gnuradio-digital`) that are **not** installable via standard `pip install`. Installing GNU Radio via Conda is the easiest and most reliable method on Windows.

---

## 🚀 Step-by-Step Setup Guide

Clone the repository to your local machine:
```bash
git clone <YOUR_REPO_URL>
cd signal-analyzer
```

Choose **Option A** (Simplest) or **Option B**:

---

### Option A: Conda / Radioconda (Recommended)

1. **Create the Conda Environment** with GNU Radio:
   ```bash
   conda create -n signal_analyzer -c conda-forge python=3.11 gnuradio -y
   ```

2. **Activate the Environment**:
   ```bash
   conda activate signal_analyzer
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**:
   ```bash
   python -c "import gnuradio.gr, PyQt6, torch; print('Setup successful!')"
   ```

---

### Option B: Python Virtual Environment (`venv`)

If you prefer running inside a standard Python `venv`:

1. **Create and Activate a virtual environment**:
   ```bash
   # On Windows Command Prompt / PowerShell:
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Link GNU Radio from Conda**:
   If GNU Radio is installed in a Conda environment (e.g. `signal_analyzer` or Radioconda), ensure the Conda path is discoverable.
   The system automatically scans:
   - Your active `CONDA_PREFIX`
   - `%USERPROFILE%\radioconda\envs\signal_analyzer`
   - `%USERPROFILE%\miniconda3\envs\signal_analyzer`
   - `%USERPROFILE%\anaconda3\envs\signal_analyzer`

   *(Optional)* To explicitly link them into your `venv`, create a `.pth` file inside `venv\Lib\site-packages\gnuradio_path.pth` containing the absolute path to your Conda environment's `Lib\site-packages`:
   ```text
   C:\Users\<YOUR_USERNAME>\miniconda3\envs\signal_analyzer\Lib\site-packages
   ```

---

## 💻 Running the Application

### Method 1: Using the Batch Launcher (Windows)
Double-click `run.bat` in the project root, or execute:
```cmd
run.bat
```

### Method 2: Command Line
Ensure your environment is active (`conda activate signal_analyzer` or `.\venv\Scripts\activate`), then run:
```bash
python main.py
```

The PyQt6 Dark-Mode Modern GUI will launch.

---

## 🧪 Testing & Sanity Verification

Before modifying code, run the pre-flight checks:

### 1. Model Inference Smoke Test
Validates that the PyTorch 1D-CNN classifier loads and accurately predicts test signals from RadioML:
```bash
python models/test_inference.py
```
*Expected output: All test cases show `[correct]`.*

### 2. End-to-End File Loader & Inference Chain Test
Loads a captured WAV file and runs slice windowing and classification:
```bash
python chain_test.py synthetic_QPSK_1000kHz_SNR20dB.wav
```
*Expected output: Displays sample rate, window count, predicted modulation class, and confidence scores.*

---

## 🖥️ GUI Features & How to Use

### 1. Loading RF Captures
- Click **Open File** (or `File -> Open File...`).
- Supported formats:
  - **WAV files** (`.wav`): Stereo 16-bit PCM (Left channel = In-phase $I$, Right channel = Quadrature $Q$).
  - **Raw IQ files** (`.iq`): Continuous interleaved $I/Q$ binary (supports float32, int16, uint8).
- Bundled demo files ready for immediate testing:
  - `synthetic_BPSK_1000kHz_SNR20dB.wav`
  - `synthetic_QPSK_1000kHz_SNR20dB.wav`
  - `synthetic_QPSK_viterbi_k7_block_4x8_1000kHz_SNR20dB.wav`
  - `test_A.wav`, `test_B.wav`, `test_C.wav`

### 2. Synthetic Signal Generator (No Hardware Required!)
No SDR dongles or physical antennas are required to test features:
1. Click **Generate Synthetic Signal** in the toolbar (or `Tools -> Synthetic Signal Generator`).
2. Select:
   - **Modulation**: `BPSK`, `QPSK`, `GMSK`
   - **Signal Quality**: SNR slider ($-5\text{ dB}$ to $+30\text{ dB}$), Carrier Frequency Offset (CFO), Phase Offset.
   - **Interleaver**: `None`, `Block (4x8, 8x8, 16x16)`, `Convolutional (Ramsey II)`
   - **FEC**: `None`, `Viterbi (k=7 CCSDS Rate 1/2)`, `Viterbi (k=3 Rate 1/2)`, `Reed-Solomon (RS 128,120)`
   - **Sync Preamble**: `CCSDS ASM (32-bit)`, `Barker (13-bit)`, `HDLC Flag`
3. Click **Generate & Inject** to stream directly into the analysis pipeline.

### 3. Real-Time Visualizers
- **Constellation Plot**: Displays $I$ vs $Q$ constellation points post-Costas Loop recovery with decision boundaries.
- **Welch Spectrogram & PSD**: Real-time spectral power distribution and 99% Occupied Bandwidth (OBW) calculations.
- **Bitstream & Protocol Analyzer**:
  - Raw demodulated bits
  - Frame sync markers highlighted in color
  - De-interleaving transformation metrics
  - FEC syndrome check & bit error correction stats
  - ASCII and Hex payload decoders

---

## 📁 Project Structure

```
signal-analyzer/
├── correlation/              # Frame sync detection & phase ambiguity resolution
│   ├── sync_correlator.py    # Cross-correlator for Barker, CCSDS, HDLC sync words
├── gnuradio_pipeline/        # GNU Radio DSP & algebraic channel decoders
│   ├── demod.py              # GNU Radio C++ DSP wrappers (Costas loop, clock sync)
│   ├── deinterleave.py       # Blind matrix & convolutional de-interleaving search
│   └── fec_decode.py         # Viterbi and Reed-Solomon algebraic decoders
├── gui/                      # PyQt6 application interface & panels
│   ├── main_window.py        # Central dashboard & background analysis QThread
│   ├── constellation_panel.py# IQ constellation scatter plot
│   ├── spectrogram_panel.py  # Welch PSD & frequency spectrum
│   ├── bitstream_panel.py    # Bitstream, sync lock, and payload viewer
│   ├── params_panel.py       # Live classification telemetry & SNR metrics
│   └── synthetic_dialog.py   # In-app synthetic signal generator dialog
├── models/                   # Deep learning automatic modulation recognition
│   ├── classifier.py         # 1D-CNN PyTorch architecture
│   ├── inference.py          # Windowed inference with softmax confidence
│   ├── modclassifier_best.pth# Pre-trained RadioML weights (DO NOT DELETE)
│   ├── label_map.json        # Class index to modulation mapping
│   └── test_inference.py     # Inference verification script
├── preprocessing/            # Signal conditioning and synthesis
│   ├── file_loader.py        # Ingestion for .wav and raw .iq formats
│   ├── spectral_features.py  # Welch PSD, OBW, Oerder-Meyr symbol rate
│   ├── synthetic_iq.py       # Pure-math RF signal synthesizer with channel models
│   └── waveform_view.py      # Time-domain I/Q envelope rendering
├── test_data/                # Ground truth test samples
│   └── test_samples.json     # RadioML validation sample vectors
├── requirements.txt          # Python pip dependencies
├── run.bat                   # 1-click Windows runner
├── chain_test.py             # End-to-end integration test
├── main.py                   # Application entry point
└── README.md                 # Project documentation (this file)
```

---

## ❓ Troubleshooting & FAQs

### Q1: `ImportError: GNU Radio Python bindings could not be found`
**Cause**: The Python runtime cannot find GNU Radio DLLs and site-packages.  
**Fix**:
1. Make sure you installed GNU Radio in your Conda environment:
   ```bash
   conda activate <your_conda_env>
   conda install -c conda-forge gnuradio -y
   ```
2. Run the application from within that active Conda environment:
   ```bash
   conda activate <your_conda_env>
   python main.py
   ```
3. If running from a `venv`, ensure `venv\Lib\site-packages\gnuradio_path.pth` contains the path to your Conda environment's `Lib\site-packages`.

### Q2: `qt.qpa.plugin: Could not find the Qt platform plugin "windows"`
**Cause**: PyQt6 cannot locate the platform plugins.  
**Fix**: Reinstall PyQt6 cleanly:
```bash
pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
pip install PyQt6 pyqtgraph
```

### Q3: `No full windows available — file shorter than 1024 samples`
**Cause**: The neural network classifier expects slices of 1024 complex samples.  
**Fix**: Supply an RF capture file with at least 1,024 samples. You can use any of the bundled `synthetic_*.wav` files or generate one using the in-app Synthetic Signal Generator.

---

## 🤝 Git & Repository Guidelines

To keep the repository clean and avoid conflicts:

1. **Do NOT commit virtual environments**:
   The `venv/`, `venv312/`, or `.conda/` directories are ignored by `.gitignore`.
2. **Do NOT delete `models/modclassifier_best.pth`**:
   This is the trained neural network model required for blind classification.
3. **Do NOT commit raw multi-GB SDR dumps**:
   Keep demo test captures under 5 MB. Store large datasets in external storage (Google Drive / S3).
4. **Before pushing your branch**:
   Run sanity tests to make sure you didn't break core pipelines:
   ```bash
   python models/test_inference.py
   python chain_test.py synthetic_BPSK_1000kHz_SNR20dB.wav
   ```

