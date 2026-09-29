# 📡 Sanchar (संचार)

## Automated RF Signal Analysis & Parameter Extraction

> **Sanchar (संचार)** is an automated RF signal analysis system designed to analyze raw `.IQ` and `.WAV` signal recordings and extract important signal parameters, modulation information, synchronization details, FEC/interleaving information, and decoded bitstream or payload data through a unified graphical interface.

---

## 🏆 Smart India Hackathon 2026

| **Parameter** | **Details** |
|---|---|
| **Problem Statement ID** | **SIH26147** |
| **Problem Statement** | **Automated model for analysis of .IQ and .wav files along with signal parameter extraction** |
| **Organization** | **National Technical Research Organisation (NTRO)** |
| **Theme** | **Space Technology** |
| **Category** | **Software** |
| **Project Name** | **Sanchar (संचार)** |

---

## 📖 Overview

Modern RF signal captures can contain a large amount of raw information, while important parameters such as sampling frequency, modulation type, symbol rate, interleaving scheme, and FEC may not always be available with the recording.

Traditional analysis often requires analysts to manually inspect waveforms, spectra, constellations, synchronization patterns, and bitstreams using multiple tools.

**Sanchar** brings these stages together into a single automated workflow.

The system accepts raw `.IQ` and `.WAV` recordings and processes them through signal preprocessing, modulation classification, spectral analysis, synchronization, demodulation, de-interleaving, FEC decoding, bitstream analysis, and payload extraction.

The results are presented through a PyQt6-based desktop dashboard with waveform, spectrogram, PSD, constellation, and bitstream views.

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [Proposed Solution](#-proposed-solution)
- [Key Features](#-key-features)
- [Core System Architecture](#-core-system-architecture)
- [Technical Approach](#-technical-approach)
- [Signal Processing Pipeline](#-signal-processing-pipeline)
- [Modulation Classification](#-modulation-classification)
- [Spectral Analysis](#-spectral-analysis)
- [Demodulation & Synchronization](#-demodulation--synchronization)
- [De-interleaving & FEC](#-de-interleaving--fec)
- [Bitstream & Payload Analysis](#-bitstream--payload-analysis)
- [Signal Visualization](#-signal-visualization)
- [Synthetic Signal Testing](#-synthetic-signal-testing)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Team](#-team)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

# 🇮🇳 Problem Statement

## 🎯 Problem Overview

RF signal analysis can become difficult when the captured data contains only raw signal samples without complete information about the signal structure.

The available capture may not clearly provide:

- Sampling frequency
- Symbol rate
- Modulation scheme
- Carrier information
- Frame structure
- Interleaving method
- Forward Error Correction (FEC)
- Header and payload boundaries

The problem statement focuses on developing an automated system capable of analyzing `.IQ` and `.WAV` signal recordings and extracting useful signal parameters.

### Main Challenges

1. **Raw Signal Input**

   Signal recordings may arrive as raw `.IQ` or `.WAV` data without complete metadata.

2. **Unknown Signal Parameters**

   Important parameters such as modulation, sampling frequency, and symbol rate may not be directly available.

3. **Different Signal Conditions**

   Signals can vary depending on the sensor, recording location, frequency range, noise level, and capture conditions.

4. **Manual Analysis**

   Conventional signal analysis may require multiple tools and manual inspection of different signal representations.

5. **Decoding Complexity**

   Recovering useful information may require synchronization, demodulation, de-interleaving, FEC decoding, and bitstream analysis.

---

# 💡 Proposed Solution

Sanchar provides a unified automated pipeline:

```text
Raw RF Capture
     │
     ▼
File Ingestion
(.IQ / .WAV)
     │
     ▼
Preprocessing
DC Removal + Normalization
     │
     ▼
Signal Segmentation
     │
     ├───────────────► Waveform
     │
     ├───────────────► Spectrogram
     │
     └───────────────► PSD
     │
     ▼
Modulation Classification
(CNN)
     │
     ▼
Spectral & Parameter Analysis
     │
     ▼
Demodulation
     │
     ▼
Frame Synchronization
     │
     ▼
De-interleaving
     │
     ▼
FEC Decoding
     │
     ▼
Bitstream Analysis
     │
     ▼
Payload Extraction
     │
     ▼
Analysis Report
---

###⚡ Key Features
1. **Dual-Format Signal Input**

Sanchar supports:
.IQ
.WAV
The input data is converted into a suitable complex I/Q representation for further processing.

2. **Automatic Signal Preprocessing**

The preprocessing stage includes:

DC offset removal
Signal normalization
I/Q separation
Signal windowing
Frame segmentation
Sample preparation for analysis

I/Q samples are normalized for consistent downstream processing.

3. **Automatic Modulation Classification**

A 1D CNN-based classifier is used to identify the modulation family/type from signal samples.

Supported modulation families include:

FSK Family
2-FSK
4-FSK
MSK
GMSK
PSK Family
BPSK
QPSK
8-PSK
16-PSK
π/4-QPSK
QAM Family
16-QAM
64-QAM
256-QAM
1024-QAM

The classifier also provides a confidence value for the detected modulation.
4. Spectral Analysis

Sanchar uses spectral analysis to extract signal characteristics such as:

Sampling frequency
Occupied bandwidth
Peak frequency
Power spectral density
Noise floor
Center-frequency offset
Frequency-domain characteristics

Welch PSD estimation is used for power spectral analysis.

5. Constellation Analysis

The constellation view displays complex I/Q samples in the I-Q plane.

It helps visualize:

Modulation structure
Symbol distribution
I/Q balance
Signal spread
Noise effects
Average magnitude
The application also displays useful constellation statistics alongside the visualization.

6. Waterfall / Spectrogram Analysis

The spectrogram provides a time-frequency representation of the signal.

It allows the user to observe:

Frequency changes over time
Signal activity
Bursts
Frequency transitions
Time-varying signal characteristics

🏛 Core System Architecture

Sanchar is organized as a sequence of processing stages.




🔬 Technical Approach
1. File Ingestion

The application first identifies the input format.

.IQ  → Complex I/Q samples
.WAV → Stereo/recorded IQ samples

Metadata such as sample rate and signal length is extracted when available.

2. Preprocessing

The signal passes through:
Raw Samples
     ↓
DC Removal
     ↓
Normalization
     ↓
I/Q Separation
     ↓
Windowing
     ↓
Frame Segmentation


3. Evidence Collection

Different analysis modules independently examine the signal.
CNN Classification
        +
Spectral Analysis
        +
Constellation
        +
Cyclostationary Analysis
        +
Synchronization Evidence
        ↓
Combined Signal Information

🤖 Modulation Classification

Sanchar uses a 1D CNN-based model for modulation classification.

The classification pipeline is:
I/Q Signal
    ↓
Preprocessing
    ↓
Signal Frames
    ↓
1D CNN
    ↓
Modulation Class
    +
Confidence

The model can classify signals belonging to different PSK, QAM and FSK families.

The detected modulation is then used to select the appropriate demodulation path.

🔄 Demodulation & Synchronization

After modulation identification, the appropriate demodulation path is selected.

The processing stage can include:

Carrier recovery
Clock recovery
Symbol synchronization
Phase correction
Bit recovery

For phase-based signals, Costas-loop based carrier recovery can be used.

Polyphase clock synchronization is used where applicable to recover symbol timing.

Frame Synchronization

After demodulation, the recovered bitstream is searched for known synchronization patterns.

For example:
Received Bitstream
       ↓
Correlation
       ↓
Preamble / Sync Detection
       ↓
Frame Boundary
       ↓
Header + Payload

The synchronization stage can also test phase/polarity possibilities when required.

🔀 De-interleaving & FEC

Many communication systems use interleaving and Forward Error Correction to improve reliable data transmission.

Sanchar provides processing stages for:

De-interleaving

Possible approaches include:

Block interleaving
Convolutional interleaving
Diagonal interleaving
Pseudo-random approaches

The system can evaluate candidate configurations and use subsequent decoding results as feedback.


Forward Error Correction

Supported/implemented decoding paths include:
Viterbi

Used for convolutionally encoded data.

Reed-Solomon

Used for block-based error correction.

LDPC

Included as part of the planned/extended decoding architecture.

The decoded output is checked for consistency before being passed to later stages.


🔍 Bitstream & Payload Analysis

After demodulation and decoding, the recovered bitstream is analyzed.

The system can display:

Raw bits
Bitstream offsets
Header information
Payload boundaries
Hex representation
Decoded bytes
Payload data

The processing flow is:
Demodulated Bits
      ↓
Frame Synchronization
      ↓
De-interleaving
      ↓
FEC Decoding
      ↓
Bitstream Correlation
      ↓
Header / Payload Detection
      ↓
Payload Output


🧪 Synthetic Signal Testing

Sanchar includes a synthetic signal generation workflow for testing the complete analysis pipeline.

Synthetic signals can be generated with configurable parameters such as:

Modulation
Number of symbols
Sample rate
Samples per symbol
SNR
Carrier frequency offset
Random seed
FEC scheme
Interleaver
Synchronization preamble

🛠 Technology Stack

| **Technology**         | **Purpose**                                          |
| ---------------------- | ---------------------------------------------------- |
| **Python**             | Core application and signal-processing logic         |
| **PyQt6**              | Desktop graphical user interface                     |
| **pyqtgraph**          | Waveform, PSD and constellation visualization        |
| **Plotly**             | Interactive visualization where applicable           |
| **NumPy**              | Numerical computation and signal arrays              |
| **SciPy**              | FFT, filtering, PSD, correlation and signal analysis |
| **PyTorch**            | CNN-based modulation classification                  |
| **GNU Radio**          | Signal processing and demodulation components        |
| **Matplotlib**         | Signal plots and report visualizations               |
| **Conda / RadioConda** | Python environment and RF-related packages           |
| **Git / GitHub**       | Version control and source management                |


📂 Project Structure
signal-analyzer/
│
├── main.py
│
├── gui/
│   ├── main_window.py
│   ├── params_panel.py
│   ├── spectrogram_panel.py
│   ├── constellation_panel.py
│   ├── waveform_panel.py
│   ├── bitstream_panel.py
│   ├── synthetic_dialog.py
│   ├── report_generator.py
│   └── theme.py
│
├── models/
│   └── inference.py
│
├── correlation/
│   └── sync_correlator.py
│
├── gnuradio_pipeline/
│   └── ...
│
├── assets/
│   └── ntro_logo.png
│
├── synthetic/
│   └── ...
│
├── README.md
│
└── signal_analysis_report_verified.pdf


💻 Installation
Prerequisites

Recommended environment:
Python 3.11
Conda / RadioConda
Windows / Linux

Create / Activate Environment

If the environment has already been created:
conda activate signal-analyzer

Install Required Packages

Install the required Python packages used by the application:
pip install numpy scipy matplotlib pyqt6 pyqtgraph pandas
Additional packages required by the ML and RF-processing components can be installed according to the project environment.

▶️ Running the Application

Clone the repository:
git clone https://github.com/Divya-Mondal-14/Sanchar.git

Move into the project directory:
cd Sanchar

Run the application:
python main.py


👥 Team
Sanchar — SIH 2026

Team:
Government College of Engineering and Leather Technology (GCELT), Kolkata

Problem Statement: SIH26147

Organization: National Technical Research Organisation (NTRO)

Theme: Space Technology

📜 License

This project is developed as a Smart India Hackathon 2026 prototype.

The project is intended for research, development, demonstration, and academic purposes.


🙏 Acknowledgements

We acknowledge the support and problem statement provided through Smart India Hackathon 2026 and the National Technical Research Organisation (NTRO).

We also acknowledge the open-source scientific and engineering ecosystem used in the development of the prototype, including:

Python
NumPy
SciPy
PyTorch
PyQt6
pyqtgraph
Matplotlib
GNU Radio
Conda / RadioConda








