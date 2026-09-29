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
- [Report Generation](#-report-generation)
- [Impact & Benefits](#-impact--benefits)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Example Workflow](#-example-workflow)
- [Validation](#-validation)
- [Future Scope](#-future-scope)
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

⚡ Key Features
1. Dual-Format Signal Input

Sanchar supports:
.IQ
.WAV
The input data is converted into a suitable complex I/Q representation for further processing.

2. Automatic Signal Preprocessing

The preprocessing stage includes:

DC offset removal
Signal normalization
I/Q separation
Signal windowing
Frame segmentation
Sample preparation for analysis

I/Q samples are normalized for consistent downstream processing.


