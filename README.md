
# audiofeatures

![PyPI](https://img.shields.io/pypi/v/audiofeatures)
![Python](https://img.shields.io/pypi/pyversions/audiofeatures)
![License](https://img.shields.io/pypi/l/audiofeatures)
![Downloads](https://static.pepy.tech/badge/audiofeatures)
![GitHub stars](https://img.shields.io/github/stars/Dhyan007/audiofeatures?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/Dhyan007/audiofeatures)

A lightweight Python library for audio preprocessing and handcrafted feature extraction for Machine Learning, Deep Learning, Speech Processing, and Music Information Retrieval (MIR).

---

## Overview

`audiofeatures` provides an end-to-end pipeline for extracting handcrafted audio descriptors from music, speech, and environmental sounds. It performs preprocessing, window-based analysis, and returns the extracted features as a **Pandas DataFrame** ready for machine learning workflows.

---

# Why audiofeatures?

- ✅ 185 handcrafted audio features
- ✅ Built-in preprocessing pipeline
- ✅ Window-based feature extraction
- ✅ Dataset-level feature extraction
- ✅ Pandas DataFrame output
- ✅ Machine Learning ready
- ✅ Supports WAV, MP3, FLAC, OGG and M4A
- ✅ Simple, clean API

---

# Installation

```bash
pip install audiofeatures
```

---

# Quick Start

## Single Audio File

```python
from audiofeatures import extract_features

df = extract_features(
    audio_path="song.wav",
    window_size=0.5
)

print(df.head())
```

## Entire Dataset

```python
from audiofeatures import extract_dataset

df = extract_dataset(
    audio_folder="dataset/audio",
    window_size=0.5,
    output_csv="features.csv"
)

print(df.head())
```

---

# Feature Categories

| Category | Description |
|-----------|-------------|
| Preprocessing | Cleans and standardizes audio before feature extraction |
| Temporal | Describes signal characteristics in the time domain |
| Spectral | Describes frequency-domain properties |
| Cepstral | MFCC-based spectral envelope descriptors |
| Tonal | Harmonic and musical pitch descriptors |
| Pitch | Fundamental frequency statistics |
| Rhythm | Onset and rhythmic activity descriptors |

---

# Feature Descriptions

## Audio Preprocessing

### Mono Conversion
Converts multi-channel audio into a single channel for consistent processing.

### Resampling
Converts audio to a common sampling rate (default: 22050 Hz).

### Normalization
Scales audio amplitudes to a consistent range without altering relative dynamics.

### Silence Removal
Removes silent regions to focus analysis on meaningful audio content.

### Audio Cropping
Allows extraction from a selected region of an audio file.

---

## Temporal Features

### RMS Energy
Measures the average signal energy within a window. Higher values generally indicate louder audio.

### Zero Crossing Rate (ZCR)
Counts how often the waveform crosses zero. Useful for distinguishing voiced, unvoiced, and noisy sounds.

### Energy
Represents the total signal energy in the analysis window.

### Peak Amplitude
Maximum absolute amplitude observed in the window.

### Crest Factor
Ratio between peak amplitude and RMS energy. Indicates impulsive or transient signals.

### Dynamic Range
Measures the difference between the quietest and loudest portions of the signal.

### Energy Variability
Measures how energy changes across short frames inside the window.

### Amplitude Variability
Measures variation in waveform amplitude over time.

---

## Spectral Features

### Spectral Centroid
Represents the "center of mass" of the spectrum. Higher values correspond to brighter sounds.

### Spectral Bandwidth
Measures the spread of frequencies around the spectral centroid.

### Spectral Rolloff
Frequency below which most (typically 85%) of the spectral energy is contained.

### Spectral Contrast
Measures differences between spectral peaks and valleys across frequency bands.

---

## Cepstral Features

### MFCC
Captures the spectral envelope using the Mel scale. Widely used in speech and music analysis.

### Delta MFCC
First-order derivative of MFCCs describing short-term spectral changes.

### Delta-Delta MFCC
Second-order derivative of MFCCs describing spectral acceleration.

---

## Tonal Features

### Chroma
Represents the intensity of the 12 musical pitch classes independent of octave.

### Tonnetz
Captures harmonic relationships between pitches for tonal analysis.

---

## Pitch Features

### Pitch Mean
Average estimated fundamental frequency.

### Pitch Standard Deviation
Variation of pitch over the analysis window.

### Pitch Minimum
Lowest detected pitch.

### Pitch Maximum
Highest detected pitch.

---

## Rhythm Features

### Onset Strength Mean
Average strength of detected note or beat onsets.

### Onset Strength Standard Deviation
Variation in onset strength.

### Onset Strength Maximum
Strongest detected onset within the window.

---

# Supported Audio Formats

- WAV
- MP3
- FLAC
- OGG
- M4A

---

# Returned Data

The library returns a **Pandas DataFrame**.

For `extract_features()`:
- One row per analysis window.

For `extract_dataset()`:
- One row per analysis window.
- Includes `song_id` and `filename` columns.

---

# Applications

- Music Emotion Recognition
- Speech Emotion Recognition
- Music Information Retrieval (MIR)
- Environmental Sound Classification
- Audio Event Detection
- Bird Sound Recognition
- Feature Engineering for Machine Learning

---

# Roadmap

- Parallel feature extraction
- Additional audio descriptors
- Feature visualization utilities
- Real-time streaming support
- GPU acceleration (optional)

---

# License

MIT License

---

# Author

**Dhyan Sudheer**

Integrated MSc Data Science  
Amrita Vishwa Vidyapeetham
