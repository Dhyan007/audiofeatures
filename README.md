# audiofeatures

A Python library for audio preprocessing and handcrafted feature extraction for machine learning, deep learning, and music information retrieval (MIR).

`audiofeatures` extracts temporal, spectral, cepstral, tonal, pitch, and rhythm features from audio files and returns a Pandas DataFrame.

---

## Features

### Audio Preprocessing
- Mono conversion
- Resampling
- Normalization
- Silence removal
- Audio cropping

### Temporal Features
- RMS Energy
- Zero Crossing Rate
- Energy
- Peak Amplitude
- Crest Factor
- Dynamic Range
- Energy Variability
- Amplitude Variability

### Spectral Features
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Spectral Contrast

### Cepstral Features
- MFCC
- Delta MFCC
- Delta-Delta MFCC

### Tonal Features
- Chroma
- Tonnetz

### Pitch Features
- Pitch Mean
- Pitch Standard Deviation
- Pitch Minimum
- Pitch Maximum

### Rhythm Features
- Onset Strength Mean
- Onset Strength Standard Deviation
- Onset Strength Maximum

---

## Installation

```bash
pip install audiofeatures
```

---

## Quick Start

```python
from audiofeatures import extract_features

df = extract_features("song.wav")

print(df.head())
```

---

## Default Parameters

| Parameter | Default |
|-----------|---------|
| Sample Rate | 22050 Hz |
| Window Size | 0.5 seconds |

---

## Returned Features

- 185 handcrafted audio features
- Window-based extraction
- Pandas DataFrame output

---

## Requirements

- Python 3.9+
- NumPy
- Pandas
- Librosa
- SoundFile

---

## License

MIT License

---

## Author

Dhyan Sudheer
