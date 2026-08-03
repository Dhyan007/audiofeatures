import librosa
import numpy as np

from ..preprocessing import validate_waveform


def extract_tonnetz(
    window: np.ndarray,
    sr: int
) -> dict:

    validate_waveform(
        window,
        sr
    )

    features = {}

    try:

        harmonic = librosa.effects.harmonic(window)

        if np.allclose(harmonic, 0):

            for i in range(6):
                features[f"tonnetz_{i+1}_mean"] = 0.0
                features[f"tonnetz_{i+1}_std"] = 0.0

            return features

        
        chroma = librosa.feature.chroma_stft(
            y=harmonic,
            sr=sr,
            tuning=0.0,
            n_fft=min(1024, len(harmonic))
        )

        tonnetz = librosa.feature.tonnetz(
            chroma=chroma,
            sr=sr
        )

        for i in range(6):
            features[f"tonnetz_{i+1}_mean"] = float(np.mean(tonnetz[i]))
            features[f"tonnetz_{i+1}_std"] = float(np.std(tonnetz[i]))

    except Exception:

        for i in range(6):
            features[f"tonnetz_{i+1}_mean"] = 0.0
            features[f"tonnetz_{i+1}_std"] = 0.0

    return features