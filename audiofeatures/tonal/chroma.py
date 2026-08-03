import librosa
import numpy as np

from ..preprocessing import validate_waveform


def extract_chroma(
    window: np.ndarray,
    sr: int
) -> dict:

    validate_waveform(
        window,
        sr
    )

    chroma = librosa.feature.chroma_stft(
        y=window,
        sr=sr,
        tuning=0.0,
        n_fft=min(1024, len(window))
    )

    features = {}

    for i in range(chroma.shape[0]):

        features[f"chroma_{i+1}_mean"] = float(
            np.mean(chroma[i])
        )

        features[f"chroma_{i+1}_std"] = float(
            np.std(chroma[i])
        )

    return features