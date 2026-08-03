import librosa
import numpy as np


def extract_spectral_rolloff(
    window: np.ndarray,
    sr: int
) -> float:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    rolloff = librosa.feature.spectral_rolloff(
        y=window,
        sr=sr
    )

    return float(np.mean(rolloff))