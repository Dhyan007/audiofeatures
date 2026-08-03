import librosa
import numpy as np


def extract_spectral_flatness(
    window: np.ndarray
) -> float:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    flatness = librosa.feature.spectral_flatness(
        y=window
    )

    return float(np.mean(flatness))