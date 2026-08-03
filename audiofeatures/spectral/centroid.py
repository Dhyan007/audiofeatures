import librosa
import numpy as np


def extract_spectral_centroid(
    window: np.ndarray,
    sr: int
) -> float:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    centroid = librosa.feature.spectral_centroid(
        y=window,
        sr=sr
    )

    return float(np.mean(centroid))