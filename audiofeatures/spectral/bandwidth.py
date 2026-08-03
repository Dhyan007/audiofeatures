import librosa
import numpy as np


def extract_spectral_bandwidth(
    window: np.ndarray,
    sr: int
) -> float:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    bandwidth = librosa.feature.spectral_bandwidth(
        y=window,
        sr=sr
    )

    return float(np.mean(bandwidth))