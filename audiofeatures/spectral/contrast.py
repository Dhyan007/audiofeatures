import librosa
import numpy as np


def extract_spectral_contrast(
    window: np.ndarray,
    sr: int
) -> dict:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    contrast = librosa.feature.spectral_contrast(
        y=window,
        sr=sr
    )

    features = {}

    for i in range(contrast.shape[0]):
        features[f"spectral_contrast_band_{i+1}"] = float(
            np.mean(contrast[i])
        )

    return features