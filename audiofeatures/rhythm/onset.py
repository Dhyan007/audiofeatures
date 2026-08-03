import librosa
import numpy as np


def extract_onset_strength(
    window: np.ndarray,
    sr: int
) -> dict:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    onset = librosa.onset.onset_strength(
        y=window,
        sr=sr
    )

    return {
        "onset_strength_mean": float(np.mean(onset)),
        "onset_strength_std": float(np.std(onset)),
        "onset_strength_max": float(np.max(onset))
    }