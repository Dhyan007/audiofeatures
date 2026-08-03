import numpy as np

from .onset import extract_onset_strength


def extract_rhythm_features(
    window: np.ndarray,
    sr: int
) -> dict:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    return extract_onset_strength(window, sr)