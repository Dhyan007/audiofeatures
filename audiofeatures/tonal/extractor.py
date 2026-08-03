import numpy as np

from .chroma import extract_chroma
from .tonnetz import extract_tonnetz


def extract_tonal_features(
    window: np.ndarray,
    sr: int
) -> dict:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    features = {}

    features.update(
        extract_chroma(window, sr)
    )

    features.update(
        extract_tonnetz(window, sr)
    )

    return features