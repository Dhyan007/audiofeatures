import numpy as np

from .mfcc import extract_mfcc
from .delta import extract_delta_mfcc
from .delta_delta import extract_delta_delta_mfcc


def extract_cepstral_features(
    window: np.ndarray,
    sr: int
) -> dict:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    features = {}

    features.update(
        extract_mfcc(window, sr)
    )

    features.update(
        extract_delta_mfcc(window, sr)
    )

    features.update(
        extract_delta_delta_mfcc(window, sr)
    )

    return features