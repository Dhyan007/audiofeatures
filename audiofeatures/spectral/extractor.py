import numpy as np

from .centroid import extract_spectral_centroid
from .bandwidth import extract_spectral_bandwidth
from .rolloff import extract_spectral_rolloff
from .contrast import extract_spectral_contrast
from .flatness import extract_spectral_flatness


def extract_spectral_features(
    window: np.ndarray,
    sr: int
) -> dict:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    features = {
        "spectral_centroid": extract_spectral_centroid(window, sr),
        "spectral_bandwidth": extract_spectral_bandwidth(window, sr),
        "spectral_rolloff": extract_spectral_rolloff(window, sr),
        "spectral_flatness": extract_spectral_flatness(window),
    }

    features.update(
        extract_spectral_contrast(window, sr)
    )

    return features