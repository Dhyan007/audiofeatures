import librosa
import numpy as np


def extract_delta_mfcc(
    window: np.ndarray,
    sr: int,
    n_mfcc: int = 20
) -> dict:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    mfcc = librosa.feature.mfcc(
        y=window,
        sr=sr,
        n_mfcc=n_mfcc
    )

    delta = librosa.feature.delta(mfcc)

    features = {}

    for i in range(n_mfcc):

        features[f"delta_mfcc_{i+1}_mean"] = float(np.mean(delta[i]))
        features[f"delta_mfcc_{i+1}_std"] = float(np.std(delta[i]))

    return features