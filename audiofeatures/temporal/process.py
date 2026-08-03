import numpy as np

from .windowing import split_audio
from .extractor import extract_temporal_features


def extract_temporal_over_time(
    waveform: np.ndarray,
    sr: int,
    window_size: float = 0.5
) -> list[dict]:

    windows = split_audio(
        waveform=waveform,
        sr=sr,
        window_size=window_size
    )

    results = []

    for index, window in enumerate(windows):

        features = extract_temporal_features(window)

        features["start_time"] = index * window_size
        features["end_time"] = (index + 1) * window_size

        results.append(features)

    return results