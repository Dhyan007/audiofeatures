import numpy as np

from ..preprocessing import validate_waveform


def split_audio(
    waveform: np.ndarray,
    sr: int,
    window_size: float = 0.5
):

    validate_waveform(
        waveform,
        sr
    )

    window_samples = int(window_size * sr)

    windows = []

    for start in range(
        0,
        len(waveform),
        window_samples
    ):

        end = start + window_samples

        window = waveform[start:end]

        if len(window) == window_samples:
            windows.append(window)

    return windows