import numpy as np

from .validate import validate_waveform


def normalize_audio(
    waveform: np.ndarray,
    sr: int
):

    validate_waveform(
        waveform,
        sr
    )

    max_amplitude = np.max(np.abs(waveform))

    if max_amplitude == 0:
        return waveform

    normalized_waveform = waveform / max_amplitude

    return normalized_waveform