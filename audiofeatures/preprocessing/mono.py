import numpy as np

from .validate import validate_waveform


def convert_to_mono(
    waveform: np.ndarray,
    sr: int
):

    validate_waveform(
        waveform,
        sr
    )

    if waveform.ndim == 1:
        return waveform

    mono_waveform = np.mean(
        waveform,
        axis=0
    )

    return mono_waveform