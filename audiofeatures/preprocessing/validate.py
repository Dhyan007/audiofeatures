import numpy as np


def validate_waveform(
    waveform: np.ndarray,
    sr: int
):

    if not isinstance(waveform, np.ndarray):
        raise TypeError("waveform must be a NumPy array.")

    if waveform.ndim not in (1, 2):
        raise ValueError(
            "waveform must be mono (1D) or stereo (2D)."
        )

    if waveform.size == 0:
        raise ValueError("waveform cannot be empty.")

    if np.isnan(waveform).any():
        raise ValueError("waveform contains NaN values.")

    if np.isinf(waveform).any():
        raise ValueError("waveform contains infinite values.")

    if sr <= 0:
        raise ValueError("sample rate must be greater than 0.")

    return True