import librosa
import numpy as np

from .validate import validate_waveform


def resample_audio(
    waveform: np.ndarray,
    original_sr: int,
    target_sr: int
):

    validate_waveform(
        waveform,
        original_sr
    )

    if target_sr <= 0:
        raise ValueError("target_sr must be greater than 0.")

    if original_sr == target_sr:
        return waveform, original_sr

    resampled_waveform = librosa.resample(
        y=waveform,
        orig_sr=original_sr,
        target_sr=target_sr
    )

    return resampled_waveform, target_sr