import numpy as np

from .validate import validate_waveform


def crop_audio(
    waveform: np.ndarray,
    sr: int,
    start: float = 0.0,
    duration: float | None = None
):

    validate_waveform(
        waveform,
        sr
    )

    if start < 0:
        raise ValueError("start must be greater than or equal to 0.")

    if duration is not None and duration <= 0:
        raise ValueError("duration must be greater than 0.")

    start_sample = int(start * sr)

    if duration is None:
        end_sample = waveform.shape[-1]
    else:
        end_sample = start_sample + int(duration * sr)

    if start_sample >= waveform.shape[-1]:
        raise ValueError("start exceeds the audio length.")

    end_sample = min(end_sample, waveform.shape[-1])

    if waveform.ndim == 1:
        cropped_waveform = waveform[start_sample:end_sample]
    else:
        cropped_waveform = waveform[:, start_sample:end_sample]

    return cropped_waveform