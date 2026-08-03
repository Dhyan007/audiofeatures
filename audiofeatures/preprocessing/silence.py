import librosa
import numpy as np

from .validate import validate_waveform


def remove_silence(
    waveform: np.ndarray,
    sr: int,
    top_db: int = 20
):

    validate_waveform(
        waveform,
        sr
    )

    if waveform.ndim == 1:

        trimmed_waveform, _ = librosa.effects.trim(
            waveform,
            top_db=top_db
        )

    else:

        trimmed_waveform = []

        for channel in waveform:

            trimmed_channel, _ = librosa.effects.trim(
                channel,
                top_db=top_db
            )

            trimmed_waveform.append(trimmed_channel)

        min_length = min(len(channel) for channel in trimmed_waveform)

        trimmed_waveform = np.array([
            channel[:min_length]
            for channel in trimmed_waveform
        ])

    return trimmed_waveform