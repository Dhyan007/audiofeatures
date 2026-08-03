import librosa
import numpy as np


def extract_pitch(
    window: np.ndarray,
    sr: int
) -> dict:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    pitch = librosa.yin(
        window,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    pitch = pitch[~np.isnan(pitch)]

    if len(pitch) == 0:
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_min": 0.0,
            "pitch_max": 0.0
        }

    return {
        "pitch_mean": float(np.mean(pitch)),
        "pitch_std": float(np.std(pitch)),
        "pitch_min": float(np.min(pitch)),
        "pitch_max": float(np.max(pitch))
    }