import librosa
import pandas as pd

from .preprocessing import preprocess_audio

from .temporal import (
    split_audio,
    extract_temporal_features
)

from .spectral import (
    extract_spectral_features
)

from .cepstral import (
    extract_cepstral_features
)

from .tonal import (
    extract_tonal_features
)

from .pitch import (
    extract_pitch_features
)

from .rhythm import (
    extract_rhythm_features
)

from .logger import logger


def extract_features(
    audio_path: str,
    target_sr: int = 22050,
    window_size: float = 0.5
) -> pd.DataFrame:

    logger.info("Loading audio")

    waveform, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    logger.info("Applying preprocessing")

    waveform, sr = preprocess_audio(
        waveform,
        sr,
        target_sr=target_sr
    )

    logger.info("Splitting audio into windows")

    windows = split_audio(
        waveform,
        sr,
        window_size
    )

    all_features = []

    logger.info("Extracting features")

    for index, window in enumerate(windows):

        features = {
            "window": index,
            "start_time": index * window_size,
            "end_time": (index + 1) * window_size
        }

        features.update(
            extract_temporal_features(window)
        )

        features.update(
            extract_spectral_features(
                window,
                sr
            )
        )

        features.update(
            extract_cepstral_features(
                window,
                sr
            )
        )

        features.update(
            extract_tonal_features(
                window,
                sr
            )
        )

        features.update(
            extract_pitch_features(
                window,
                sr
            )
        )

        features.update(
            extract_rhythm_features(
                window,
                sr
            )
        )

        all_features.append(features)

    logger.info("Feature extraction completed")

    return pd.DataFrame(all_features)