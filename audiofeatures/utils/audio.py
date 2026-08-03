import librosa

from ..config import DEFAULT_SAMPLE_RATE


def load_audio(
    audio_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True
):

    waveform, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=mono
    )

    return waveform, sr