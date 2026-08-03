import librosa


def extract_tempo(
    waveform,
    sr: int
) -> float:

    tempo, _ = librosa.beat.beat_track(
        y=waveform,
        sr=sr
    )

    return float(tempo)