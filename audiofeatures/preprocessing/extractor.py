from .validate import validate_waveform
from .mono import convert_to_mono
from .crop import crop_audio
from .resample import resample_audio
from .normalize import normalize_audio
from .silence import remove_silence


def preprocess_audio(
    waveform,
    sr,
    target_sr=22050,
    mono=True,
    normalize=True,
    trim_silence=True,
    start=None,
    duration=None
):

    validate_waveform(
        waveform,
        sr
    )

    if mono:
        waveform = convert_to_mono(
            waveform,
            sr
        )

    if start is not None or duration is not None:
        waveform = crop_audio(
            waveform,
            sr,
            start=0.0 if start is None else start,
            duration=duration
        )

    waveform, sr = resample_audio(
        waveform,
        sr,
        target_sr
    )

    if normalize:
        waveform = normalize_audio(
            waveform,
            sr
        )

    if trim_silence:
        waveform = remove_silence(
            waveform,
            sr
        )

    return waveform, sr