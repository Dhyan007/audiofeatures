from ..exceptions import InvalidAudioError
from ..exceptions import EmptyAudioError


def validate_window(window):

    if window.ndim != 1:
        raise InvalidAudioError(
            "window must be mono (1-dimensional)"
        )

    if len(window) == 0:
        raise EmptyAudioError(
            "window cannot be empty"
        )