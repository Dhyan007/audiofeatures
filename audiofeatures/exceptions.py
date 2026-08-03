class AudioFeaturesError(Exception):
    pass


class InvalidAudioError(AudioFeaturesError):
    pass


class InvalidSampleRateError(AudioFeaturesError):
    pass


class EmptyAudioError(AudioFeaturesError):
    pass