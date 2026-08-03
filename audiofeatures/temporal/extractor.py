import numpy as np

from .rms import extract_rms
from .zcr import extract_zcr
from .energy import extract_energy
from .peak_amplitude import extract_peak_amplitude
from .crest_factor import extract_crest_factor
from .dynamic_range import extract_dynamic_range
from .energy_variability import extract_energy_variability
from .amplitude_variability import extract_amplitude_variability


def extract_temporal_features(window: np.ndarray) -> dict:

    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")

    return {
        "rms": extract_rms(window),
        "zcr": extract_zcr(window),
        "energy": extract_energy(window),
        "peak_amplitude": extract_peak_amplitude(window),
        "crest_factor": extract_crest_factor(window),
        "dynamic_range": extract_dynamic_range(window),
        "energy_variability": extract_energy_variability(window),
        "amplitude_variability": extract_amplitude_variability(window),
    }