import numpy as np


def extract_amplitude_variability(window:np.ndarray)->float:
    if window.ndim!= 1:
        raise ValueError("window must be mono (1-dimensional)")
    if len(window) ==0:
        raise ValueError("window cannot be empty")
    amplitude=np.abs(window)
    variability=np.std(amplitude)
    return float(variability)