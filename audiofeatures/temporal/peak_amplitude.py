import numpy as np

def extract_peak_amplitude(window:np.ndarray)->float:
    if window.ndim!=1:
        raise ValueError("window must be mono (1-dimensional)")
    
    if len(window)==0:
        raise ValueError("window cannot be empty")
    
    peak_amplitude=np.max(np.abs(window))
    return float(peak_amplitude)