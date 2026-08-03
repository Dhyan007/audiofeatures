import numpy as np

def extract_crest_factor(window:np.ndarray)->float:
    if window.ndim!=1:
        raise ValueError("window must be mono (1-dimensional)")
    
    if len(window) == 0:
        raise ValueError("window cannot be empty")
    
    rms=np.sqrt(np.mean(window**2))

    if rms ==0:
        return 0.0
    
    peak=np.max(np.abs(window))
    crest_factor=peak/rms

    return float(crest_factor)

