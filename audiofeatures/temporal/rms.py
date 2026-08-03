import numpy as np

def extract_rms(window: np.ndarray)-> float:
    if window.ndim !=1:
        raise ValueError("window must be mono (1-dimensional)")
    
    if len(window) ==0:
        raise ValueError("window cannot be empty")
    
    rms=np.sqrt(np.mean(window**2))
    return float(rms)