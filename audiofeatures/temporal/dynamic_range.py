import numpy as np

def extract_dynamic_range(window:np.ndarray)->float:
    if window.ndim != 1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")
    
    dynamic_range=np.max(window)-np.min(window)
    return float(dynamic_range)