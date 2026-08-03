import numpy as np

def extract_energy(window: np.ndarray)->float:
    if window.ndim !=1:
        raise ValueError("window must be mono (1-dimensional)")
    
    if len(window)==0:
        raise ValueError("window cannot be empty")
    
    energy =np.sum(window**2)
    return float(energy)