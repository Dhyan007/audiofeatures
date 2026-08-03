import numpy as np

def extract_zcr(window:np.ndarray)->float:

    if window.ndim !=1:
        raise ValueError("window must be mono (1-dimensional)")
    
    if len(window)==0:
        raise ValueError("window cannot be empty")
    
    zero_crossing=np.sum(
        np.signbit(window[:-1])!=np.signbit(window[1:])
    )

    zcr=zero_crossing/len(window)-1

    return float(zcr)