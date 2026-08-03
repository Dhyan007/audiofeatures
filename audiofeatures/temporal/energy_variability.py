import numpy as np

def extract_energy_variability(window:np.ndarray,
                        frame_length: int=2048,
                        hop_length: int=512)->float:
    
    if window.ndim!=1:
        raise ValueError("window must be mono (1-dimensional)")

    if len(window) == 0:
        raise ValueError("window cannot be empty")
    
    frame_energies=[]

    for start in range(0,len(window)-frame_length+1,hop_length):
        frame=window[start:start+frame_length]
        energy=np.mean(frame**2)
        frame_energies.append(energy)

    if len(frame_energies)<2:
        return 0.0
    
    variability=np.std(frame_energies)
    return float(variability)


