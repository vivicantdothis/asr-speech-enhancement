import numpy as np

def snr(clean,enhanced):
    clean=np.asarray(clean)
    enhacned=np.asarray(enhanced)
    noise=clean-enhanced
    return 10 * np.log10(np.sum(clean**2)/(np.sum(noise**2)+1e-8))
