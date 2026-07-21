import numpy as np

def snr(clean,enhanced):
    clean=np.asarray(clean,dtype=np.float32)
    enhanced=np.asarray(enhanced,dtype=np.float32)
    noise=clean-enhanced
    signal_power=np.sum(clean**2)
    noise_power=np.sum(noise**2)
    return 10 * np.log10(signal_power/noise_power+1e-8)
