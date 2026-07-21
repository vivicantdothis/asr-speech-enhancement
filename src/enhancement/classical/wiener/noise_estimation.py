from __future__ import annotations
import numpy as np

class NoiseEstimator:
    #estimates the noise power spectrum.
    def __init__(self,num_noise_frames:int=6,smoothing:float=0.98,):
        self.num_noise_frames=num_noise_frames
        self.smoothing=smoothing
        self.noise_psd=None
    
    def estimate(self,magnitude:np.ndarray)->np.ndarray:
        if magnitude.ndim!=2:
            raise ValueError("Magnitude spectrogram must have shape" "(freq_bins,frames)")
        n=min(self.num_noise_frames,magnitude.shape[1])
        noise_mag=magnitude[:,:n]
        self.noise_psd=np.mean(noise_mag**2,axis=1)
        return self.noise_psd
    
    def update(self,current_magnitude:np.ndarray,)->np.ndarray:
        if self.noise_psd is None:
            raise RuntimeError("Noise must be estimated before update().")
        current_psd=current_magnitude**2
        self.noise_psd=(self.smoothing*self.noise_psd+(1.0-self.smoothing)*current_psd)
        return self.noise_psd
    
    def reset(self):
        self.noise_psd=None