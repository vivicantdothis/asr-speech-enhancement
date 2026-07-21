from __future__ import annotations
import numpy as np

class WienerFilter:
    """
    we use the classical wiener filter functiosn where
    gain is given by g=snr/(snr+1) where snr = signalpsd/noisepsd
    and signal psd is estimated from max(noisypsd-noisepsd,0)
    """
    def __init__(self,eps:float=1e-12,gain_floor:float=0.05,):
        self.eps=eps
        self.gain_floor=gain_floor

    def estimate_signal_psd(self,noisy_psd:np.ndarray,noise_psd:np.ndarray,)->np.ndarray:
        noise_psd=noise_psd[:,None]
        signal_psd=np.maximum(noisy_psd-noise_psd,0.0,)
        return signal_psd
    
    def compute_gain(self,noisy_psd:np.ndarray,noise_psd:np.ndarray,)->np.ndarray:
        signal_psd=self.estimate_signal_psd(noisy_psd,noise_psd,)
        noise=noise_psd[:,None]
        posterior_snr=signal_psd/(noise+self.eps)
        gain=posterior_snr/(posterior_snr+1.0)
        gain=np.clip(gain,self.gain_floor,1.0,)
        return gain
    
    def apply(self,noisy_complex:np.ndarray,noise_psd:np.ndarray,):
        magnitude=np.abs(noisy_complex)
        noisy_psd=magnitude**2
        gain=self.compute_gain(noisy_psd,noise_psd,)
        enhanced=gain*noisy_complex
        return enhanced,gain
    