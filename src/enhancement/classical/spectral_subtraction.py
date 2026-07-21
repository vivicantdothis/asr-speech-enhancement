from __future__ import annotations
import numpy as np
import librosa

class SpectralSubtraction:
    """
    Classical Spectral Subtraction: uses magnitude
    subtraction while preserving the noisy phase.
    """
    def __init__(self,sample_rate=16000,n_fft=512,hop_length=128,win_length=512,noise_frames=20,subtraction_factor=1.0,spectral_floor=0.02,):
        self.sample_rate=sample_rate
        self.n_fft=n_fft
        self.hop=hop_length
        self.win=win_length
        self.noise_frames=noise_frames
        self.alpha=subtraction_factor
        self.floor=spectral_floor
    def estimate_noise(self,magnitude):
        """
        estimates the average noise spectrum
        using the first few frames.
        """
        return np.mean(magnitude[:,:self.noise_frames],axis=1,keepdims=True,)
    def enhance(self,waveform):
        waveform=np.asarray(waveform,dtype=np.float32,)
        stft=librosa.stft(waveform,n_fft=self.n_fft,hop_length=self.hop,win_length=self.win,window="hann",)
        magnitude=np.abs(stft)
        phase=np.angle(stft)
        noise=self.estimate_noise(magnitude)
        enhanced_mag=magnitude-self.alpha*noise
        floor=self.floor*magnitude
        enhanced_mag=np.maximum(enhanced_mag,floor,)
        enhanced_stft=enhanced_mag*np.exp(1j*phase)
        enhanced=librosa.istft(enhanced_stft,hop_length=self.hop,win_length=self.win,window="hann",length=len(waveform),)
        enhanced=enhanced.astype(np.float32)
        peak=np.max(np.abs(enhanced))
        if peak>1:
            enhanced/=peak
        return enhanced