from __future__ import annotations
import numpy as np
from src.enhancement.base import BaseEnhancer
from .stft import STFTProcessor, STFTConfig
from .noise_estimation import NoiseEstimator
from .filter import WienerFilter


class WienerEnhancer(BaseEnhancer):
    def __init__(self,device="cpu",sample_rate=16000,n_fft=512,hop_length=256,win_length=512,noise_frames=10,):
        super().__init__(device)
        config=STFTConfig(sample_rate=sample_rate,n_fft=n_fft,hop_length=hop_length,win_length=win_length,)
        self.stft=STFTProcessor(config)
        self.noise_estimator=NoiseEstimator(num_noise_frames=noise_frames,)
        self.wiener_filter=WienerFilter()

    @property
    def input_type(self):
        return "waveform"
    
    def load_weights(self,checkpoint=None):
        return 
    
    def enhance(self,waveform):
        waveform=np.asarray(waveform,dtype=np.float32)
        spectrum=self.stft.forward(waveform)
        noise_psd=self.noise_estimator.estimate(spectrum)
        enhanced_spec,gain=self.wiener_filter.apply(spectrum,noise_psd,)
        enhanced=self.stft.inverse(enhanced_spec,length=len(waveform),)
        return enhanced.astype(np.float32)
    
    def enhanced_with_debug(self,waveform):
        waveform=np.asarray(waveform,dtype=np.float32)
        noisy_spec=self.stft.forward(waveform)
        noise_psd=self.noise_estimator.estimate(noisy_spec)
        enhanced_spec,gain=self.wiener_filter.apply(noisy_spec,noise_psd,)
        enhanced=self.stft.inverse(enhanced_spec,length=len(waveform),)
        return {"waveform":waveform,"noisy_spec":noisy_spec,"noise_psd":noise_psd,"gain":gain,"enhanced_spec":enhanced_spec,"enhanced":enhanced.astype(np.float32),}