"""
wrapper file that allows the classical spectral subtraction
algorithm to behave exactly like the other pretrained/classical 
enhancers in the project.

it follows the BaseEnhancer API so that the factory and enhancement
pipeline class require no modifications."""

from __future__ import annotations
import numpy as np
import torch

from src.enhancement.base import BaseEnhancer
from .spectral_subtraction import SpectralSubtraction

class SpectralSubtractionEnhancer(BaseEnhancer):
    def __init__(self,device="cpu",sample_rate=16000,n_fft=512,hop_length=128,win_length=512,noise_frames=20,subtraction_factor=1.0,spectral_floor=0.02,):
        super().__init__(device)
        self.processor=SpectralSubtraction(sample_rate=sample_rate,n_fft=n_fft,hop_length=hop_length,win_length=win_length,noise_frames=noise_frames,subtraction_factor=subtraction_factor,spectral_floor=spectral_floor,)
    
    @property
    def input_type(self):
        #loads raw audio (EnhancementPipeline)
        return "waveform"
    
    def load_weights(self,checkpoint=None):
        #classical dsp algorithm without pretrained weights
        return 
    
    @torch.no_grad()
    def enhance(self,waveform):
        if isinstance(waveform,torch.Tensor):
            waveform=waveform.detach().cpu().numpy()
        waveform=np.asarray(waveform,dtype=np.float32,)
        if waveform.ndim>1:
            waveform=np.squeeze(waveform)
        enhanced=self.processor.enhance(waveform)
        enhanced=np.asarray(enhanced,dtype=np.float32,)
        return enhanced
    def eval(self):
        return self
    