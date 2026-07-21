"""
this module provided reusable stft utilities and a consistent
stft implementation used by all classical enhancement algorithms
in the project.

this implementation wraps librosa so every enhancement method
uses identical fft parameters."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import librosa

@dataclass
class STFTConfig:
    sample_rate: int=16000
    n_fft: int=512
    win_length: int=512
    hop_length: int=256
    window: str="hann"
    center: bool=True
    pad_mode: str="reflect"
class STFTProcessor:
    def __init__(self,config: STFTConfig | None=None):
        self.config=config or STFTConfig()
    
    def stft(self,waveform:np.ndarray)->np.ndarray:
        """
        compute complex-valued stft (forward stft).
        the parameters are waveform: np.ndarray and it returns
        complex spectrogram of shape (frequency_bins,time_frames)"""

        waveform=np.asarray(waveform,dtype=np.float32)
        spectrum=librosa.stft(waveform,n_fft=self.config.n_fft,hop_length=self.config.hop_length,win_length=self.config.win_length,window=self.config.window,center=self.config.center,pad_mode=self.config.pad_mode,)
        return spectrum
    
    def forward(self,waveform:np.ndarray)->np.ndarray:
        return self.stft(waveform)
    
    def inverse(self,spectrum:np.ndarray,length: int | None=None,)->np.ndarray:
        return self.istft(spectrum,length=length,)
    #inverse stft
    def istft(self,spectrum:np.ndarray,length:int|None=None,)->np.ndarray:
        #reconstructs the waveform from complex stft
        waveform=librosa.istft(spectrum,hop_length=self.config.hop_length,win_length=self.config.win_length,window=self.config.window,center=self.config.center,length=length,)
        return waveform.astype(np.float32)
    
    @staticmethod
    def phase(spectrum:np.ndarray,)->np.ndarray:
        #phase spectrogram
        return np.angle(spectrum)
    
    @staticmethod
    def magnitude(spectrum:np.ndarray)->np.ndarray:
        return np.abs(spectrum)
    
    @staticmethod
    #power spectrum
    def power(spectrum:np.ndarray,)->np.ndarray:
        mag=np.abs(spectrum)
        return mag**2
    
    @staticmethod
    def combine(magnitude:np.ndarray,phase:np.ndarray,)->np.ndarray:
        return magnitude * np.exp(1j*phase)
    
    #convenience functions
    def decompose(self,waveform:np.ndarray,):
        spectrum=self.stft(waveform)
        magnitude=self.magnitude(spectrum)
        phase=self.phase(spectrum)
        power=magnitude**2
        return spectrum,magnitude,phase,power