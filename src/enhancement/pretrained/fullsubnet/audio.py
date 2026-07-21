"""
this file will contain audio processing utilities used by the
FullSubNet wrapper.

Responsibilities:
1.Waveform->STFT
2.STFT->Magnitude/Phase
3.Apply complex ratio mask
4.ISTFT reconstruction
"""

import soundfile as sf
import numpy as np
import torch
from third_party.FullSubNet.audio_zen.acoustics.feature import stft,istft
from third_party.FullSubNet.audio_zen.acoustics.mask import decompress_cIRM, complex_mul
from .config import SAMPLE_RATE,N_FFT,WIN_LENGTH,HOP_LENGTH

class FullSubNetAudioProcessor:
    def __init__(self,sample_rate=SAMPLE_RATE,n_fft=N_FFT,win_length=WIN_LENGTH,hop_length=HOP_LENGTH,):
        self.sample_rate=sample_rate
        self.n_fft=n_fft
        self.win_length=win_length
        self.hop_length=hop_length
    
    def stft(self,waveform):
        if waveform.ndim==1:
            waveform=waveform.unsqueeze(0)
        mag,phase,real,imag=stft(waveform,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.win_length,)
        return mag,phase,real,imag
    
    #applying cIRM mask
    def apply_mask(self,real,imag,predicted_mask,):
        mask=decompress_cIRM(predicted_mask)
        mask_real=mask[:,0]
        mask_imag=mask[:,1]
        enhanced_real,enhanced_imag=complex_mul(real,imag,mask_real,mask_imag,)
        return enhanced_real, enhanced_imag
    
    def istft(self,enhanced_real,enhanced_imag,length,):
        waveform=istft((enhanced_real,enhanced_imag),n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.win_length,length=length,input_type="real_imag",)
        return waveform
    
    #complete enhancement

    def enhance(self,waveform,model,):
        original_length=waveform.shape[-1]
        mag,phase,real,imag=self.stft(waveform)
        model_input=mag.unsqueeze(1)
        predicted_mask=model(model_input)
        enhanced_real,enhanced_imag=self.apply_mask(real,imag,predicted_mask,)
        enhanced=self.istft(enhanced_real,enhanced_imag,original_length,)
        return enhanced.squeeze(0)