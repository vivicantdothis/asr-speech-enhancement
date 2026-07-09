import librosa 
import numpy as np

class SpectrogramReconstructor:
    @staticmethod
    def griffin_lim(magnitude,config):
        waveform=librosa.griffinlim(magnitude,hop_length=config.hop_length,win_length=config.win_length,n_fft=config.n_fft,)
        return waveform.astype(np.float32)