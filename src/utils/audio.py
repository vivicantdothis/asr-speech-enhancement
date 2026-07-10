from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

class AudioIO:
    @staticmethod
    def load_audio(filepath,sample_rate=16000, mono=True):
        waveform, sr=librosa.load(filepath,sr=sample_rate,mono=mono)
        waveform=waveform.astype(np.float32)
        return waveform,sr
    @staticmethod
    def save_audio(filepath,waveform,sample_rate=16000):
        filepath=Path(filepath)
        filepath.parent.mkdir(parents=True,exist_ok=True)
        if hasattr(waveform,"detach"):
            waveform=waveform.detach().cpu().numpy()
        waveform=np.asarray(waveform,dtype=np.float32)
        waveform=AudioIO.normalize_peak(waveform)
        sf.write(filepath,waveform,sample_rate)
    @staticmethod
    def normalize_peak(waveform,target_peak=0.99):
        peak=np.max(np.abs(waveform))
        if peak<1e-8:
            return waveform
        return target_peak * waveform/peak
    
    @staticmethod
    def rms(waveform):
        return np.sqrt(np.mean(waveform ** 2))
    
    @staticmethod
    def get_duration(filepath):
        import soundfile as sf
        info=sf.info(filepath)
        return info.frames/info.samplerate
    
    @staticmethod
    def duration_seconds(waveform,sample_rate):
        return len(waveform)/sample_rate
    
    @staticmethod
    def ensure_same_length(clean,noisy):
        min_len=min(len(clean), len(noisy))
        return(clean[:min_len],noisy[:min_len])