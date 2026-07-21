from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

class AudioIO:
    default_sr=16000
    @staticmethod
    def load_audio(filepath,sample_rate=16000, mono=True):
        waveform, sr=librosa.load(filepath,sr=sample_rate,mono=mono)
        waveform=waveform.astype(np.float32)
        return waveform,sr
    
    @staticmethod
    def save_audio(filepath,waveform,sample_rate=16000,normalize=False,subtype="PCM_16",):
        filepath=Path(filepath)
        filepath.parent.mkdir(parents=True,exist_ok=True)
        #torch tensor to numpy
        if hasattr(waveform,"detach"):
            waveform=waveform.detach().cpu().numpy()
        waveform=np.asarray(waveform,dtype=np.float32)
        #remove singleton channel dimension
        waveform=np.squeeze(waveform)
        if normalize:
            waveform=AudioIO.normalize_peak(waveform)
        sf.write(filepath,waveform,sample_rate,subtype=subtype,)

    @staticmethod
    def normalize_peak(waveform,target_peak=0.99):
        waveform=np.asarray(waveform,dtype=np.float32)
        peak=np.max(np.abs(waveform))
        if peak<1e-8:
            return waveform
        return waveform*(target_peak/peak)
    
    @staticmethod
    def rms(waveform):
        waveform=np.asarray(waveform,dtype=np.float32)
        return np.sqrt(np.mean(waveform ** 2))
    
    @staticmethod
    def peak(waveform):
        waveform=np.asarray(waveform,dtype=np.float32)
        return np.max(np.abs(waveform))
    
    @staticmethod
    def energy(waveform):
        waveform=np.asarray(waveform,dtype=np.float32)
        return np.sum(waveform**2)
    
    @staticmethod
    def get_duration(filepath):
        info=sf.info(filepath)
        return info.frames/info.samplerate
    
    @staticmethod
    def duration_seconds(waveform,sample_rate):
        return len(waveform)/sample_rate
    
    @staticmethod
    def ensure_same_length(*signals):
        #we trim every signal to the shortest length
        min_len=min(len(sig) for sig in signals)
        return tuple(sig[:min_len] for sig in signals)