from __future__ import annotations
from pathlib import Path
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (SAMPLE_RATE,N_FFT,HOP_LENGTH,WIN_LENGTH,EPSILON,USE_LOG_MAGNITUDE,PATCH_FRAMES)
from .mask import MaskGenerator

class SpeechEnhancementDataset(Dataset):
    def __init__(self,clean_root,noisy_root,):
        self.clean_root=Path(clean_root)
        self.noisy_root=Path(noisy_root)
        self.pairs=[]
        self._collect_pairs()
        self.mask_generator= MaskGenerator()
        if len(self.pairs)==0:
            raise RuntimeError("No dataset pairs found.")
        
    def _collect_pairs(self):
        noisy_files=sorted(self.noisy_root.rglob("*.wav"))
        for noisy in noisy_files:
            relative=noisy.relative_to(self.noisy_root)
            stem=relative.stem
            clean_name=stem.split("_")[0]+".wav"
            clean=self.clean_root/relative.parent/clean_name
            if clean.exists():
                self.pairs.append((clean,noisy,))
    
    def __len__(self):
        return len(self.pairs)
    
    @staticmethod
    def _load_audio(path):
        audio,_=librosa.load(path,sr=SAMPLE_RATE,mono=True,)
        return audio.astype(np.float32)
    
    @staticmethod
    def _stft(audio):
        spec=librosa.stft(audio,n_fft=N_FFT,hop_length=HOP_LENGTH,win_length=WIN_LENGTH,)
        magnitude=np.abs(spec)
        phase=np.angle(spec)
        if USE_LOG_MAGNITUDE:
            magnitude=np.log1p(magnitude)
        return magnitude,phase
    
    @staticmethod
    def _extract_patch(clean_mag,noisy_mag,patch_frames):
        frames=clean_mag.shape[1]
        if frames >= patch_frames:
            start=np.random.randint(0,frames-patch_frames+1)
            clean_mag=clean_mag[:,start:start+patch_frames]
            noisy_mag=noisy_mag[:,start:start+patch_frames]
        else:
            pad=patch_frames-frames
            clean_mag=np.pad(clean_mag,((0,0),(0,pad)),mode="constant",)
            noisy_mag=np.pad(noisy_mag,((0,0),(0,pad)),mode="constant",)
        return clean_mag,noisy_mag
    
    def __getitem__(self,index):
        clean_path, noisy_path = self.pairs[index]
        clean_audio=self._load_audio(clean_path)
        noisy_audio=self._load_audio(noisy_path)
        clean_mag, _=self._stft(clean_audio)
        noisy_mag,noisy_phase=self._stft(noisy_audio)
        clean_mag,noisy_mag=self._extract_patch(clean_mag,noisy_mag,PATCH_FRAMES,)
        mask=self.mask_generator(clean_mag,noisy_mag,)
        clean_mag=torch.from_numpy(clean_mag).float().unsqueeze(0)
        noisy_mag=torch.from_numpy(noisy_mag).float().unsqueeze(0)
        mask=torch.from_numpy(mask).float().unsqueeze(0)
        noisy_phase=torch.from_numpy(noisy_phase).float()

        metadata={"clean_path":str(clean_path),"noisy_path":str(noisy_path),"filename":clean_path.stem,}
        return {"noisy":noisy_mag, "clean":clean_mag, "mask":mask,"metadata":metadata,}