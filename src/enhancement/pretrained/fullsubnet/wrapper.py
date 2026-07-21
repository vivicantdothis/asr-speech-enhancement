from __future__ import annotations
import numpy as np
import torch
from .audio import FullSubNetAudioProcessor
from .model_loader import get_model
from .config import DEFAULT_DEVICE

class FullSubNetWrapper:
    def __init__(self,device=DEFAULT_DEVICE):
        if torch.cuda.is_available():
            self.device=torch.device(device)
        else:
            self.device=torch.device("cpu")
        self.model=get_model(device=self.device,)
        self.processor=FullSubNetAudioProcessor()
    def _prepare_input(self,waveform):
        if isinstance(waveform,np.ndarray):
            waveform=torch.from_numpy(waveform)
        if not isinstance(waveform,torch.Tensor):
            raise TypeError(f"Unsupported waveform type:{type(waveform)}")
        waveform=waveform.float()
        if waveform.ndim==2:
            if waveform.shape[0]==1:
                waveform=waveform.squeeze(0)
            else:
                raise ValueError("Only mono audio is supported.")
        return waveform.to(self.device)
        
    @torch.no_grad()
    def enhance(self,waveform):
        waveform=self._prepare_input(waveform)
        enhanced=self.processor.enhance(waveform,self.model,)
        if torch.is_tensor(enhanced):
            enhanced=enhanced.detach().cpu().numpy()
        return enhanced.astype(np.float32)
    
    #convenience
    def eval(self):
        self.model.eval()
    def train(self):
        self.model.train()
    def to(self,device):
        self.device=torch.device(device)
        self.model.to(self.device)
        return self