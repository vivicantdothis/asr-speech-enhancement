from __future__ import annotations
import numpy as np
import torch

from src.enhancement.base import BaseEnhancer
from .wrapper import FullSubNetWrapper
from .checkpoint import load_pretrained_model

class FullSubNetEnhancer(BaseEnhancer):
    def __init__(self,device="cpu"):
        super().__init__(device)
        self.wrapper=FullSubNetWrapper(device=device)
        self.model=self.wrapper.model
        self.model.eval()
    
    @property
    def input_type(self):
        return "waveform"
    
    def load_weights(self,checkpoint=None):
        if checkpoint is None:
            return
        self.model=load_pretrained_model(device=self.device,checkpoint_path=checkpoint,)
        self.wrapper.model=self.model
    @torch.no_grad()
    def enhance(self,waveform):
        enhanced=self.wrapper.enhance(waveform)
        if isinstance(enhanced,torch.Tensor):
            enhanced=enhanced.cpu().numpy()
        enhanced=np.asarray(enhanced,dtype=np.float32,)
        return np.squeeze(enhanced)
    
    def eval(self):
        self.model.eval()
        return self