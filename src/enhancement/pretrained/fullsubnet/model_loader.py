"""
this file serves as the centralized fullsubnet
model loader. this module ensures that the pretrained
checkpoint is loaded only once.
every wrapper/enhancer imports this module instead of 
loading checkpoints repeatedly.
"""
from __future__ import annotations
import torch
from .checkpoint import load_pretrained_model
from .config import DEFAULT_DEVICE

class FullSubNetModelLoader:
    _model=None
    def __init__(self,device=DEFAULT_DEVICE,):
        if torch.cuda.is_available():
            self.device=torch.device(device)
        else:
            self.device=torch.device("cpu")
        if FullSubNetModelLoader._model is None:
            FullSubNetModelLoader._model = load_pretrained_model(device=self.device)
    
    @property
    def model(self):
        return FullSubNetModelLoader._model
    def eval(self):
        self.model.eval()
        return self.model

    def to(self,device):
        self.device=torch.device(device)
        self.model.to(self.device)
        return self.model
    
    @property
    def dtype(self):
        return next(self.model.parameters()).dtype
    
    @property
    def training(self):
        return self.model.training
    
#convenience function
_loader=None
def get_model(device=DEFAULT_DEVICE):
    global _loader
    if _loader is None:
        _loader=FullSubNetModelLoader(device)
    return _loader.model