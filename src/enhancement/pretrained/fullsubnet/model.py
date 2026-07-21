from pathlib import Path
from .wrapper import FullSubNetWrapper
from .checkpoint import load_checkpoint

class FullSubNetModel:
    def __init__(self,checkpoint_path,device,):
        self.checkpoint_path=Path(checkpoint_path)
        self.device=device
        self.device=device
        self.network=load_checkpoint(self.checkpoint_pagth,device,)
        self.wrapper=FullSubNetWrapper(self.network,device,)
    
    def enhance(self,waveform):
        return self.wrapper.enhance_waveform(waveform)
    