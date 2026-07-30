from __future__ import annotations
from pathlib import Path
import numpy as np
from src.enhancement.base import BaseEnhancer
from .config import (CHECKPOINT_ROOT,CHECKPOINT_NAME,)
from .inference import CNNInference

class CNNEnhancer(BaseEnhancer):
    def __init__(self,device="cpu",noise_type="white",):
        super().__init__(device)
        checkpoint=(CHECKPOINT_ROOT/noise_type/CHECKPOINT_NAME)
        if not checkpoint.exists():
            raise FileNotFoundError(f"\nCNN checkpoint not found:\n\n{checkpoint}")
        self.inference=CNNInference(checkpoint=checkpoint,)
    
    @property
    def input_type(self):
        return "waveform"
    
    def load_weights(self,checkpoint,):
        self.inference=CNNInference(checkpoint,)

    def enhance(self,waveform,):
        waveform=np.asarray(waveform,dtype=np.float32,)
        result=self.inference.enhance(waveform,)
        return result["waveform"]
    
    def enhance_with_debug(self,waveform,):
        waveform=np.asarray(waveform,dtype=np.float32,)
        return self.inference.enhance(waveform,)
    
    def export_results(self,filename,result,clean_mag=None,):
        self.inference.export_results(filename=filename,clean_mag=clean_mag,noisy_mag=result["noisy_mag"],enhanced_mag=result["enhanced_mag"],predicted_mask=result["mask"],method="cnn",)
        