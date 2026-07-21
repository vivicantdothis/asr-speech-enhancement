import torch
import numpy as np

from df.enhance import enhance, init_df
from src.enhancement.base import BaseEnhancer

class DeepFilterNetEnhancer(BaseEnhancer):
    def __init__(self,device="cpu"):
        super().__init__(device)
        self.model, self.df_state, _=init_df()
        self.model.eval()

    @property
    def input_type(self):
        return "waveform"
    
    def load_weights(self, checkpoint=None):
        pass

    @torch.no_grad()
    def enhance(self,waveform):
        if isinstance(waveform,np.ndarray):
            waveform=torch.from_numpy(waveform)
        waveform=waveform.float()
        if waveform.ndim==1:
            waveform=waveform.unsqueeze(0)
        #debug
        print(f"Input shape:{tuple(waveform.shape)}")
        print(f"Input peak:{waveform.abs().max().item():.4f}")
        print(f"Input RMS:{torch.sqrt(torch.mean(waveform**2)).item():.4f}")
        #enhancement
        enhanced=enhance(self.model,self.df_state,waveform,pad=True,)
        #debug
        print(f"Output shape:{tuple(enhanced.shape)}")
        print(f"Output peak:{enhanced.abs().max().item():.4f}")
        print(f"Output RMS: {torch.sqrt(torch.mean(enhanced**2)).item():.4f}")
        enhanced=enhanced.squeeze(0)
        return enhanced.cpu().numpy().astype(np.float32)