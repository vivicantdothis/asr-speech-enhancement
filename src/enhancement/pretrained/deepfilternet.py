import torch
import numpy as np

from df.enhance import enhance, init_df
from src.enhancement.base import BaseEnhancer

class DeepFilterNetEnhancer(BaseEnhancer):
    def __init__(self,device="cpu"):
        super().__init__(device)
        self.model, self.df_state, _=init_df()
    @property
    def input_type(self):
        return "waveform"
    def load_weights(self, checkpoint=None):
        pass
    def enhance(self,waveform):
        if isinstance(waveform,np.ndarray):
            waveform=torch.from_numpy(waveform)
        waveform=waveform.float()
        if waveform.ndim==1:
            waveform=waveform.unsqueeze(0)
        enhanced=enhance(self.model,self.df_state,waveform,)
        enhanced=enhanced.squeeze(0)
        return enhanced.cpu().numpy()