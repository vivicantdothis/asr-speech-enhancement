from pathlib import Path
import sys
from .config import FULLSUBNET_ROOT
sys.path.append(str(FULLSUBNET_ROOT))
from third_party.FullSubNet.audio_zen.utils import initialize_module

class FullSubNetInference:
    def __init__(self,model,inferencer):
        self.model=model
        self.inferencer=inferencer
    def enhance(self,noisy_tensor):
        return self.inferencer.full_band_crm_mask(noisy_tensor,{"n_neighbor":15})
    