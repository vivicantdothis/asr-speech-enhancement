"""
losses.py: loss functions for the cnn speech enhancement.
implements mask mse loss, spectrogram l1 loss, combined enhancement
loss and si_dr, pesq surrogate loss.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import (MASK_LOSS_WEIGHT,SPECTROGRAM_LOSS_WEIGHT,)

class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.MSELoss()
    def forward(self,predicted_mask,target_mask):
        return self.loss(predicted_mask,target_mask,)
    
class SpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.L1Loss()
    def forward(self,enhanced_mag,clean_mag,):
        return self.loss(enhanced_mag,clean_mag,)
    
#Combined Enhancement Loss
class EnhancementLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_loss=MaskLoss()
        self.spec_loss=SpectrogramLoss()
    
    def forward(self,predicted_mask,target_mask,noisy_mag,clean_mag,):
        mask_loss=self.mask_loss(predicted_mask,target_mask,)
        enhanced_mag=predicted_mask*noisy_mag
        spec_loss=self.spec_loss(enhanced_mag,clean_mag,)
        total=(MASK_LOSS_WEIGHT*mask_loss+SPECTROGRAM_LOSS_WEIGHT*spec_loss)
        metrics={"total":total.detach(),"mask":mask_loss.detach(),"spectrogram":spec_loss.detach(),}
        return total,metrics
    
    
