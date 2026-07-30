"""
mask.py- target mask generation for supervised speech enhancement.
supported masks are *ideal ratio mask(irm), *ideal binary mask (ibm),
*phase sensitive mask (psm).

only the irm is currently used for training, but the others are implemented
for possible futgure experiments."""

from __future__ import annotations
import numpy as np

class MaskGenerator:
    def __init__(self,epsilon:float=1e-8,clip:bool=True,):
        self.epsilon=epsilon
        self.clip=clip
    def ideal_ratio_mask(self,clean_mag:np.ndarray,noisy_mag:np.ndarray,)->np.ndarray:
        mask=clean_mag/(noisy_mag+self.epsilon)
        if self.clip:
            mask=np.clip(mask,0.0,1.0)
        return mask.astype(np.float32)
    
    def ideal_binary_mask(self,clean_mag:np.ndarray,noise_mag:np.ndarray,)->np.ndarray:
        mask=(clean_mag>=noise_mag).astype(np.float32)
        return mask
    
    def phase_sensitive_mask(self,clean_mag:np.ndarray,noisy_mag:np.ndarray,clean_phase:np.ndarray,noisy_phase:np.ndarray,)->np.ndarray:
        phase_difference=clean_phase-noisy_phase
        mask=(clean_mag*np.cos(phase_difference))/(noisy_mag+self.epsilon)
        if self.clip:
            mask=np.clip(mask,0.0,1.0)
        return mask.astype(np.float32)
    
    def __call__(self,clean_mag,noisy_mag,):
        return self.ideal_ratio_mask(clean_mag,noisy_mag,)
    