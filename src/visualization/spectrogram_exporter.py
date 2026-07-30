from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class SpectrogramExporter:
    def __init__(self,output_root="outputs/spectrograms",cmap="magma",dpi=300,):
        self.output_root=Path(output_root)
        self.cmap=cmap
        self.dpi=dpi
    
    def _directories(self,method,category,):
        png=(self.output_root/method/category/"png")
        npy=(self.output_root/method/category/"npy")
        png.mkdir(parents=True,exist_ok=True,)
        npy.mkdir(parents=True,exist_ok=True,)
        return png,npy
    
    def save(self,spectrogram,method,category,filename,title=None,):
        png_dir,npy_dir=self._directories(method,category,)
        np.save(npy_dir/f"{filename}.npy",spectrogram,)
        plt.figure(figsize=(10,4))
        plt.imshow(spectrogram,origin="lower",aspect="auto",cmap=self.cmap,)
        plt.colorbar()
        plt.title(title or category)
        plt.tight_layout()
        plt.savefig(png_dir/f"{filename}.png",dpi=self.dpi,)
        plt.close()
    
    def export_all(self,method,filename,clean=None,noisy=None,enhanced=None,mask=None,):
        if clean is not None:
            self.save(clean,method,"clean",filename,"Clean Spectrogram",)
        if noisy is not None:
            self.save(noisy,method,"noisy",filename,"Noisy Spectrogram",)
        if enhanced is not None:
            self.save(enhanced,method,"enhanced",filename,"Enhanced Spectrogram",)
        if mask is not None:
            self.save(mask,method,"predicted_mask",filename,"Predicted Mask",)
