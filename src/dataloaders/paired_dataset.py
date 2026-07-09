from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src.utils.file_utils import FileUtils

class PairedDataset(Dataset):
    def __init__(self,metadata_csv,clean_root,noisy_root,feature_type="spectrogram",noise_type="white",snr=10,):
        self.df=pd.read_csv(metadata_csv)
        self.clean_root=Path(clean_root)
        self.noisy_root=Path(noisy_root)
        self.feature_type=feature_type
        self.noise_type=noise_type
        self.snr=snr
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        relative=FileUtils.relative_timit_path(row["filepath"]).with_suffix(".npy")
        clean_path=(self.clean_root/relative.parent/self.feature_type/relative.name)
        noisy_name=FileUtils.noisy_filename(relative,self.noise_type,self.snr,).with_suffix(".npy")
        noisy_path=(self.noisy_root/noisy_name.parent/self.feature_type/noisy_name.name)
        if not clean_path.exists():
            raise FileNotFoundError(clean_path)
        if not noisy_path.exists():
            raise FileNotFoundError(noisy_path)
        clean=np.load(clean_path)
        noisy=np.load(noisy_path)
        return {"input":noisy.astype(np.float32),"target":clean.astype(np.float32),"speaker":row["speaker_id"],"sentence":relative.stem,"noise_type":self.noise_type,"snr":self.snr,}
    