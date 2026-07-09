from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SpeechDataset(Dataset):
    def __init__(self,metadata_csv,feature_root,feature_type="logmel",):
        self.df=pd.read_csv(metadata_csv)
        self.feature_root=Path(feature_root)
        self.feature_type=feature_type
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        relative=Path(row["filepath"])
        parts=relative.parts
        if "TRAIN" in parts:
            i=parts.index("TRAIN")
        else:
            i=parts.index("TEST")
        relative=Path(*parts[i+1:]).with_suffix(".npy")
        feature=np.load(self.feature_root/relative.parents/self.feature_type/relative.name)
        return {"feature":feature.astype(np.float32),"speaker":row["speaker_id"],"sentence":row["sentence_id"],"filepath":row["filepath"]}