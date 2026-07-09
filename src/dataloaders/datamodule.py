from torch.utils.data import DataLoader
from src.dataloaders.paired_dataset import PairedDataset
from src.dataloaders.collate import collate_fn

def build_loader(metadata,clean_root,noisy_root,batch_size=8,shuffle=True,feature_type="spectrogram",noise_type="white",snr=10,num_workers=0,pin_memory=False,):
    dataset=PairedDataset(metadata_csv=metadata,clean_root=clean_root,noisy_root=noisy_root,feature_type=feature_type,noise_type=noise_type,snr=snr,)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=pin_memory,collate_fn=collate_fn,)