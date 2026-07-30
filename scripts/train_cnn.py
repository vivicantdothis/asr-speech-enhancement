from __future__ import annotations
import argparse
import random
from pathlib import Path
import time

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.enhancement.classical.cnn.dataset import SpeechEnhancementDataset
from src.enhancement.classical.cnn.model import build_model
from src.enhancement.classical.cnn.trainer import CNNTrainer
from src.enhancement.classical.cnn.config import (BATCH_SIZE,NUM_WORKERS,CHECKPOINT_ROOT,DEVICE,SEED,)

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_datasets(noise):
    train_clean=Path("data/clean/train")
    val_clean=Path("data/clean/val")
    train_noisy=Path("data/noisy")/noise/"train"
    val_noisy=Path("data/noisy")/noise/"val"
    train_dataset=SpeechEnhancementDataset(clean_root=train_clean,noisy_root=train_noisy,)
    validation_dataset=SpeechEnhancementDataset(clean_root=val_clean,noisy_root=val_noisy,)
    return (train_dataset,validation_dataset,train_clean,train_noisy,val_clean,val_noisy,)

#DATALOADERS
def build_loaders(train_dataset,validation_dataset):
    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available(),drop_last=True,)
    validation_loader=DataLoader(validation_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available(),)
    return train_loader,validation_loader

def main():
    start=time.time()
    parser=argparse.ArgumentParser(description="Train CNN U-Net Speech Enhancement Model")
    parser.add_argument("--noise",default="white",choices=["white","babble",])
    parser.add_argument("--resume", type=str,default=None,help="Resume training from checkpoint",)
    args=parser.parse_args()

    set_seed(SEED)
    checkpoint_dir=CHECKPOINT_ROOT/args.noise
    checkpoint_dir.mkdir(parents=True,exist_ok=True,)
    (train_dataset,validation_dataset,train_clean,train_noisy,val_clean,val_noisy,)=build_datasets(args.noise)

    train_loader,validation_loader=build_loaders(train_dataset,validation_dataset,)

    model=build_model()
    trainer=CNNTrainer(model=model,train_loader=train_loader,validation_loader=validation_loader,checkpoint_dir=checkpoint_dir,)
    if args.resume is not None:
        print(f"\n Resuming from:\n{args.resume}\n")
        trainer.load_checkpoint(args.resume)

    config_file=checkpoint_dir/"experiments.txt"
    with open(config_file,"w") as f:
        f.write(f"Noise={args.noise}\n")
        f.write(f"Batch size={BATCH_SIZE}\n")
        f.write(f"Workers={NUM_WORKERS}\n")
        f.write(f"Device={DEVICE}\n")
        f.write(f"Seed={SEED}\n")
    with open(checkpoint_dir/"architecture.txt","w",) as f:
        f.write(str(model))
    
    print("="*50)
    print("CNN Speech Enhancement Training")
    print("="*50)
    print(f"Device:{DEVICE}")
    print(f"Noise:{args.noise}")
    print()
    print(f"Training clean:{train_clean}")
    print(f"Training noisy:{train_noisy}")
    print(f"Validation clean:{val_clean}")
    print(f"Validation noisy:{val_noisy}")
    print()
    print(f"Training examples:{len(train_dataset)}")
    print(f"Validation examples:{len(validation_dataset)}")
    print()
    print(f"Checkpoint directory:{checkpoint_dir}")
    print("="*50)
    trainer.train()
    if hasattr(trainer,"history"):
        history=pd.DataFrame(trainer.history)
        history.to_csv(checkpoint_dir/"history.csv",index=False,)
    
    elapsed=time.time()-start
    print()
    print("="*50)
    print("Training completed.")
    print(f"Training time:{elapsed/60:.2f minutes}")
    print("="*50)

if __name__=="__main__":
    main()
