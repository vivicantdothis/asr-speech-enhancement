from __future__ import annotations
from pathlib import Path
from tqdm import tqdm

import torch
from torch.cuda.amp import (autocast,GradScaler,)
from .config import (DEVICE, EPOCHS, LEARNING_RATE, GRADIENT_CLIP, CHECKPOINT_ROOT, CHECKPOINT_NAME, LAST_CHECKPOINT,)
from .losses import EnhancementLoss

class CNNTrainer:
    def __init__(self,model,train_loader,validation_loader,checkpoint_dir):
        self.device=DEVICE
        self.model=model.to(self.device)
        self.train_loader=train_loader
        self.validation_loader=validation_loader
        self.checkpoint_dir=Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True,exist_ok=True,)
        self.loss_fn=EnhancementLoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=LEARNING_RATE,)
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode="min",factor=0.5,patience=5,verbose=True,)
        self.scaler=GradScaler()
        self.best_loss=float("inf")
        self.history={"train":[],"validation":[],}
    def train(self):
        print("="*70)
        print("Starting CNN Training")
        print("="*70)
        for epoch in range(EPOCHS):
            train_loss=self.train_epoch(epoch)
            validation_loss=self.validate(epoch)
            self.scheduler.step(validation_loss)
            self.history["train"].append(train_loss)
            self.history["validation"].append(validation_loss)
            self.save_checkpoint(LAST_CHECKPOINT,epoch,)
            if validation_loss<self.best_loss:
                self.best_loss=validation_loss
                self.save_checkpoint(CHECKPOINT_NAME,epoch,)
                print()
                print(f"Epoch {epoch+1:0.3f}"f" | Train {train_loss:.5f}"f" | Val {validation_loss:.5f}")
                print()
        
    
    def train_epoch(self,epoch):
        self.model.train()
        running_loss=0.0
        progress=tqdm(self.train_loader,desc=f"Train {epoch+1}",)
        for batch in progress:
            noisy= batch["noisy"].to(self.device)
            clean=batch["clean"].to(self.device)
            mask=batch["mask"].to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                prediction=self.model(noisy)
                loss,metrics=self.loss_fn(prediction,mask,noisy,clean,)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm(self.model.parameters(),GRADIENT_CLIP,)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        return running_loss/len(self.train_loader)
    
    @torch.no_grad()
    def validate(self,epoch):
        self.model.eval()
        running_loss=0.0
        progress=tqdm(self.validation_loader,desc=f"Val{epoch+1}",)
        for batch in progress:
            noisy=batch["noisy"].to(self.device)
            clean=batch["clean"].to(self.device)
            mask=batch["mask"].to(self.device)
            prediction=self.model(noisy)
            loss,metrics=self.loss_fn(prediction,mask,noisy,clean,)
            running_loss += loss.item()
        return running_loss/len(self.validation_loader)
    
    def save_checkpoint(self,filename,epoch,):
        path=self.checkpoint_dir/filename
        torch.save({"epoch":epoch,"model":self.model.state_dict(),"optimizer":self.optimizer.state_dict(),"scheduler":self.scheduler.state_dict(),"best_loss":self.best_loss,"history":self.history,},path,)

    def load_checkpoint(self,checkpoint,):
        checkpoint=torch.load(checkpoint,map_location=self.device,)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_loss=checkpoint["best_loss"]
        self.history=checkpoint["history"]
        return checkpoint["epoch"]
    
        