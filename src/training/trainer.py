import torch

class Trainer:
    def __init__(self,model,optimizer,criterion,device):
        self.model=model
        self.optimizer=optimizer
        self.criterion= criterion
        self.device=device
    def train_epoch(self,loader):
        self.model.train()
        running_loss=0
        for batch in loader:
            x=batch["input"].to(self.device)
            y=batch["target"].to(self.device)
            pred=self.model(x)
            loss=self.criterion(pred,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss+=loss.item()
        return running_loss/len(loader)