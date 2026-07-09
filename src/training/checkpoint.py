from pathlib import Path
import torch

class CheckpointManager:
    @staticmethod
    def save_checkpoint(model,optimizer,epoch,save_path,**kwargs):
        save_path=Path(save_path)
        save_path.parent.mkdir(parents=True,exist_ok=True)
        checkpoint={"epoch":epoch, "model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict() if optimizer is not None else None}
        checkpoint.update(kwargs)
        torch.save(checkpoint, save_path)
    
    @staticmethod
    def load_checkpoint(model,checkpoint_path,optimizer=None, map_location="cpu"):
        checkpoint=torch.load(checkpoint_path,map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        if(optimizer is not None and checkpoint["optimizer_state_dict"] is not None):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch=checkpoint.get("epoch",0)
        return model,optimizer,epoch