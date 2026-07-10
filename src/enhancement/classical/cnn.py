import torch
from src.models.cnn_baseline import CNNBaseline
from src.enhancement.base import BaseEnhancer
from src.training.checkpoint import CheckpointManager

class CNNEnhancer(BaseEnhancer):
    def __init__(self,checkpoint=None,device="cpu"):
        super().__init__(device)
        self.model=CNNBaseline().to(device)
        if checkpoint is not None:
            self.load_weights(checkpoint)
        self.model.eval()
    def enhance(self,features):
        self.model.eval()
        with torch.no_grad():
            features=features.to(self.device)
            output=self.model(features)
        return output.cpu()
    def load_weights(self,checkpoint):
        self.model, _, _, _=CheckpointManager.load_checkpoint(model=self.model,checkpoint_path=checkpoint,optimizer=None,map_location=self.device,)
        self.model.eval()