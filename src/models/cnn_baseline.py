import torch
import torch.nn as nn
from .base_model import BaseSpeechEnhancementModel

class CNNBaseline(BaseSpeechEnhancementModel):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Conv2d(1,16,kernel_size=3,padding=1),nn.ReLU(),nn.Conv2d(16,32,kernel_size=3,padding=1),nn.ReLU(),nn.Conv2d(32,1,kernel_size=3,padding=1),)
    def forward(self,x):
        return self.network(x)
    