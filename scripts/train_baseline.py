import torch

from src.training.device import get_device
from src.training.trainer import Trainer
from src.models.cnn_baseline import CNNBaseline
from src.models.loss import mse_loss

device=get_device()
model=CNNBaseline().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
criterion=mse_loss()
print(model)
print(device)