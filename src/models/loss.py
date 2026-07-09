import torch.nn as nn

def mse_loss():
    return nn.MSELoss()

def l1_loss():
    return nn.L1Loss()