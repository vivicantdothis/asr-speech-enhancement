import torch 
def move_to_device(tensor,device):
    return tensor.to(device)
def detach_cpu(tensor):
    return tensor.detach().cpu()
