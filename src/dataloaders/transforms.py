import torch

class ToTensor:
    def __call__(self,sample):
        sample["input"]=torch.from_numpy(sample["input"])
        sample["target"]=torch.from_numpy(sample["target"])
        return sample