from abc import ABC, abstractmethod

class BaseEnhancer(ABC):
    def __init__(self,device="cpu"):
        self.device=device
    @abstractmethod
    def enhance(self,features):
        pass
    @abstractmethod
    def load_weights(self,checkpoint):
        pass
    def eval(self):
        return self
    