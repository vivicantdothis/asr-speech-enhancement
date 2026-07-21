from abc import ABC, abstractmethod
class BaseEnhancer(ABC):
    def __init__(self,device="cpu"):
        self.device=device
    @property
    @abstractmethod
    def input_type(self):
        pass
    @abstractmethod
    def enhance(self,input_data):
        pass
    @abstractmethod
    def load_weights(self,checkpoint):
        pass
    def eval(self):
        return self
    