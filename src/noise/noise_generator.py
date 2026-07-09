import numpy as np


class NoiseGenerator:
    @staticmethod
    def white(length):
        return np.random.normal(0,1,length).astype(np.float32)