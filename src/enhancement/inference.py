import torch
import numpy as np
from src.features.feature_extractor import FeatureExtractor
from src.enhancement.pipeline import EnhancementPipeline

class EnhancementInference:
    def __init__(self,enhancer):
        self.enhancer=enhancer
    def enhance_tensor(self,features):
        if features.ndim==3:
            features=features.unsqueeze(0)
        enhanced=self.enhancer.enhance(features)
        return enhanced.squeeze(0)
    def enhance_batch(self,batch):
        return self.enhancer.enhance(batch)
    
class InferenceEngine:
    def __init__(self,enhancer,config):
        self.enhancer=enhancer
        self.config=config
    def enhance_waveform(self,waveform):
        features=FeatureExtractor.extract(waveform,self.config)
        logmel=features["logmel"]
        x=torch.tensor(logmel,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        enhanced=self.enhancer.enhance(x)
        return enhanced.squeeze().numpy()