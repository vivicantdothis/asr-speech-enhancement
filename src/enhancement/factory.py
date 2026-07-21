from src.enhancement.classical.cnn import CNNEnhancer
from src.enhancement.pretrained.deepfilternet import DeepFilterNetEnhancer
from src.enhancement.pretrained.fullsubnet import FullSubNetEnhancer
from src.enhancement.classical.enhancer import SpectralSubtractionEnhancer
from src.enhancement.classical.wiener.enhancer import WienerEnhancer

_ENHANCERS={"cnn":CNNEnhancer, "deepfilternet":DeepFilterNetEnhancer, "fullsubnet":FullSubNetEnhancer,"spectral":SpectralSubtractionEnhancer,"wiener":WienerEnhancer}

def build_enhancer(model_name,**kwargs):
    model_name=model_name.lower()
    if model_name not in _ENHANCERS:
        raise ValueError(f"Unknown enhancement model:{model_name}."f"Available models: {list(_ENHANCERS.keys())}")
    
    return _ENHANCERS[model_name](**kwargs)