from src.enhancement.classical.cnn import CNNEnhancer
from src.enhancement.pretrained.deepfilternet import DeepFilterNetEnhancer

_ENHANCERS={"cnn":CNNEnhancer, "deepfilternet":DeepFilterNetEnhancer,}

def build_enhancer(model_name,**kwargs):
    model_name=model_name.lower()
    if model_name not in _ENHANCERS:
        raise ValueError(f"Unknown enhancer:{model_name}")
    return _ENHANCERS[model_name](*kwargs)