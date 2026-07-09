from src.enhancement.cnn import CNNEnhancer

def build_enhancer(model_name,checkpoint=None,device="cpu"):
    model_name=model_name.lower()
    if model_name=="cnn":
        return CNNEnhancer(checkpoint=checkpoint,device=device,)
    raise ValueError(f"Unknown enhancement model:{model_name}")