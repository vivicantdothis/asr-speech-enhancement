from src.utils.audio import AudioIO
from src.enhancement.reconstruct import SpectrogramReconstructor
from src.enhancement.inference import InferenceEngine
from src.enhancement.inference_manager import InferenceManager

class EnhancementPipeline:
    def __init__(self,enhancer,config):
        self.enhancer=enhancer
        self.engine=InferenceEngine(enhancer,config)
        self.config=config
    def __init__(self,enhancer):
        self.manager=InferenceManager(enhancer)
    def enhance_file(self,input_file,output_file):
        return self.manager.enhance_file(input_file,output_file,)