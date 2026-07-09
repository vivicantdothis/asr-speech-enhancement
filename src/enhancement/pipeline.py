from src.utils.audio import AudioIO
from src.enhancement.reconstruct import SpectrogramReconstructor
from src.enhancement.inference import InferenceEngine

class EnhancementPipeline:
    def __init__(self,enhancer,config):
        self.engine=InferenceEngine(enhancer,config)
        self.config=config
    def enhance_file(self,input_file,output_file):
        waveform,sr=AudioIO.load_audio(input_file)
        enhanced_features=self.engine.enhance_waveform(waveform)
        enhanced_audio=SpectrogramReconstructor.griffin_lim(enhanced_features,self.config)
        AudioIO.save_audio(output_file,enhanced_audio,sr)