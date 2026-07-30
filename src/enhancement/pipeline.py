from pathlib import Path
from src.enhancement.inference_manager import InferenceManager

class EnhancementPipeline:
    def __init__(self,enhancer):
        self.manager=InferenceManager(enhancer)
    def enhance_file(self,input_file,output_file):
        return self.manager.enhance_file(input_file,output_file,)
    def enhance_directory(self,input_directory,output_directory,):
        return self.manager.enhance_directory(input_directory,output_directory,preserve_structure=True,)
    
    def export_debug(self,input_file,output_file,):
        if hasattr(self.manager,"export_debug",):
            return self.manager.export_debug(input_file,output_file,)
        return None
    