from __future__ import annotations
from pathlib import Path
from src.utils.audio import AudioIO

class InferenceManager:
    def __init__(self,enhancer):
        self.enhancer=enhancer
    def enhance_file(self,input_path,output_path):
        input_path=Path(input_path)
        output_path=Path(output_path)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        waveform,sr = AudioIO.load_audio(input_path)
        enhanced=self.enhancer.enhance(waveform)
        AudioIO.save_audio(output_path,enhanced,sr,)
        return output_path
    
    def export_debug(self,input_path,output_path,):
        if not hasattr(self.enhancer,"enhance_with_debug",):
            return
        if not hasattr(self.enhancer,"export_results",):
            return
        input_path=Path(input_path)
        output_path=Path(output_path)
        waveform,sr=AudioIO.load_audio(input_path,)
        result=self.enhancer.enhance_with_debug(waveform,)
        self.enhancer.export_results(filename=input_path.stem,result=result,)