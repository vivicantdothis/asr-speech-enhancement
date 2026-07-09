from pathlib import Path
import random
import numpy as np
from src.utils.audio import AudioIO
class BabbleGenerator:
    def __init__(self,metadata):
        self.metadata=metadata
    def generate(self,target_length):
        speakers=self.metadata["speaker_id"].unique()
        num_speakers=min(5,len(speakers))
        selected=random.sample(list(speakers),num_speakers)
        speech=[]
        for speaker in selected:
            speaker_files=self.metadata[self.metadata["speaker_id"]==speaker]
            file=speaker_files.sample(1).iloc[0]["filepath"]
            audio, _=AudioIO.load_audio(file)
            speech.append(audio)
        min_len=min(len(s) for s in speech)
        speech=[s[:min_len] for s in speech]
        babble=np.sum(speech,axis=0)
        if len(babble)<target_length:
            repeats=target_length//len(babble)+1
            babble=np.tile(babble,repeats)
        babble=babble[:target_length]
        babble/=np.max(np.abs(babble))+1e-10
        return babble.astype(np.float32)