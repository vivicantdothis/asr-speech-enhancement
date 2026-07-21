import random
import numpy as np
from src.utils.audio import AudioIO
class BabbleGenerator:
    """
    will be generating babble noise by mixing several randomly selected speakers
    from the dataset. will exclude the target speaker, random crop from each utterance,
    miz 5-7 speakers, rms-normalize each utterance and return exactly target_length
    samples. <3"""

    def __init__(self,metadata,seed=42):
        self.metadata=metadata
        random.seed(seed)
        np.random.seed(seed)

    def _random_crop(self,signal,target_length):
        if len(signal)<=target_length:
            return signal
        start=random.randint(0,len(signal)-target_length)
        return signal[start:start+target_length]
    
    def _normalize_rms(self,signal):
        rms=np.sqrt(np.mean(signal**2))
        if rms<1e-8:
            return signal
        return signal/rms
    
    def generate(self,target_length,exclude_speaker=None,):
        available=self.metadata.copy()
        if exclude_speaker is not None:
            available=available[available["speaker_id"]!=exclude_speaker]
        if len(available)==0:
            available=self.metadata.copy()
        speakers=available["speaker_id"].unique()
        num_speakers=min(random.randint(5,8),len(speakers),)
        selected=random.sample(list(speakers),num_speakers)
        mixture=np.zeros(target_length,dtype=np.float32)
        
        for speaker in selected:
            files=available[available["speaker_id"]==speaker]
            wav_path=files.sample(1).iloc[0]["filepath"]
            audio, _=AudioIO.load_audio(wav_path)
            if len(audio)<target_length:
                repeats=target_length//len(audio)+1
                audio=np.tile(audio,repeats)
            audio=self._random_crop(audio,target_length)
            audio=self._normalize_rms(audio)
            mixture+=audio
        mixture /= num_speakers
        rms=np.sqrt(np.mean(mixture**2))
        if rms>1e-8:
            mixture/=rms
        return mixture.astype(np.float32)