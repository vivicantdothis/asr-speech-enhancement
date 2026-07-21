from dataclasses import dataclass
import numpy as np

@dataclass
class EnhancementResult:
    waveform: np.ndarray
    sample_rate:int