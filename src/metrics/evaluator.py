import numpy as np
from src.metrics.snr_metric import snr
from src.metrics.sisdr_metric import si_sdr
from src.metrics.stoi_metric import compute_stoi
try:
    from src.metrics.pesq_metric import compute_pesq
    HAS_PESQ=True
except Exception:
    HAS_PESQ=False


class Evaluator:
    def evaluate(self,clean,enhanced,sr,):
        clean=np.asarray(clean)
        enhanced=np.asarray(enhanced)
        min_len=min(len(clean),len(enhanced))
        clean=clean[:min_len]
        enhanced=enhanced[:min_len]
        metrics={}
        metrics["SNR"]=snr(clean,enhanced)
        metrics["SI_SDR"]=si_sdr(clean,enhanced)
        metrics["STOI"]=compute_stoi(clean,enhanced,sr)
        try:
            metrics["PESQ"]=compute_pesq(clean,enhanced,sr,)
        except Exception:
            metrics["PESQ"]=None
        return metrics