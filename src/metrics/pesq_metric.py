try:
    from pesq import pesq
except ImportError:
    pesq=None

def compute_pesq(clean,enhanced,sr):
    if pesq is None:
        raise RuntimeError("PESQ is not installed.")
    return pesq(sr,clean,enhanced,"wb",)
