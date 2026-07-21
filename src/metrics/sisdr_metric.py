import numpy as np

def si_sdr(reference,estimation):
    reference=np.asarray(reference,dtype=np.float32)
    estimation=np.asarray(estimation,dtype=np.float32)
    alpha=(np.dot(estimation,reference)/np.dot(reference,reference)+1e-8)
    target=alpha*reference
    noise=estimation-target
    return 10*np.log10(np.sum(target**2)/np.sum(noise**2)+1e-8)
