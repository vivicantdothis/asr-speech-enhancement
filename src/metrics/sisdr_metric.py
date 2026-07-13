import numpy as np

def si_sdr(reference,estimation):
    reference=np.asarray(reference)
    estimation=np.asarray(estimation)
    alpha=(np.dot(estimation,reference)/np.dot(reference,reference))
    target=alpha*reference
    noise=estimation-target
    return 10*np.log10(np.sum(target**2)/np.sum(noise**2))
