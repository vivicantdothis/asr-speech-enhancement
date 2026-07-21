import numpy as np

class SNRMixer:
    @staticmethod
    def rms(signal):
        return np.sqrt(np.mean(signal.astype(np.float32) ** 2))

    @staticmethod
    def mix(clean, noise, snr_db=10):
        clean = clean.astype(np.float32)
        noise = noise.astype(np.float32)
        if len(noise) < len(clean):
            repeats = len(clean) // len(noise) + 1
            noise = np.tile(noise, repeats)
        noise = noise[:len(clean)]
        clean_rms = SNRMixer.rms(clean)
        noise_rms = SNRMixer.rms(noise)
        target_noise_rms = clean_rms / (10 ** (snr_db / 20))
        scale = target_noise_rms / (noise_rms + 1e-10)
        noisy = clean + noise * scale
        peak=np.max(np.abs(noisy))
        if peak>0.999:
            noisy=noisy/peak*0.999
        return noisy.astype(np.float32)
    
    @staticmethod
    def compute_snr(clean,noisy):
        clean=clean.astype(np.float32)
        noisy=noisy.astype(np.float32)
        noise=noisy-clean
        signal_power=np.mean(clean**2)
        noise_power=np.mean(noise**2)
        return 10*np.log10(signal_power/(noise_power+1e-10))
    