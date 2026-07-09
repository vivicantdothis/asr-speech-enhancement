from pathlib import Path

class FileUtils:
    @staticmethod
    def relative_timit_path(filepath):
        filepath=Path(filepath)
        parts=list(filepath.parts)
        if "TRAIN" in parts:
            idx = parts.index("TRAIN")
        elif "TEST" in parts:
            idx=parts.index("TEST")
        else:
            raise ValueError(f"Cannot locate TRAIN/TEST in {filepath}")
        relative=Path(*parts[idx+1:])
        return relative.with_suffix(".wav")
    
    @staticmethod
    def noisy_filename(relative_path,noise_type,snr):
        relative_path=Path(relative_path)
        filename=(f"{relative_path.stem}" f"_{noise_type}" f"_{snr}dB.wav")
        return relative_path.parent/filename
