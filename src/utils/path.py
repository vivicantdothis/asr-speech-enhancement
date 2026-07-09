from pathlib import Path
class PathManager:
    REQUIRED_DIRS=["data/clean", "data/noisy", "data/enhanced","features/clean","features/noisy", "features/enhanced", "metrics", "logs", "outputs/waveforms","outputs/spectrograms","outputs/reports","outputs/audio","outputs/features","outputs/metrics", "checkpoints"]
    @classmethod
    def initialize(cls):
        for directory in cls.REQUIRED_DIRS:
            Path(directory).mkdir(parents=True,exist_ok=True)
            print("[INFO] Directory structure initialized.")