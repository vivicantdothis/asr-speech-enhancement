from pathlib import Path
from loguru import logger

class ExperimentLogger:
    @staticmethod
    def setup(log_dir="logs"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger.remove()
        logger.add(f"{log_dir}/experiment.log", rotation="10 MB", level="INFO")
        return logger