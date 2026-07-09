from src.utils.audio import AudioIO
from src.utils.path import PathManager
from src.utils.seed import SeedManager
from src.utils.logger import ExperimentLogger
from src.utils.experiment import ExperimentManager

def main():
    SeedManager.set_seed(42)
    logger=ExperimentLogger.setup()
    PathManager.initialize()
    experiment=ExperimentManager("speech_enhancement")
    logger.info(f"Experiment directory:" f"{experiment.path}")
    logger.info("Phase 1.5 successful")

if __name__ == "__main__":
    main()