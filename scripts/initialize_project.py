from src.utils.config import ConfigLoader
from src.utils.seed import SeedManager
from src.utils.logger import ExperimentLogger
from src.utils.path import PathManager

def main():
    configs=ConfigLoader.load_all()
    seed=configs["experiment"]["experiment"]["seed"]
    SeedManager.set_seed(seed)
    logger=ExperimentLogger.setup()
    PathManager.initialize()
    logger.info("Project initialized successfully")
    logger.info(f"Seed:{seed}")
    logger.info("Configuration files loaded")

if __name__=="__main__":
    main()