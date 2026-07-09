from pathlib import Path
import yaml

class ConfigLoader:
    @staticmethod
    def load_yaml(path: str):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    @classmethod
    def load_all(cls, config_dir="configs"):
        config_dir = Path(config_dir)
        configs = {}
        for yaml_file in config_dir.glob("*.yaml"):
            configs[yaml_file.stem] = cls.load_yaml(yaml_file)
        return configs