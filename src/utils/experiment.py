from pathlib import Path
import json
from datetime import datetime

class ExperimentManager:
    def __init__(self,experiment_name):
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir=Path(f"outputs/{experiment_name}/{timestamp}")
        self.run_dir.mkdir(parents=True,exist_ok=True)
    def save_config(self,config_dict):
        config_file=(self.run_dir/"config_snapshot.json")
        with open(config_file,"w") as f:
            json.dump(config_dict,f,indent=4)
    @property
    def path(self):
        return self.run_dir