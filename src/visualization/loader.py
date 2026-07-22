from __future__ import annotations
from pathlib import Path
import pandas as pd

class EvaluationLoader:
    def __init__(self,root="outputs/evaluation"):
        self.root=Path(root)
    def discover(self):
        return sorted(self.root.rglob("results.csv"))
    def load_file(self,csv_file:Path):
        df=pd.read_csv(csv_file)
        relative=csv_file.relative_to(self.root)
        parts=relative.parts
        df["model"]=parts[0]
        df["noise"]=parts[1]
        df["snr"]=parts[2]
        df["split"]=parts[3]
        return df
    
    def load_all(self):
        files=self.discover()
        if len(files)==0:
            raise FileNotFoundError(f"No evaluation CSVs found under {self.root}")
        dfs=[]
        for csv in files:
            dfs.append(self.load_file(csv))
        merged=pd.concat(dfs,ignore_index=True,)
        metadata=["mode,","noise","snr","split",]
        remaining=[c for c in merged.columns if c not in metadata]
        merged=merged[metadata+remaining]
        merged=merged.sort_values(["model","noise","snr","split",]).reset_index(drop=True)
        return merged
    
    def save_master_csv(self,output="outputs/reports/master_results.csv",):
        df=self.load_all()
        output=Path(output)
        output.parent.mkdir(parents=true,exist_ok=True,)
        df.to_csv(output,index=False,)
        print(f"Saved merged results -> {output}")
        return df