import pandas as pd
from pathlib import Path

class EvaluationReport:
    def __init__(self):
        self.rows=[]
    def add(self,filename,model_name,metrics):
        row={"file":filename,"model":model_name,}
        row.update(metrics)
        self.rows.append(row)
    def dataframe(self):
        return pd.DataFrame(self.rows)
    def save(self,output_csv):
        output_csv=Path(output_csv)
        output_csv.parent.mkdir(parents=True,exist_ok=True,)
        df=self.dataframe()
        df.to_csv(output_csv,index=False,)
        return df