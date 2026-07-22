from __future__ import annotations
import pandas as pd
import numpy as np

class MetricAggregator:
    DEFAULT_METRICS=["SNR","SI_SDR","STOI",]
    
    def __init__(self,dataframe:pd.DataFrame):
        self.df=dataframe.copy()
    def available_metrics(self):
        metrics=[]
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                metrics.append(column)
        return metrics
    
    def summary(self,metrics=None,):
        metrics=metrics or self.DEFAULT_METRICS
        summary=self.df[metrics].agg(["mean","std","min","max","median",])
        return summary.T
    
    def by_model(self,metrics=None,):
        metrics=metrics or self.DEFAULT_METRICS
        return (self.df.groupby("model")[metrics].agg(["mean","std","median",]).round(4))
    def by_snr(self,metrics=None,):
        metrics=metrics or self.DEFAULT_METRICS
        return (self.df.groupby("snr")[metrics].agg(["mean","std","median",]).round(4))
    def by_split(self,metrics=None,):
        metrics=metrics or self.DEFAULT_METRICS
        return (self.df.groupby("split")[metrics].agg(["mean","std","median",]).round(4))
    
    def grouped(self,group_columns,metrics=None,):
        metrics=metrics or self.DEFAULT_METRICS
        return (self.df.groupby(group_columns)[metrics].agg(["mean","std","median","min","max",]).round(4))
    
    def correlation(self):
        numeric=self.df.select_dtypes(include=np.number)
        return numeric.corr()
    
    def export_tables(self,output_directory="outputs/reports/tables",):
        from pathlib import Path
        output_directory=Path(output_directory)
        output_directory.mkdir(parents=True,exist_ok=True,)
        self.summary().to_csv(output_directory/"overall_summary.csv")
        self.by_model().to_csv(output_directory/"by_model.csv")
        self.by_noise().to_csv(output_directory/"by_noise.csv")
        self.by_snr().to_csv(output_directory/"by_snr.csv")
        self.by_split().to_csv(output_directory/"by_split.csv")
        self.correlation().to_csv(output_directory/"correlation.csv")
        print("="*60)
        print("Saved aggregation tables")
        print(output_directory)
        print("="*60)