from __future__ import annotations
import numpy as np
import pandas as pd

from scipy.stats import (ttest_rel,wilcoxon,t,)

class StatisticsAnalyzer:
    def __init__(self,dataframe):
        self.df=dataframe.copy()
    
    @staticmethod
    def confidence_interval(values,confidence=0.95):
        values=np.asarray(values)
        n=len(values)
        mean=np.mean(values)
        sem=np.std(values,ddof=1)/np.sqrt(n)
        margin=sem*t.ppf((1+confidence)/2,n-1)
        return {"mea,":mean,"lower":mean-margin,"upper":mean+margin,"margin":margin,}
    
    @staticmethod
    def cohens_d(x,y):
        #cohen's d effect size
        x=np.asarray(x)
        y=np.asarray(y)
        diff=x-y
        return diff.mean()/diff.std(ddof=1)
    
    def compare_models(self,model_a,model_b,metric="STOI",):
        a=(self.df[self.df.model==model_a].sort_values("file").reset_index(drop=True))
        b=(self.df[self.df.model==model_b].sort_values("file").reset_index(drop=True))
        n=min(len(a),len(b))
        x=a[metric].values[:n]
        y=b[metric].values[:n]
        t_result=ttest_rel(x,y)
        try:
            w_result=wilcoxon(x,y)
        except ValueError:
            w_result=None
        return {
            "metric":metric,
            "model_a":model_a,
            "model_b":model_b,
            "paired_t_statistic":t_result.statistic,
            "paired_t_pvalue":t_result.pvalue,
            "wilcoxon_statistic": None if w_result is None else w_result.statistic,
            "wilcoxon_pvalue": None if w_result is None else w_result.p_value,
            "cohens_d":
            self.cohes_d(x,y),
            "mean_model_a":np.mean(x),
            "mean_model_b":np.mean(y),
        }
    
    def confidence_table(self,metric="STOI",):
        rows=[]
        for model in sorted(self.df.model.unique()):
            values=self.df[self.df.model==model][metric]
            ci=self.confidence_interval(values)
            rows.append({"Model":model,"Mean":ci["mean"],"CI Lower":ci["lower"],"CI Upper":ci["upper"],"Margin":ci["margin"],})
        return pd.DataFrame(rows)
    
    def descriptive_statistics(self,metrics=None,):
        if metrics is None:
            metrics=["SNR","SI_SDR","STOI",]
            rows=[]
        for model in sorted(self.df.model.unique()):
            subset=self.df[self.df.model==model]
            for metric in metrics:
                values=subset[metric]
                rows.append({"Model":model,"Metric":metric,"Mean":values.mean(),"Std":values.std(),"Median":values.median(),"Min":values.min(),"Max":values.max(),})
        return pd.DataFrame(rows)
            
    def export(self,output_directory="outputs/reports/statistics",):
        from pathlib import Path
        output_directory=Path(output_directory)
        output_directory.mkdir(parents=True,exist_ok=True,)
        self.descriptive_statistics().to_csv(output_directory/"descriptive_statistics.csv",index=False,)
        for metric in ["SNR","SI_SDR","STOI",]:
            self.confidence_table(metric).to_csv(output_directory/f"{metric.lower()}_confidence.csv",index=False,)
        print("="*50)
        print("Statistical analysis exported!")
        print(output_directory)
        print("="*50)
