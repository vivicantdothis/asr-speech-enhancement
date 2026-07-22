"""
ranks speech enhancement algorithms according to objective metrics.
this file is intentionally independent from plotting and table generation so that rankings can be resused throughout the visualization 
pipeline.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

class ModelRanker:
    METRICS=["SNR","SI_SDR","STOI",]
    def __init__(self,dataframe:pd.DataFrame,output_directory="outputs/reports/rankings",):
        self.df=dataframe.copy()
        self.output_directory=Path(output_directory)
        self.output_directory.mkdir(parents=True,exist_ok=True,)
    
    def metric_means(self):
        return (self.df.groupby("model")[self.METRICS].mean())
    def metric_ranks(self):
        means=self.metric_means()
        ranks=pd.DataFrame(index=means.index)
        for metric in self.METRICS:
            ranks[f"{metric}_Rank"]=(means[metric].rank(ascending=False,method="min",).astype(int))
        return ranks
    
    def overall_ranking(self):
        means=self.metric_means()
        ranks=self.metric_ranks()
        results=means.copy()
        for column in ranks.columns:
            results[column]=ranks[column]
        results["AverageRank"]=(ranks.mean(axis=1))
        results=results.sort_values("AverageRank",ascending=True,)
        results["OverallRank"]=(range(1,len(results)+1))
        return results
    
    def best_models(self):
        means=self.metric_means()
        winners={}
        for metric in self.METRICS:
            winners[metric]={"model":means[metric].idxmax(),"value":means[metric].max(),}
        return pd.DataFrame(winners).T
    
    def save(self):
        metric_ranks=self.metric_ranks()
        overall=self.overall_ranking()
        winners=self.best_models()
        metric_ranks.to_csv(self.output_directory/"metric_ranks.csv")
        overall.to_csv(self.output_directory/"overall_ranking.csv")
        winners.to_csv(self.output_directory/"best_models.csv")
        return {"metric_ranks":metric_ranks,"overall":overall,"best_models":winners,}
    
    def generate_all(self):
        outputs=self.save()
        print("="*60)
        print("Model rankings generated.")
        print(self.output_directory)
        print("="*60)
        return outputs
    