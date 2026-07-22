from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

class PlotGenerator:
    def __init__(self,dataframe:pd.DataFrame,output_directory="outputs/reports/plots",):
        self.df=dataframe.copy()
        self.output_directory=Path(output_directory)
        self.output_directory.mkdir(parents=True,exist_ok=True,)
    def _save(self,filename):
        plt.tight_layout()
        plt.savefig(self.output_directory/filename,dpi=300,bbox_inches="tight",)
        plt.close()
    def boxplot(self,metric="STOI",):
        plt.figure(figsize=(10,6))
        models=sorted(self.df.model.unique())
        data=[self.df[self.df.model==m][metric] for m in models]
        plt.boxplot(data,tick_labels=models,)
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.title(f"{metric} distribution")
        self._save(f"boxplot_{metric}.png")
    
    def violin(self,metric="STOI",):
        plt.figure(figsize=(10,6))
        models=sorted(self.df.model.unique())
        data=[self.df[self.df.model==m][metric] for m in models]
        plt.violinplot(data,showmeans=True,showmedians=True,)
        plt.xticks(range(1,len(models)+1),models,)
        plt.ylabel(metric)
        plt.title(f"{metric} distribution")
        self._save(f"violin_{metric}.png")

    def mean_bar(self,metric="STOI",):
        means=(self.df.groupby("model")[metric].mean())
        plt.figure(figsize=(9,5))
        plt.bar(means.index,means.values,)
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.title(f"Average {metric}")
        self._save(f"mean_{metric}.png")

    def grouped_bar(self,metric="STOI",group="noise",):
        pivot=self.df.pivot_table(values=metric,index="model",columns=group,aggfunc="mean",)
        ax=pivot.plot(kind="bar",figsize=(10,6),)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} grouped by {group}")
        plt.xticks(rotation=20)
        self._save(f"grouped_{metric}_{group}.png")
    
    def line_by_snr(self,metric="STOI",):
        summary=(self.df.groupby(["snr","model",])[metric].mean().unstack())
        plt.figure(figsize=(10,6))
        for column in summary.columns:
            plt.plot(summary.index,summary[column],marker="o",label=column,)
        plt.xlabel("Input SNR")
        plt.ylabel(metric)
        plt.title(f"{metric} vs SNR")
        plt.legend()
        self._save(f"{metric}_vs_snr.png")
    
    def histogram(self,metric="STOI",bins=40,):
        plt.figure(figsize=(9,6))
        for model in sorted(self.df.model.unique()):
            values=self.df[self.df.model==model][metric]
            plt.hist(values,bins=bins,alpha=0.4,label=model,)
        plt.legend()
        plt.xlabel(metric)
        plt.ylabel("Count")
        plt.title(f"{metric} histogram")
        self._save(f"histogram_{metric}.png")
    
    def scatter(self,x="SNR",y="STOI",):
        plt.figure(figsize=(8,6))
        for model in sorted(self.df.model.unique()):
            subset=self.df[self.df.model==model]
            plt.scatter(subset[x],subset[y],alpha=0.5,label=model,)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        plt.title(f"{y} vs {x}")
        self._save(f"scatter_{x}_{y}.png")
    
    def generate_all(self):
        metrics=["SNR","SI_SDR","STOI",]
        for metric in metrics:
            self.boxplot(metric)
            self.violin(metric)
            self.mean_bar(metric)
            self.grouped_bar(metric)
            self.line_by_snr(metric)
            self.histogram(metric)
        self.scatter("SNR","STOI")
        self.scatter("SI_SDR","STOI")
        print("="*50)
        print("Plots saved to")
        print(self.output_directory)
        print("="*50)