"""
creates radar(spider) charts comparing the various speech enhancement models
used throughout this project.

the metrics are automatically normalized so different scales can be visualized together. 
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RadarChartGenerator:
    DEFAULT_METRICS=["SNR","SI_SDR","STOI",]
    def __init__(self,dataframe:pd.DataFrame,output_directory="outputs/reports/radar",):
        self.df=dataframe.copy()
        self.output_directory=Path(output_directory)
        self.output_directory.mkdir(parents=True,exist_ok=True,)

    def metric_table(self,metrics=None,):
        metrics=metrics or self.DEFAULT_METRICS
        return (self.df.groupby("model")[metrics].mean())
    
    def normalize(self,table):
        normalized=table.copy()
        for column in normalized.columns:
            minimum=normalized[column].min()
            maximum=normalized[column].max()
            if np.isclose(maximum,minimum):
                normalized[column]=1.0
            else:
                normalized[column]=((normalized[column]-minimum)/(maximum-minimum))
        return normalized
    
    def radar(self,metrics=None,filename="radar_chart.png",normalize=True,):
        metrics=metrics or self.DEFAULT_METRICS
        table=self.metric_table(metrics)
        if normalize:
            table=self.normalize(table)
        categories=list(table.columns)
        num_vars=len(categories)
        angles=np.linspace(0,2*np.pi,num_vars,endpoint=False,).tolist()
        angles += angles[:1]
        fig=plt.figure(figsize=(8,8))
        ax=plt.subplot(111,polar=True,)
        for model in table.index:
            values=table.loc[model].tolist()
            values+=values[:1]
            ax.plot(angles,values,linewidth=2,label=model,)
            ax.fill(angles,values,alpha=0.15)
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0,1)
        ax.set_title("Speech Enhancement Model Comparison",fontsize=14,)
        plt.legend(loc="upper right",bbox_to_anchor=(1.35,1.1),)
        plt.tight_layout()
        plt.savefig(self.output_directory/filename,dpi=300,bbox_inches="tight",)
        plt.close()

    def radar_per_noise(self,metrics=None,):
        metrics=metrics or self.DEFAULT_METRICS
        if "noise" not in self.df.columns:
            return
        for noise in sorted(self.df.noise.unique()):
            subset=self.df[self.df.noise==noise]
            generator=RadarChartGenerator(subset,self.output_directory,)
            generator.radar(metrics=metrics,filename=f"radar_{noise}.png",)
        
    def radar_per_snr(self,metrics=None,):
        metrics=metrics or self.DEFAULT_METRICS
        if "snr" not in self.df.columns:
            return
        for snr in sorted(self.df.snr.unique()):
            subset=self.df[self.df.snr==snr]
            generator=RadarChartGenerator(subset,self.output_directory,)
            safe_name=str(snr).replace("/","_")
            generator.radar(metrics=metrics,filename=f"radar_{safe_name}.png",)

    
    def generate_all(self):
        self.radar()
        self.radar_per_noise()
        self.radar_per_snr()
        print("="*60)
        print("Radar charts generated.")
        print(self.output_directory)
        print("="*60)