from __future__ import annotations
from pathlib import Path
import pandas as pd

class TableGenerator:
    def __init__(self,dataframe:pd.DataFrame,output_directory="outputs/reports/tables",):
        self.df=dataframe.copy()
        self.output_directory=Path(output_directory)
        self.output_directory.mkdir(parents=True,exist_ok=True,)
    
    def summary_table(self):
        summary=(self.df.groupby("model")[["SNR","SI_SDR","STOI"]].agg(["mean","std"]).round(4))
        summary.to_csv(self.output_directory/"summary.csv")
        return summary
    
    def ranking_table(self):
        ranking=(self.df.groupby("model")[["SNR","SI_SDR","STOI"]].mean())
        ranking["Overall"]=(ranking["SNR"].rank(ascending=False)+ranking["SI_SDR"].rank(ascending=False)+ranking["STOI"].rank(ascending=False))
        ranking=(ranking.sort_values("Overall").round(4))
        ranking.to_csv(self.output_directory/"ranking.csv")
        return ranking
    
    def noise_summary(self):
        table=(self.df.groupby(["noise","model"])[["SNR","SI_SDR","STOI"]].mean().round(4))
        table.to_csv(self.output_directory/"noise_summary.csv")
        return table
    
    def snr_summary(self):
        table=(self.df.groupby(["snr","model"])[["SNR","SI_SDR","STOI"]].mean().round(4))
        table.to_csv(self.output_directory/"snr_summary.csv")
        return table
    
    def latex(self):
        summary=(self.df.groupby("model")[["SNR","SI_SDR","STOI"]].mean().round(3))
        with open(self.output_directory/"latex_table.tex","w",) as f:
            f.write(summary.to_latex())
    
    def markdown(self):
        summary=(self.df.groupby("model")[["SNR","SI_SDR","STOI"]].mean().round(3))
        with open(self.output_directory/"markdown_table.md","w",) as f:
            f.write(summary.to_markdown())

    def generate_all(self):
        self.summary_table()
        self.ranking_table()
        self.noise_summary()
        self.snr_summary()
        self.latex()
        self.markdown()
        print("="*50)
        print("Tables saved to: {self.output_directory}")
        print("="*50)