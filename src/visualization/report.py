from __future__ import annotations
from pathlib import Path
import pandas as pd
from .plots import PlotGenerator
from .tables import TableGenerator
from .ranking import ModelRanker
from .radar import RadarChartGenerator

class ReportGenerator:
    def __init__(self,dataframe:pd.DataFrame,output_directory="outputs/reports",):
        self.df=dataframe.copy()
        self.output_directory=Path(output_directory)
        self.output_directory.mkdir(parents=True,exist_ok=True,)
    
    def generate_tables(self):
        print("\nGenerating tables...")
        tables=TableGenerator(self.df,self.output_directory/"tables",)
        tables.generate_all()
    
    def generate_plots(self):
        print("\nGenerating plots...")
        plots=PlotGenerator(self.df,self.output_directory/"plots",)
        plots.generate_all()
    
    def generate_rankings(self):
        print("\nGenerating rankings...")
        ranking=ModelRanker(self.df,self.output_directory/"ranking",)
        ranking.generate_all()
    
    def generate_radar(self):
        print("\nGenerating radar charts...")
        radar=RadarChartGenerator(self.df, self.output_directory/"radar",)
        radar.generate_all()
    
    def dataset_summary(self):
        summary={}
        summary["Files"]=len(self.df)
        if "model" in self.df.columns:
            summary["Models"]=sorted(self.df.model.unique())
        if "noise" in self.df.columns:
            summary["Noise"]=sorted(self.df.noise.unique())
        if "snr" in self.df.columns:
            summary["SNR"]=sorted(self.df.noise.unique())
        summary_path=self.output_directory/"dataset_summary.txt"
        with open(summary_path,"w") as f:
            f.write("Speech Enhancement Evaluation\n")
            f.write("="*50+"\n\n")
            for key,value in summary.items():
                f.write(f"{key}:{value}\n")
        return summary
    
    def generate_all(self):
        print("="*50)
        print("Generating complete visualization report!")
        print("="*50)
        self.dataset_summary()
        self.generate_tables()
        self.generate_rankings()
        self.generate_plots()
        self.generate_radar()
        print("\n")
        print("="*50)
        print("Visualization report complete <3!")
        print(self.output_directory)
        print("="*50)