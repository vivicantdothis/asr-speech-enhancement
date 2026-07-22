"""
universal script for the generation of every comparison artifact for the speech
enhancement study.
it creates outputs/reports/tables-plots-radar-report.md
"""

from pathlib import Path
import pandas as pd

from src.visualization.tables import TableGenerator
from src.visualization.plots import PlotGenerator
from src.visualization.radar import RadarChartGenerator
from src.visualization.ranking import ModelRanker
from src.visualization.report import ReportGenerator

def load_results(results_root="outputs/evaluation"):
    results_root=Path(results_root)
    csv_files=sorted(results_root.rglob("results.csv"))
    if len(csv_files)==0:
        raise FileNotFoundError(f"No evaluation results found inside {results_root}")
    dfs=[]
    for csv in csv_files:
        df=pd.read_csv(csv)
        parts=csv.parts
        try:
            idx=parts.index("evaluation")
            df["model"]=parts[idx+1]
            df["noise"]=parts[idx+2]
            df["snr"]=parts[idx+3]
            df["split"]=parts[idx+4]
        except Exception:
            pass
        dfs.append(df)
    return pd.concat(dfs,ignore_index=True)

def main():
    print("="*50)
    print("Generating enhancemnet comparison report..")
    print("="*50)
    df=load_results()
    print(f"Loaded {len(df)} evaluated utterances.")
    tables=TableGenerator(df)
    tables.generate_all()
    plots=PlotGenerator(df)
    plots.generate_all()
    radar=RadarChartGenerator(df)
    radar.generate_all()
    ranking=ModelRanker(df)
    ranking.generate_all()
    report=ReportGenerator(df)
    report.generate_all()
    print()
    print("="*50)
    print("Everything finished successfully", "\nReports saved to outputs/reports/")
    print("="*50)

if __name__=="__main__":
    main()

