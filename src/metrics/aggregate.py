import pandas as pd

class MetricAggregator:
    @staticmethod
    def summarize(df):
        numeric=df.select_dtypes(include="number")
        summary=pd.DataFrame({"Mean":numeric.mean(),"Std":numeric.std(),"Min":numeric.min(),"Max":numeric.max(),})
        return summary
    