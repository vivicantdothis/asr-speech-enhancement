from pathlib import Path
from src.enhancement.factory import build_enhancer
from src.enhancement.pipeline import EnhancementPipeline
from src.metrics.evaluator import Evaluator
from src.metrics.aggregate import MetricAggregator
from src.utils.audio import AudioIO

class ExperimentManager:
    def __init__(self,model_name,clean_root="data/clean/test",):
        self.model_name=model_name
        self.clean_root=Path(clean_root)
        self.output_root(Path("outputs")/model_name)
        self.evaluation_root=(Path("outputs")/"evaluation"/model_name)
        enhancer=build_enhancer(model_name)
        self.pipeline=EnhancementPipeline(enhancer)
        self.evaluator=Evaluator()
        self.report=EvaluationReport()