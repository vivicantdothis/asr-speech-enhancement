import argparse
from pathlib import Path
from tqdm import tqdm 
from src.metrics.evaluator import Evaluator
from src.metrics.report import EvaluationReport
from src.metrics.aggregate import MetricAggregator
from src.utils.audio import AudioIO

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default="deepfilternet",help="Model to evaluate",)
    args=parser.parse_args()
    MODEL_NAME=args.model
    clean_root=Path("data/clean/test")
    enhanced_root=Path("outputs")/MODEL_NAME
    evaluation_dir=(Path("outputs")/"evaluation"/MODEL_NAME)
    evaluator=Evaluator()
    report=EvaluationReport()
    wavs=sorted(clean_root.rglob("*.wav"))
    print("="*50)
    print(f"Model:{MODEL_NAME}")
    print(f"Files:{len(wavs)} ")
    print("="*50)
    processed=0
    failed=[]
    for clean_file in tqdm(wavs):
        try:
            relative=clean_file.relative_to(clean_root)
            enhanced_file=(enhanced_root/relative.parent/f"{relative.stem}_enhanced.wav")
            if not enhanced_file.exists():
                failed.append((relative.as_posix(),"Enhanced file missing",))
                continue
            clean,sr= AudioIO.load_audio(clean_file)
            enhanced,_=AudioIO.load_audio(enhanced_file)
            metrics=evaluator.evaluate(clean,enhanced,sr,)
            report.add(relative.as_posix(),MODEL_NAME,metrics)
            processed+=1
            if processed %100==0:
                print(f"Processed {processed}/{len(wavs)}")
        except Exception as e:
            print("\n")
            print("="*80)
            print("ERROR")
            print("="*80)
            print(f"File:{clean_file}")
            print(f"Type:{type(e).__name__}")
            print(f"Error:{e}")
            import traceback
            traceback.print_exc()
            failed.append((clean_file.as_posix(),str(e),))
    print(f"\nEvalauted files: {len(report.rows)}")
    df = report.save(evaluation_dir/"results.csv")
    summary=MetricAggregator.summarize(df)
    summary.to_csv(evaluation_dir/"summary.csv",index=False,)
    print("\n")
    print("="*50)
    print("Evaluation finished!")
    print("="*50)
    print(f"Processed:{processed}")
    print(f"Failed:{len(failed)}")
    print("="*50)
    if failed:
        log=(Path("logs")/f"{MODEL_NAME}_evaluation_failures.txt")
        log.parent.mkdir(parents=True,exist_ok=True,)
        with open(log,"w") as f:
            for wav, err in failed:
                f.write(f"{wav}\n")
                f.write(f"{err}\n\n")
                
if __name__=="__main__":
    main()
