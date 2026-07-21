import argparse
import traceback
from pathlib import Path
from tqdm import tqdm 
from src.metrics.evaluator import Evaluator
from src.metrics.report import EvaluationReport
from src.metrics.aggregate import MetricAggregator
from src.utils.audio import AudioIO

def main():
    parser=argparse.ArgumentParser(description="evaluate enhanced speech against a clean reference <3")
    parser.add_argument("--model",default="deepfilternet",help="enhancement model",)
    parser.add_argument("--noise",required=True,choices=["white","babble"],help="noise type used during enhancement",)
    parser.add_argument("--snr",required=True,help="snr level (such as 10dB)")
    parser.add_argument("--split",default="test",choices=["train","test","val"],help="dataset split",)
    args=parser.parse_args()
    MODEL_NAME=args.model
    noise_type=args.noise
    split=args.split
    snr=args.snr
    clean_root=Path("data")/"clean"/split
    enhanced_root=(Path("outputs")/MODEL_NAME/noise_type/snr/split)
    evaluation_dir=(Path("outputs")/"evaluation"/MODEL_NAME/noise_type/snr/split)
    evaluation_dir.mkdir(parents=True,exist_ok=True)
    evaluator=Evaluator()
    report=EvaluationReport()
    wavs=sorted(clean_root.rglob("*.wav"))
    print("="*50)
    print(f"Model:{MODEL_NAME}")
    print(f"Noise:{noise_type}")
    print(f"Split:{split}")
    print(f"Clean root:{clean_root}")
    print(f"Enhanced:{enhanced_root}")
    print(f"Files:{len(wavs)} ")
    print("="*50)
    processed=0
    failed=[]
    for clean_file in tqdm(wavs):
        try:
            relative=clean_file.relative_to(clean_root)
            candidates=list((enhanced_root/relative.parent).glob(f"{relative.stem}*_enhanced.wav"))
            if len(candidates)==0:
                failed.append((relative.as_posix(),"enhanced file missing :("))
                continue
            enhanced_file=candidates[0]
            clean,sr= AudioIO.load_audio(clean_file)
            enhanced,_=AudioIO.load_audio(enhanced_file)
            metrics=evaluator.evaluate(clean,enhanced,sr,)
            report.add(relative.as_posix(),MODEL_NAME,metrics)
            processed+=1
            if processed and processed %100==0:
                print(f"Processed {processed}/{len(wavs)}")
        except Exception as e:
            print("\n"+"="*80)
            print("ERROR")
            print("="*80)
            print(f"File:{clean_file}")
            print(f"Type:{type(e).__name__}")
            print(f"Error:{e}")
            traceback.print_exc()
            failed.append((clean_file.as_posix(),str(e),))
    if len(report.rows)==0:
        print("\nNo files successfully evaluated.")
        return
    df = report.save(evaluation_dir/"results.csv")
    summary=MetricAggregator.summarize(df)
    summary.to_csv(evaluation_dir/"summary.csv",index=True,)
    print("\n")
    print("="*50)
    print("Evaluation finished!")
    print(f"\nEvalauted files: {len(report.rows)}")
    print("="*50)
    print(f"Processed:{processed}")
    print(f"Failed:{len(failed)}")
    print(f"Results:{evaluation_dir}")
    print("="*50)
    if failed:
        log=(Path("logs")/f"{MODEL_NAME}_{noise_type}_{split}_evaluation_failures.txt")
        log.parent.mkdir(parents=True,exist_ok=True,)
        with open(log,"w") as f:
            for wav, err in failed:
                f.write(f"{wav}\n")
                f.write(f"{err}\n\n")
                
if __name__=="__main__":
    main()
