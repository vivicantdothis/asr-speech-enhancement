import argparse
import time
from pathlib import Path
from tqdm import tqdm 
from src.enhancement.factory import build_enhancer
from src.enhancement.pipeline import EnhancementPipeline

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", default="deepfilternet",help="Enhancement model",)
    args=parser.parse_args()
    start=time.time()
    MODEL_NAME=args.model
    input_root=Path("data/clean/test")
    output_root=Path("outputs")/MODEL_NAME
    enhancer=build_enhancer(MODEL_NAME)
    pipeline=EnhancementPipeline(enhancer)
    wav_files=sorted(input_root.rglob("*.wav"))
    print("="*50)
    print(f"Model:{MODEL_NAME}")
    print(f"Input:{input_root}")
    print(f"Output:{output_root}")
    print(f"Files: {len(wav_files)} .wav files.")
    print("="*50)
    success=0
    failed=[]
    for wav in tqdm(wav_files):
        relative=wav.relative_to(input_root)
        output_file=(output_root/relative.parent/f"{relative.stem}_enhanced.wav")
        if output_file.exists():
            continue
        try:
            pipeline.enhance_file(wav,output_file,)
            success += 1
        except Exception as e:
            failed.append((wav,str(e)))
    elapsed=time.time()-start
    print("="*50)
    print("\nBatch enhancement completed.")
    print(f"Succesfull:{success}")
    print(f"Failed:{len(failed)}")
    if failed:
        print("\nFailed files:")
        for wav,err in failed:
            print(f"{wav}")
            print(err)
    print(f"Outputs saved to: {output_root}")
    print(f"Runtime: {elapsed/60:.2f} minutes")
    print("="*50)

    log = Path("logs")/f"{MODEL_NAME}_failures.txt"
    if failed:
        with open(log,"w") as f:
            for wav,err in failed:
                f.write(f"{wav}\n")
                f.write(f"{err}\n\n")

if __name__=="__main__":
    main()