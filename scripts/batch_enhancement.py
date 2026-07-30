import argparse
import time
from pathlib import Path
from tqdm import tqdm 
from src.enhancement.factory import build_enhancer
from src.enhancement.pipeline import EnhancementPipeline

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", default="deepfilternet",help="Enhancement model",)
    parser.add_argument("--noise",required=True,choices=["white","babble"],help="Noise type",)
    parser.add_argument("--snr",required=True,help="snr level")
    parser.add_argument("--split",default="test",choices=["train","val","test"],)
    parser.add_argument("--overwrite",action="store_true",help="Overwrite existing enhanced files",)
    args=parser.parse_args()
    start=time.time()
    MODEL_NAME=args.model
    snr=args.snr
    input_root=Path("data")/"noisy"/args.noise/args.split
    output_root=Path("outputs")/MODEL_NAME/args.noise/snr/args.split
    enhancer=build_enhancer(MODEL_NAME,noise_type=args.noise,)
    pipeline=EnhancementPipeline(enhancer)
    wav_files=sorted(input_root.rglob("*.wav"))
    print("="*50)
    print(f"Model:{MODEL_NAME}")
    print(f"Noise:{args.noise}")
    print(f"Split:{args.split}")
    print(f"Input:{input_root}")
    print(f"Output:{output_root}")
    print(f"Files: {len(wav_files)} .wav files.")
    print("="*50)
    success=0
    skipped=0
    failed=[]
    for wav in tqdm(wav_files):
        relative=wav.relative_to(input_root)
        output_file=(output_root/relative.parent/f"{relative.stem}_enhanced.wav")
        if output_file.exists() and not args.overwrite:
            skipped +=1
            continue
        try:
            pipeline.enhance_file(wav,output_file,)
            if MODEL_NAME.lower()=="cnn":
                try:
                    pipeline.export_debug(wav,output_file,)
                except Exception as e:
                    print(f"Spectrogram export failed:{relative}")
                    print(e)
            success += 1
        except Exception as e:
            failed.append((wav,str(e)))
    elapsed=time.time()-start
    print("="*50)
    print("\nBatch enhancement completed.")
    print(f"Succesful:{success}")
    print(f"Skipped:{skipped}")
    print(f"Failed:{len(failed)}")
    print(f"Runtime:{elapsed/60:.2f} min")
    print(f"Outputs saved to: {output_root}")
    print("="*50)

    if failed:
        log=Path("logs")/f"{MODEL_NAME}_{args.noise}_{args.split}_enhancement_failures.txt"
        with open(log,"w") as f:
            for wav,err in failed:
                f.write(f"{wav}\n")
                f.write(f"{err}\n\n")

if __name__=="__main__":
    main()