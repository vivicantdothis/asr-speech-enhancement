from pathlib import Path
import numpy as np
from src.utils.audio import AudioIO
from src.enhancement.factory import build_enhancer
from src.metrics.evaluator import Evaluator

def stats(name,signal):
    print()
    print("+"*60)
    print(name)
    print("+"*60)
    print(f"Length:{len(signal)}")
    print(f"Peak: {np.max(np.abs(signal)):.5f}")
    print(f"RMS:{np.sqrt(np.mean(signal**2)):.5f}")
    print(f"Mean:{np.mean(signal):.5f}")

def main():
    clean_file=Path("data/clean/test/DR1/FAKS0/SA1.wav")
    noisy_file=Path("data/noisy/white/test/DR1/FAKS0/SA1_white_10dB.wav")
    output_file=Path("outputs/debug/SA1_enhanced.wav")
    clean,sr= AudioIO.load_audio(clean_file)
    noisy,_=AudioIO.load_audio(noisy_file)
    enhancer=build_enhancer("deepfilternet")
    enhanced=enhancer.enhance(noisy)
    AudioIO.save_audio(output_file,enhanced,sr,normalize=False,)
    stats("Clean",clean)
    stats("Noisy",noisy)
    stats("Enhanced",enhanced)
    evaluator=Evaluator()
    noisy_metrics=evaluator.evaluate(clean,noisy,sr,)
    enhanced_metrics=evaluator.evaluate(clean,enhanced,sr,)
    print()
    print("+"*60)
    print("NOISY")
    print("+"*60)
    for k,v in noisy_metrics.items():
        print(f"{k:10s}:{v}")
    print()
    print("+"*60)
    print("ENHANCED")
    print("+"*60)
    for k,v in enhanced_metrics.items():
        print(f"{k:10s}:{v}")
if __name__=="__main__":
    main()
