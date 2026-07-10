from pathlib import Path
import random
from src.enhancement.factory import build_enhancer
from src.enhancement.pipeline import EnhancementPipeline

def random_wav():
    files=list(Path("data/clean/test").rglob("*.wav"))
    if not files:
        raise RuntimeError("No wav files found.")
    return random.choice(files)

def main():
    enhancer=build_enhancer("deepfilternet")
    pipeline=EnhancementPipeline(enhancer)
    wav=random_wav()
    relative=wav.relative_to("data/clean/test")
    output=Path("outputs/deepfilternet")/relative
    output=output.with_name(output.stem+"_enhanced.wav")
    print(f"Input:{wav}")
    print(f"Output:{output}")
    pipeline.enhance_file(wav,output,)
    
if __name__=="__main__":
    main()