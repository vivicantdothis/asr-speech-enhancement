from pathlib import Path
import random
from src.enhancement.factory import build_enhancer
from src.enhancement.pipeline import EnhancementPipeline
from src.features.feature_config import FeatureConfig

def get_random_test_file():
    test_root=Path("data/clean/test")
    wav_files=list(test_root.rglob("*.wav"))
    if len(wav_files)==0:
        raise RuntimeError("No wav files found in data/clean/test")
    return random.choice(wav_files)

def main():
    config=FeatureConfig()
    enhancer=build_enhancer("cnn")
    pipeline=EnhancementPipeline(enhancer,config)
    input_file=get_random_test_file()
    output_root=Path("outputs")/"speech_enhancement"
    relative=input_file.relative_to(Path("data/clean/test"))
    output_file=(output_root/relative.parent/f"{relative.stem}_enhanced.wav")
    output_file.parent.mkdir(parents=True,exist_ok=True)
    speaker=input_file.parent.name
    dialect=input_file.parent.parent.name
    print(f"Dialect:{dialect}")
    print(f"Speaker:{speaker}")
    print(f"Sentence:{input_file.stem}")
    print(f"\nInput:{input_file}")
    print(f"Output:{output_file}\n")
    pipeline.enhance_file(input_file=input_file,output_file=output_file,)
    print("Enhancement completed successfully!")
if __name__=="__main__":
    main()