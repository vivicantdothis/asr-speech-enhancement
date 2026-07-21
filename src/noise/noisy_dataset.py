import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.utils.audio import AudioIO
from src.utils.file_utils import FileUtils
from src.noise.noise_generator import NoiseGenerator
from src.noise.babble_generator import BabbleGenerator
from src.noise.snr_mixer import SNRMixer

class NoisyDatasetGenerator:
    @staticmethod
    def generate_white(df,output_dir):
        snr_results=[]
        output_dir=Path(output_dir)
        progress=tqdm(df.iterrows(),total=len(df),desc="Generating White Noise")
        for _,row in progress:
            try:
                audio,sr=AudioIO.load_audio(row["filepath"])
                noise=NoiseGenerator.white(len(audio))
                noisy=SNRMixer.mix(audio,noise,snr_db=10)
                measured_snr=SNRMixer.compute_snr(audio,noisy)
                snr_results.append({"filepath":row["filepath"], "noise_type":"white", "target_snr":10, "measured_snr":measured_snr})
                relative_path=FileUtils.relative_timit_path(row["filepath"])
                relative_path=FileUtils.noisy_filename(relative_path,noise_type="white",snr=10)
                save_path=output_dir/relative_path
                AudioIO.save_audio(save_path,noisy,sr,normalize=False)
            except Exception as e:
                print(f"Skipping {row['filepath']}")
                print(e)
        results=pd.DataFrame(snr_results)
        Path("metadata").mkdir(exist_ok=True)
        results.to_csv("metadata/white_snr_validation.csv",index=False)
        print("\n"+"="*60)
        print("White noise validation:")
        print("="*60)
        print(results["measured_snr"].describe())
        print(f"\nMean Absolute SNR Error:" f"{abs(results['measured_snr']-10).mean():.3f} dB")


    @staticmethod
    def generate_babble(df,output_dir):
        snr_results=[]
        output_dir=Path(output_dir)
        progress=tqdm(df.iterrows(),total=len(df),desc="Generating Babble Noise")
        generator=BabbleGenerator(df)
        for _,row in progress:
            try:
                audio,sr=AudioIO.load_audio(row["filepath"])
                babble=generator.generate(target_length=len(audio),exclude_speaker=row["speaker_id"],)
                noisy=SNRMixer.mix(audio,babble,snr_db=10)
                measured_snr=SNRMixer.compute_snr(audio,noisy)
                snr_results.append({"filepath":row["filepath"], "noise_type":"babble", "target_snr":10, "measured_snr":measured_snr})
                relative_path=FileUtils.relative_timit_path(row["filepath"])
                relative_path=FileUtils.noisy_filename(relative_path,noise_type="babble",snr=10)
                save_path=output_dir/relative_path
                AudioIO.save_audio(save_path,noisy,sr,normalize=False)
            except Exception as e:
                print(f"Skipping {row['filepath']}")
                print(e)
        results=pd.DataFrame(snr_results)
        Path("metadata").mkdir(exist_ok=True)
        results.to_csv("metadata/babble_snr_validation.csv", index=False)
        print("\n"+"="*60)
        print("Babble Noise Validation")
        print("="*60)
        print(results["measured_snr"].describe())
        print(f"\nMean Absolute SNR Error: " f"{abs(results['measured_snr']-10).mean():.3f} dB")

