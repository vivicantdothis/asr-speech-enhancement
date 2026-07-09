from pathlib import Path
import pandas as pd
from src.features.feature_config import FeatureConfig
from src.features.dataset_extractor import DatasetFeatureExtractor

def extract_dataset(metadata_csv,audio_root,feature_root,config,noise_type=None,snr=None,):
    metadata=pd.read_csv(metadata_csv)
    DatasetFeatureExtractor.process_dataframe(dataframe=metadata,audio_root=audio_root, output_root=feature_root,config=config,noise_type=noise_type,snr=snr,)
    
def main():
    config=FeatureConfig()
    splits=["train","val","test"]
    print("\nExtracting Clean Speech Features..\n")
    for split in splits:
        extract_dataset(metadata_csv=f"metadata/{split}.csv",audio_root=f"data/clean/{split}",feature_root=f"features/clean/{split}",config=config,)
    
    #white noise
    print("\nExtracting White Noise Features\n")
    for split in splits:
        extract_dataset(metadata_csv=f"metadata/{split}.csv",audio_root=f"data/noisy/white/{split}", feature_root=f"features/noisy/white/{split}",config=config,noise_type="white", snr=10,)
    #babble noise
    print("\nExtracting Babble Noise features\n")
    for split in splits:
        extract_dataset(metadata_csv=f"metadata/{split}.csv",audio_root=f"data/noisy/babble/{split}", feature_root=f"features/noisy/babble/{split}", config=config, noise_type="babble",snr=10,)
print("\nFeature extraction completed succesfully!")

if __name__=="__main__":
    main()

