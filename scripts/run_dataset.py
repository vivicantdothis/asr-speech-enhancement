from pathlib import Path
import pandas as pd
from src.datasets.timit_dataset import (TIMITScanner,TIMITMetadataBuilder)
from src.datasets.split_generator import (SpeakerSplitGenerator)
from src.datasets.dataset_report import (DatasetReport)
from src.datasets.preprocess_clean import (CleanAudioProcessor)
from src.datasets.clean_dataset import CleanDatasetGenerator

def main():
    scanner=TIMITScanner("datasets/TIMIT")
    wavs=scanner.find_wavs()
    metadata=TIMITMetadataBuilder.build(wavs)
    print(metadata.head())
    print(metadata.columns)
    train_metadata=metadata[metadata["filepath"].str.contains("TRAIN")]
    test_metadata=metadata[metadata["filepath"].str.contains("TEST")]
    train_df,val_df=(SpeakerSplitGenerator.create_train_val_split(train_metadata))
    test_df=test_metadata
    Path("metadata").mkdir(exist_ok=True)
    train_df.to_csv("metadata/train.csv",index=False)
    val_df.to_csv("metadata/val.csv", index=False)
    test_df.to_csv("metadata/test.csv", index=False)
    CleanAudioProcessor.process_split(train_df,"data/clean/train")
    CleanAudioProcessor.process_split(val_df,"data/clean/val")
    CleanAudioProcessor.process_split(test_df,"data/clean/test")
    report=DatasetReport.generate(train_df,val_df,test_df)
    DatasetReport.save(report,"metadata/dataset_report.json")
    print(report)
    print("\nGenerating clean training set..")
    CleanDatasetGenerator.generate(train_df,"data/clean/train")
    print("\nGenerating clean validation set..")
    CleanDatasetGenerator.generate(val_df,"data/clean/val")
    print("\nGenerating clean test set...")
    CleanDatasetGenerator.generate(test_df,"data/clean/test")

if __name__ == "__main__":
    main()