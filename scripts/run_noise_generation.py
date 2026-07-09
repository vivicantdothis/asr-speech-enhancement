import pandas as pd
from src.noise.noisy_dataset import NoisyDatasetGenerator

def main():
    train=pd.read_csv("metadata/train.csv")
    val=pd.read_csv("metadata/val.csv")
    test=pd.read_csv("metadata/test.csv")
    print("Generating White Noise..")
    NoisyDatasetGenerator.generate_white(train,"data/noisy/white/train")
    NoisyDatasetGenerator.generate_white(val,"data/noisy/white/val")
    NoisyDatasetGenerator.generate_white(test,"data/noisy/white/test")
    print("Generating Babble Noise..")
    NoisyDatasetGenerator.generate_babble(train,"data/noisy/babble/train")
    NoisyDatasetGenerator.generate_babble(val,"data/noisy/babble/val")
    NoisyDatasetGenerator.generate_babble(test,"data/noisy/babble/test")
    print("Done generating white and babble noise!")
if __name__=="__main__":
    main()