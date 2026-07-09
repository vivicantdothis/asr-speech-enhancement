from src.dataloaders.datamodule import build_loader

def main():
    loader=build_loader(metadata="metadata/train.csv",clean_root="features/clean/train",noisy_root="features/noisy/white/train",batch_size=8,shuffle=True,)
    batch=next(iter(loader))
    print("Input shape:", batch["input"].shape)
    print("Target shape:", batch["target"].shape)

if __name__=="__main__":
    main()
