from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    destination=Path("pretrained_models")/"metricgan_plus"
    destination.mkdir(parents=True,exist_ok=True)
    print("="*50)
    print("Dwonlaoding MetricGan+")
    print("="*50)
    snapshot_download(repo_id="speechbrain/metricgan-plus-voicebank",local_dir=destination,local_dir_use_symlinks=False,)
    print("\nDownload complete!")
    print(destination.resolve())

if __name__=="__main__":
    main()