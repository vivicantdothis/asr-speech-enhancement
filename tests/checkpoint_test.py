import torch

ckpt = torch.load(
    r"C:\Users\Archents\Desktop\speech_enchancement\pretrained_models\fullsubnet\fullsubnet_best_model_58epochs.tar",
    map_location="cpu",
)

print(type(ckpt))

if isinstance(ckpt, dict):
    print("\nKeys:")
    for k in ckpt.keys():
        print(k)