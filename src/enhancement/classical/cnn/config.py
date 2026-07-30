from __future__ import annotations
from pathlib import Path
import torch

PROJECT_ROOT=Path(__file__).resolve().parents[3]
DATA_ROOT=PROJECT_ROOT/"data"
OUTPUT_ROOT=PROJECT_ROOT/"outputs"
CHECKPOINT_ROOT=PROJECT_ROOT/"pretrained_models"/"cnn"
SPECTROGRAM_ROOT=OUTPUT_ROOT/"spectrograms"/"cnn"
LOG_ROOT=PROJECT_ROOT/"logs"
CHECKPOINT_ROOT.mkdir(parents=True,exist_ok=True)
SPECTROGRAM_ROOT.mkdir(parents=True,exist_ok=True)
LOG_ROOT.mkdir(parents=True,exist_ok=True)

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE=16000
NUM_CHANNELS=1
N_FFT=512
WIN_LENGTH=512
HOP_LENGTH=128
WINDOW="hann"
USE_LOG_MAGNITUDE=True
EPSILON=1e-8
NORMALIZE_INPUT=True
PATCH_FRAMES=256
SHUFFLE=True
NUM_WORKERS=4
PIN_MEMORY=True
BATCH_SIZE=8
EPOCHS=100
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-5
GRADIENT_CLIP=5.0
LR_PATIENCE=5
LR_FACTOR=0.5
MIN_LR=1e-6

###################
#U-NET ARCHITECTURE
###################

INPUT_CHANNELS=1
OUTPUT_CHANNELS=1
ENCODER_CHANNELS=[32,64,128,256,]
BOTTLENECK_CHANNELS=512
DROPOUT=0.20
USE_BATCH_NORM=True

###################
#Spectrogram Saving
###################

SAVE_SPECTROGRAMS=True
SAVE_NUMPY=True
SAVE_IMAGES=True
IMAGE_FORMAT="png"

############
#Checkpoints
############

SAVE_BEST_ONLY=True
CHECKPOINT_NAME="cnn_unet_best.pt"
LAST_CHECKPOINT="cnn_unet_last.pt"

############
#Random Seed
############

SEED=42

#LOSS WEIGHTS
MASK_LOSS_WEIGHT=0.5
SPECTROGRAM_LOSS_WEIGHT=0.5
SI_SDR_LOSS_WEIGHT=0.0
PESQ_LOSS_WEIGHT=0.0
STFT_LOSS_WEIGHT=0.0

