"""
Configuration utilities for the FullSubNet integration.

This module centralizes every filesystem path used by the
FullSubNet wrapper so the rest of the pipeline never has to
hardcode locations.

Project structure assumed:

speech_enchancement/
│
├── pretrained_models/
│   └── fullsubnet/
│       └── fullsubnet_best_model_58epochs.tar
│
├── third_party/
│   └── FullSubNet/
│       ├── audio_zen/
│       ├── recipes/
│       └── ...
│
└── src/
    └── enhancement/
        └── pretrained/
            └── fullsubnet/
                config.py
"""

from pathlib import Path
import sys


###########################################################################
# Project Paths
###########################################################################

# config.py
THIS_FILE = Path(__file__).resolve()

# speech_enchancement/
PROJECT_ROOT = THIS_FILE.parents[4]

SRC_ROOT = PROJECT_ROOT / "src"

THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"

FULLSUBNET_ROOT = THIRD_PARTY_ROOT / "FullSubNet"

PRETRAINED_ROOT = PROJECT_ROOT / "pretrained_models"

FULLSUBNET_CHECKPOINT = (
    PRETRAINED_ROOT
    / "fullsubnet"
    / "fullsubnet_best_model_58epochs.tar"
)

RECIPE_ROOT = (
    FULLSUBNET_ROOT
    / "recipes"
    / "dns_interspeech_2020"
)

INFERENCE_CONFIG = (
    RECIPE_ROOT
    / "fullsubnet"
    / "inference.toml"
)


###########################################################################
# Audio Parameters
###########################################################################

SAMPLE_RATE = 16000

N_FFT = 512

WIN_LENGTH = 512

HOP_LENGTH = 256


###########################################################################
# Device
###########################################################################

DEFAULT_DEVICE = "cuda"


###########################################################################
# Validation
###########################################################################

def validate_installation():
    """
    Verify that every required FullSubNet component exists.

    Raises
    ------
    FileNotFoundError
        If the repository or checkpoint cannot be found.
    """

    if not FULLSUBNET_ROOT.exists():
        raise FileNotFoundError(
            f"FullSubNet repository not found:\n{FULLSUBNET_ROOT}"
        )

    if not FULLSUBNET_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{FULLSUBNET_CHECKPOINT}"
        )

    if not RECIPE_ROOT.exists():
        raise FileNotFoundError(
            f"Recipe directory missing:\n{RECIPE_ROOT}"
        )


###########################################################################
# Python Import Helper
###########################################################################

def add_fullsubnet_to_pythonpath():
    """
    Add the FullSubNet repository to sys.path once.

    This allows imports like

        from model import Model
        from audio_zen.acoustics.feature import stft

    without modifying the repository itself.
    """

    path = str(FULLSUBNET_ROOT)

    if path not in sys.path:
        sys.path.insert(0, path)


###########################################################################
# Convenience
###########################################################################

def print_configuration():

    print("=" * 60)
    print("FullSubNet Configuration")
    print("=" * 60)

    print(f"Project Root      : {PROJECT_ROOT}")
    print(f"Repository        : {FULLSUBNET_ROOT}")
    print(f"Checkpoint        : {FULLSUBNET_CHECKPOINT}")
    print(f"Recipe            : {RECIPE_ROOT}")
    print(f"Sample Rate       : {SAMPLE_RATE}")
    print(f"N FFT             : {N_FFT}")
    print(f"Hop Length        : {HOP_LENGTH}")
    print(f"Window Length     : {WIN_LENGTH}")

    print("=" * 60)


###########################################################################
# Automatically prepare imports
###########################################################################

add_fullsubnet_to_pythonpath()