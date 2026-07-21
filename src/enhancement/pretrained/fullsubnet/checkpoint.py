"""
checkpoint.py

Loads a pretrained FullSubNet checkpoint and reconstructs the
network exactly as trained.

This file hides all checkpoint handling from the rest of the
speech enhancement pipeline.

Usage
-----

from src.enhancement.pretrained.fullsubnet.checkpoint import \
    load_pretrained_model

model = load_pretrained_model("cuda")
"""

from pathlib import Path
import torch

from .config import FULLSUBNET_CHECKPOINT,DEFAULT_DEVICE

# Import after config.py has added the repository to sys.path
from third_party.FullSubNet.recipes.dns_interspeech_2020.fullsubnet.model import Model as FullSubNetModel

class FullSubNetCheckpoint:
    """
    Handles loading pretrained FullSubNet models.
    """

    def __init__(self,checkpoint_path=None,device=DEFAULT_DEVICE,):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else FULLSUBNET_CHECKPOINT
        )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def load(self):
        """
        Returns
        -------
        model : FullSubNet Model
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n{self.checkpoint_path}"
            )

        print("=" * 60)
        print("Loading pretrained FullSubNet")
        print("=" * 60)

        print(f"Checkpoint : {self.checkpoint_path}")
        print(f"Device     : {self.device}")

        model = self._build_model()

        checkpoint=torch.load(self.checkpoint_path,map_location=self.device,)
        print("Checkpoint keys:", checkpoint.keys())
        if "model" in checkpoint:
            state_dict=checkpoint["model"]
        elif "l1" in checkpoint:
            state_dict=checkpoint["l1"]
        elif "state_dict" in checkpoint:
            state_dict=checkpoint["state_dict"]
        else:
            state_dict=checkpoint
        model.load_state_dict(state_dict)

        model.to(self.device)

        model.eval()

        print("Checkpoint successfully loaded.\n")

        return model

    # ------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------

    def _build_model(self):
        """
        Creates the FullSubNet architecture.

        Architecture matches the official
        DNS Challenge pretrained model.
        """

        model = FullSubNetModel(

            num_freqs=257,

            look_ahead=2,

            sequence_model="LSTM",

            fb_num_neighbors=0,

            sb_num_neighbors=15,

            fb_output_activate_function="ReLU",

            sb_output_activate_function=False,

            fb_model_hidden_size=512,

            sb_model_hidden_size=384,

            weight_init=False,

            norm_type="offline_laplace_norm",

            num_groups_in_drop_band=2,

        )
        return model


# ------------------------------------------------------------------
# Convenience Function
# ------------------------------------------------------------------

def load_pretrained_model(
    device=DEFAULT_DEVICE,
    checkpoint_path=None,
):
    """
    One-line API.

    Example
    -------

    model = load_pretrained_model("cuda")
    """

    loader = FullSubNetCheckpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return loader.load()