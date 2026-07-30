from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from .config import *
from .model import build_model
from src.visualization.spectrogram_exporter import SpectrogramExporter


class CNNInference:

    def __init__(self, checkpoint):

        self.device = DEVICE

        self.model = build_model()

        ckpt = torch.load(
            checkpoint,
            map_location=self.device,
        )

        self.model.load_state_dict(ckpt["model"])

        self.model.to(self.device)

        self.model.eval()

        self.exporter = SpectrogramExporter()

    #####################################################################
    # STFT
    #####################################################################

    @staticmethod
    def stft(audio):

        spec = librosa.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )

        magnitude = np.abs(spec)

        phase = np.angle(spec)

        if USE_LOG_MAGNITUDE:
            magnitude = np.log1p(magnitude)

        return magnitude, phase

    #####################################################################
    # ISTFT
    #####################################################################

    @staticmethod
    def istft(magnitude, phase):

        if USE_LOG_MAGNITUDE:
            magnitude = np.expm1(magnitude)

        spec = magnitude * np.exp(1j * phase)

        return librosa.istft(
            spec,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )

    #####################################################################
    # Enhance waveform
    #####################################################################

    def enhance(self, waveform):

        noisy_mag, phase = self.stft(waveform)

        tensor = (
            torch.from_numpy(noisy_mag)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            print(f"Input tensor shape: {tensor.shape}")
            predicted_mask = self.model(tensor)

        predicted_mask = (
            predicted_mask.squeeze()
            .cpu()
            .numpy()
        )

        predicted_mask = np.clip(
            predicted_mask,
            0.0,
            1.0,
        )

        enhanced_mag = noisy_mag * predicted_mask

        enhanced_waveform = self.istft(
            enhanced_mag,
            phase,
        )

        return {

            "waveform": enhanced_waveform.astype(np.float32),

            "mask": predicted_mask,

            "noisy_mag": noisy_mag,

            "enhanced_mag": enhanced_mag,

            "phase": phase,

        }

    #####################################################################
    # Save waveform
    #####################################################################

    @staticmethod
    def save_waveform(waveform, output):

        output = Path(output)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            output,
            waveform,
            SAMPLE_RATE,
        )

    #####################################################################
    # Export spectrograms
    #####################################################################

    def export_results(

        self,

        filename,

        noisy_mag,

        enhanced_mag,

        predicted_mask,

        clean_mag=None,

        method="cnn",

    ):

        self.exporter.export_all(

            method=method,

            filename=filename,

            clean=clean_mag,

            noisy=noisy_mag,

            enhanced=enhanced_mag,

            mask=predicted_mask,

        )

    #####################################################################
    # Main inference function
    #####################################################################

    def process_file(

        self,

        input_path,

        output_path,

        save_mask=False,

        save_spectrograms=False,

        clean_mag=None,

    ):

        input_path = Path(input_path)

        output_path = Path(output_path)

        waveform, _ = librosa.load(

            input_path,

            sr=SAMPLE_RATE,

            mono=True,

        )

        result = self.enhance(waveform)

        self.save_waveform(

            result["waveform"],

            output_path,

        )

        mask_path = None

        if save_mask:

            mask_path = output_path.with_suffix(".npy")

            np.save(

                mask_path,

                result["mask"],

            )

        spectrogram_dir = None

        if save_spectrograms:

            self.export_results(

                filename=output_path.stem,

                clean_mag=clean_mag,

                noisy_mag=result["noisy_mag"],

                enhanced_mag=result["enhanced_mag"],

                predicted_mask=result["mask"],

                method="cnn",

            )

            spectrogram_dir = Path(
                OUTPUT_ROOT,
                "spectrograms",
                "cnn",
            )

        return {

            "waveform_path": output_path,

            "mask_path": mask_path,

            "spectrogram_dir": spectrogram_dir,

            "waveform": result["waveform"],

            "mask": result["mask"],

            "noisy_mag": result["noisy_mag"],

            "enhanced_mag": result["enhanced_mag"],

        }