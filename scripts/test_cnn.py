from __future__ import annotations

import argparse
from pathlib import Path

from src.enhancement.classical.cnn.inference import CNNInference


def main():

    parser = argparse.ArgumentParser(
        description="Run CNN speech enhancement on a single audio file."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint (.pt)"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input noisy wav"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output enhanced wav"
    )

    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save predicted mask (.npy)"
    )

    parser.add_argument(
        "--save-spectrograms",
        action="store_true",
        help="Save clean/noisy/enhanced spectrogram figures"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CNN Speech Enhancement")
    print("=" * 60)

    inference = CNNInference(args.checkpoint)

    result = inference.process_file(
        input_path=args.input,
        output_path=args.output,
        save_mask=args.save_mask,
        save_spectrograms=args.save_spectrograms,
    )

    print()
    print("Enhancement complete.")
    print(f"Enhanced audio : {result['waveform_path']}")

    if result.get("mask_path") is not None:
        print(f"Mask           : {result['mask_path']}")

    if result.get("spectrogram_dir") is not None:
        print(f"Spectrograms   : {result['spectrogram_dir']}")

    print("=" * 60)


if __name__ == "__main__":
    main()