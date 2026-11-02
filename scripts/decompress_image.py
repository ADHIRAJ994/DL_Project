
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.snn_autoencoder import SNNAutoencoder
from src.bitstream import load_bitstream


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/snn_t8_best.pth"
    )

    parser.add_argument(
        "--output",
        default="results/reconstructions/reconstructed.png"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Read compressed bitstream
    latent_np, metadata = load_bitstream(args.input)

    print("Decoded latent shape:", latent_np.shape)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    # Check model configuration
    if int(metadata["time_steps"]) != int(checkpoint["time_steps"]):
        raise ValueError("Checkpoint time steps do not match bitstream.")

    if not np.isclose(
        float(metadata["beta"]),
        float(checkpoint["beta"])
    ):
        raise ValueError("Checkpoint beta does not match bitstream.")

    # Load SNN architecture
    model = SNNAutoencoder(
        time_steps=int(checkpoint["time_steps"]),
        beta=float(checkpoint["beta"])
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Convert decoded integer counts to tensor
    latent = torch.from_numpy(
        latent_np.astype(np.float32)
    ).to(device)

    # Validate spike counts
    if latent.min().item() < 0:
        raise ValueError("Negative spike count detected.")

    if latent.max().item() > model.time_steps:
        raise ValueError("Spike count exceeds time steps.")

    # Reconstruct image
    with torch.inference_mode():

        reconstructed = model.decode_latent(latent)

    # Remove padding
    height = int(metadata["original_height"])
    width = int(metadata["original_width"])

    reconstructed = reconstructed[
        :, :, :height, :width
    ]

    reconstructed = reconstructed.clamp(0, 1)

    # Save RGB image
    array = (
        reconstructed[0]
        .cpu()
        .permute(1, 2, 0)
        .numpy() * 255.0
    )

    array = np.rint(array).astype(np.uint8)

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image.fromarray(array).save(output_path)

    print("\nDecompression completed.")
    print("Image dimensions:", width, "x", height)
    print("Output:", output_path.resolve())


if __name__ == "__main__":
    main()
