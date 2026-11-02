
import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.snn_autoencoder import SNNAutoencoder
from src.dataset import image_to_tensor
from src.bitstream import save_bitstream


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
        default="results/bitstreams/image.snnc"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load checkpoint
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    model = SNNAutoencoder(
        time_steps=int(checkpoint["time_steps"]),
        beta=float(checkpoint["beta"])
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Load original image
    input_path = Path(args.input)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with Image.open(input_path) as im:

        image = im.convert("RGB")

        original_width, original_height = image.size

        tensor = image_to_tensor(image)

    tensor = tensor.unsqueeze(0).to(device)

    # Pad dimensions to a multiple of 4
    pad_h = (-original_height) % 4
    pad_w = (-original_width) % 4

    tensor = F.pad(
        tensor,
        (0, pad_w, 0, pad_h),
        mode="replicate"
    )

    padded_height = tensor.shape[-2]
    padded_width = tensor.shape[-1]

    # Extract integer spike counts
    with torch.inference_mode():

        latent = model.encode_latent(tensor)

    latent = latent.cpu().numpy()

    # Save bitstream
    metadata = {

        "original_height": original_height,
        "original_width": original_width,

        "padded_height": padded_height,
        "padded_width": padded_width,

        "time_steps": int(checkpoint["time_steps"]),
        "beta": float(checkpoint["beta"])
    }

    result = save_bitstream(
        latent,
        metadata,
        args.output
    )

    # Calculate genuine BPP using complete file size
    total_bits = result["file_size"] * 8

    total_pixels = (
        original_height * original_width
    )

    bpp = total_bits / total_pixels

    raw_rgb_bytes = (
        original_height *
        original_width *
        3
    )

    compression_ratio = (
        raw_rgb_bytes / result["file_size"]
    )

    print("\nCompression completed")
    print("------------------------------")

    print("Input:", input_path.name)
    print("Latent shape:", latent.shape)

    print("Original RGB size:", raw_rgb_bytes, "bytes")

    print(
        "Raw latent size:",
        result["raw_size"],
        "bytes"
    )

    print(
        "Compressed payload:",
        result["compressed_payload_size"],
        "bytes"
    )

    print(
        "Metadata:",
        result["metadata_size"],
        "bytes"
    )

    print(
        "Complete bitstream:",
        result["file_size"],
        "bytes"
    )

    print(f"BPP: {bpp:.6f}")

    print(
        f"Raw RGB / bitstream size ratio: "
        f"{compression_ratio:.4f}"
    )

    print("\nSaved:", Path(args.output).resolve())


if __name__ == "__main__":
    main()
