
from pathlib import Path
import argparse
import csv
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.snn_autoencoder import SNNAutoencoder
from src.dataset import get_image_files, image_to_tensor
from src.bitstream import save_bitstream, load_bitstream


def calculate_metrics(original, reconstructed):

    mse = F.mse_loss(
        original,
        reconstructed
    ).item()

    psnr = (
        float("inf")
        if mse == 0
        else 10 * math.log10(1.0 / mse)
    )

    original_np = (
        original[0].permute(1, 2, 0).cpu().numpy()
    )

    reconstructed_np = (
        reconstructed[0].permute(1, 2, 0).cpu().numpy()
    )

    ssim = structural_similarity(
        original_np,
        reconstructed_np,
        data_range=1.0,
        channel_axis=2
    )

    return mse, psnr, float(ssim)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/snn_t8_best.pth"
    )

    parser.add_argument(
        "--kodak-dir",
        default="data/Kodak"
    )

    parser.add_argument(
        "--output-dir",
        default="results/full_codec_kodak"
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

    # Load Kodak dataset
    image_files = get_image_files(args.kodak_dir)

    if not image_files:
        raise RuntimeError("No Kodak images found.")

    output_root = Path(args.output_dir)

    bitstream_dir = output_root / "bitstreams"
    reconstruction_dir = output_root / "reconstructions"

    bitstream_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Process each Kodak image
    for image_path in tqdm(image_files, desc="SNN Codec Evaluation"):

        with Image.open(image_path) as im:
            image = im.convert("RGB")
            original = image_to_tensor(image)

        original = original.unsqueeze(0).to(device)

        height, width = original.shape[-2:]

        # Pad dimensions to multiples of four
        pad_h = (-height) % 4
        pad_w = (-width) % 4

        padded = F.pad(
            original,
            (0, pad_w, 0, pad_h),
            mode="replicate"
        )

        # Encode image into integer spike counts
        with torch.inference_mode():

            latent = model.encode_latent(padded)

        latent_np = latent.cpu().numpy()

        # Store image and model metadata
        metadata = {
            "original_height": height,
            "original_width": width,
            "padded_height": padded.shape[-2],
            "padded_width": padded.shape[-1],
            "time_steps": int(checkpoint["time_steps"]),
            "beta": float(checkpoint["beta"])
        }

        bitstream_path = (
            bitstream_dir / f"{image_path.stem}.snnc"
        )

        # Create compressed file
        save_bitstream(
            latent_np,
            metadata,
            bitstream_path
        )

        # Read the actual compressed file
        decoded_latent_np, decoded_metadata = load_bitstream(
            bitstream_path
        )

        decoded_latent = torch.from_numpy(
            decoded_latent_np.astype(np.float32)
        ).to(device)

        # Reconstruct using decoded latent
        with torch.inference_mode():

            reconstructed = model.decode_latent(
                decoded_latent
            )

        reconstructed = reconstructed[
            :, :, :height, :width
        ].clamp(0, 1)

        # Calculate image quality
        mse, psnr, ssim = calculate_metrics(
            original,
            reconstructed
        )

        # Save reconstruction
        output_image = (
            reconstructed[0]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        output_image = np.clip(
            output_image * 255.0,
            0,
            255
        ).round().astype(np.uint8)

        Image.fromarray(output_image).save(
            reconstruction_dir / f"{image_path.stem}_reconstructed.png"
        )

        # Actual BPP includes header and metadata
        file_size = bitstream_path.stat().st_size

        total_pixels = height * width

        bpp = (file_size * 8) / total_pixels

        results.append({
            "image": image_path.name,
            "bitstream_bytes": file_size,
            "bpp": bpp,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim
        })

        print(
            f"\n{image_path.name}: "
            f"BPP={bpp:.4f}, "
            f"PSNR={psnr:.3f}, "
            f"SSIM={ssim:.4f}"
        )

    # Calculate dataset averages
    total_bits = sum(
        row["bitstream_bytes"] * 8 for row in results
    )

    total_pixels = sum(
        len(image_files[i].name) * 0 + 1
        for i in range(len(results))
    )

    # All Kodak images have the same dimensions,
    # but use per-image pixel counts for generality.
    pixel_count = 0

    for image_path in image_files:
        with Image.open(image_path) as im:
            w, h = im.size
            pixel_count += w * h

    average = {
        "image": "AVERAGE",
        "bitstream_bytes": sum(
            row["bitstream_bytes"] for row in results
        ),
        "bpp": total_bits / pixel_count,
        "mse": np.mean([r["mse"] for r in results]),
        "psnr": np.mean([r["psnr"] for r in results]),
        "ssim": np.mean([r["ssim"] for r in results])
    }

    results.append(average)

    # Save CSV
    csv_path = output_root / "snn_codec_results.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "bitstream_bytes",
                "bpp",
                "mse",
                "psnr",
                "ssim"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 55)
    print("FINAL SNN CODEC RESULTS")
    print("=" * 55)

    print(f"Images evaluated: {len(image_files)}")
    print(f"Average BPP: {average['bpp']:.6f}")
    print(f"Average MSE: {average['mse']:.6f}")
    print(f"Average PSNR: {average['psnr']:.4f} dB")
    print(f"Average SSIM: {average['ssim']:.6f}")

    print("\nResults saved:", csv_path.resolve())


if __name__ == "__main__":
    main()
