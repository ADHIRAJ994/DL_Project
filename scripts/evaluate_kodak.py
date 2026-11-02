
from pathlib import Path
import argparse
import csv
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_baseline import CNNAutoencoder
from models.snn_autoencoder import SNNAutoencoder
from src.dataset import get_image_files, image_to_tensor


def load_checkpoint(path, device):
    """Load a saved training checkpoint."""

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path.resolve()}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint format: {path}"
        )

    return checkpoint


def calculate_metrics(original, reconstructed):
    """Calculate MSE, PSNR and SSIM."""

    original = original.clamp(0, 1)
    reconstructed = reconstructed.clamp(0, 1)

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
        original.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    reconstructed_np = (
        reconstructed.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    ssim = structural_similarity(
        original_np,
        reconstructed_np,
        data_range=1.0,
        channel_axis=2
    )

    return mse, psnr, float(ssim)


def save_image(tensor, path):
    """Save a normalized RGB tensor as PNG."""

    image = (
        tensor.squeeze(0)
        .detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy() * 255.0
    )

    image = np.round(image).astype(np.uint8)

    Image.fromarray(image).save(path)


def evaluate_model(
    model,
    model_name,
    image_files,
    output_dir,
    device
):

    model.eval()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    print(f"\nEvaluating {model_name}...")

    with torch.inference_mode():

        for image_path in tqdm(image_files):

            # Load original image
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                original = image_to_tensor(image)

            original = original.unsqueeze(0).to(device)

            # Reconstruct image
            reconstructed = model(original)

            # Calculate metrics
            mse, psnr, ssim = calculate_metrics(
                original,
                reconstructed
            )

            # Save reconstruction
            reconstruction_path = (
                output_dir / image_path.name
            )

            save_image(
                reconstructed,
                reconstruction_path
            )

            rows.append({
                "image": image_path.name,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim
            })

    # Calculate average metrics
    average = {
        "image": "AVERAGE",
        "mse": np.mean([r["mse"] for r in rows]),
        "psnr": np.mean([r["psnr"] for r in rows]),
        "ssim": np.mean([r["ssim"] for r in rows])
    }

    rows.append(average)

    print(f"\n{model_name} Results")
    print("-" * 55)

    print(f"Average MSE : {average['mse']:.6f}")
    print(f"Average PSNR: {average['psnr']:.4f} dB")
    print(f"Average SSIM: {average['ssim']:.6f}")

    return rows


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kodak-dir",
        default="data/Kodak"
    )

    parser.add_argument(
        "--cnn-checkpoint",
        default="checkpoints/cnn_10epoch_best.pth"
    )

    parser.add_argument(
        "--snn-checkpoint",
        default="checkpoints/snn_10epoch_best.pth"
    )

    parser.add_argument(
        "--output-dir",
        default="results/kodak"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load Kodak images
    image_files = get_image_files(args.kodak_dir)

    print("Kodak test images:", len(image_files))

    if len(image_files) != 24:
        print("WARNING: Expected 24 Kodak images.")

    output_root = Path(args.output_dir)

    # --------------------------------
    # Evaluate CNN
    # --------------------------------

    cnn_checkpoint = load_checkpoint(
        Path(args.cnn_checkpoint),
        device
    )

    cnn_model = CNNAutoencoder().to(device)

    cnn_model.load_state_dict(
        cnn_checkpoint["model_state_dict"]
    )

    cnn_rows = evaluate_model(
        cnn_model,
        "CNN BASELINE",
        image_files,
        output_root / "cnn",
        device
    )

    # --------------------------------
    # Evaluate SNN
    # --------------------------------

    snn_checkpoint = load_checkpoint(
        Path(args.snn_checkpoint),
        device
    )

    snn_model = SNNAutoencoder(
        time_steps=int(
            snn_checkpoint.get("time_steps", 4)
        ),
        beta=float(
            snn_checkpoint.get("beta", 0.9)
        )
    ).to(device)

    snn_model.load_state_dict(
        snn_checkpoint["model_state_dict"]
    )

    snn_rows = evaluate_model(
        snn_model,
        "SNN MODEL",
        image_files,
        output_root / "snn",
        device
    )

    # --------------------------------
    # Save comparison CSV
    # --------------------------------

    csv_path = output_root / "comparison.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Model",
            "Image",
            "MSE",
            "PSNR",
            "SSIM"
        ])

        for model_name, rows in [
            ("CNN", cnn_rows),
            ("SNN", snn_rows)
        ]:

            for row in rows:

                writer.writerow([
                    model_name,
                    row["image"],
                    row["mse"],
                    row["psnr"],
                    row["ssim"]
                ])

    print("\nEvaluation completed.")

    print("CNN Average PSNR:", cnn_rows[-1]["psnr"])
    print("SNN Average PSNR:", snn_rows[-1]["psnr"])

    print("Results saved:", csv_path.resolve())


if __name__ == "__main__":
    main()
