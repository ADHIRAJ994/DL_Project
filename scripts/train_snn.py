
from pathlib import Path
import argparse
import math
import random
import sys

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from models.snn_autoencoder import SNNAutoencoder
from src.dataset import DIV2KPatchDataset


def set_seed(seed):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):

    model.eval()

    total_error = 0.0
    total_pixels = 0

    with torch.no_grad():

        for images in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            reconstructed = model(images).clamp(0, 1)

            error = F.mse_loss(
                reconstructed,
                images,
                reduction="sum"
            )

            total_error += error.item()
            total_pixels += images.numel()

    mean_mse = total_error / max(total_pixels, 1)

    psnr = 10 * math.log10(
        1.0 / max(mean_mse, 1e-12)
    )

    return mean_mse, psnr


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-dir",
        default="data/DIV2K/train_hr"
    )

    parser.add_argument(
        "--valid-dir",
        default="data/DIV2K/valid_hr"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--time-steps",
        type=int,
        default=4
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.9
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/snn_best.pth"
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")

    if args.time_steps < 1:
        parser.error("Time steps must be positive.")

    if not 0 < args.beta < 1:
        parser.error("Beta must be between 0 and 1.")

    set_seed(42)

    # Device selection
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("Time steps:", args.time_steps)
    print("LIF beta:", args.beta)

    # Dataset loading
    train_dataset = DIV2KPatchDataset(
        args.train_dir,
        patch_size=args.patch_size,
        training=True
    )

    valid_dataset = DIV2KPatchDataset(
        args.valid_dir,
        patch_size=max(args.patch_size, 256),
        training=False
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(valid_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # Initialize SNN
    model = SNNAutoencoder(
        time_steps=args.time_steps,
        beta=args.beta
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    best_val_mse = float("inf")

    checkpoint_path = Path(args.checkpoint)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Training
    for epoch in range(1, args.epochs + 1):

        model.train()

        total_loss = 0.0
        total_images = 0

        progress = tqdm(
            train_loader,
            desc=f"SNN Epoch {epoch}/{args.epochs}"
        )

        for images in progress:

            images = images.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            reconstructed = model(images)

            loss = F.mse_loss(
                reconstructed,
                images
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss encountered."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_images += batch_size

            progress.set_postfix(
                train_mse=f"{loss.item():.6f}"
            )

        train_mse = total_loss / max(total_images, 1)

        # Validation
        val_mse, val_psnr = evaluate(
            model,
            valid_loader,
            device
        )

        print(
            f"\nEpoch {epoch}: "
            f"Train MSE={train_mse:.6f} | "
            f"Validation MSE={val_mse:.6f} | "
            f"PSNR={val_psnr:.3f} dB"
        )

        # Save the best checkpoint
        if val_mse < best_val_mse:

            best_val_mse = val_mse

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "val_psnr": val_psnr,
                    "time_steps": args.time_steps,
                    "beta": args.beta,
                    "config": vars(args)
                },
                checkpoint_path
            )

            print(
                "Saved checkpoint:",
                checkpoint_path.resolve()
            )

    print("\nSNN training completed.")

    print(
        "Best validation MSE:",
        best_val_mse
    )

    print(
        "Checkpoint:",
        checkpoint_path.resolve()
    )

    print(
        "\nNOTE: This is an SNN reconstruction experiment, "
        "not yet an entropy-coded compression model."
    )


if __name__ == "__main__":
    main()
