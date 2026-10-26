"""
Train and compare:

1. CNNBaseline
2. GLICPrototype (CNN + MyGFA)

IMPORTANT:
This is a reconstruction experiment.
It is NOT yet a full entropy-coded image compression model.

Metrics:
    - MSE
    - PSNR

The same train/test split is used for both models.
"""

from pathlib import Path
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from custom_compression import CNNBaseline, GLICPrototype


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path("test_images")

RESULT_DIR = Path("results/prototype")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 128

EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

TRAIN_RATIO = 0.78

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class ImageDataset(Dataset):

    def __init__(self, paths):
        self.paths = paths

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        path = self.paths[index]

        image = Image.open(path).convert("RGB")

        image = self.transform(image)

        return image, path.name


# ============================================================
# METRICS
# ============================================================

def calculate_mse(x, x_hat):

    return torch.mean(
        (x - x_hat) ** 2
    ).item()


def calculate_psnr(x, x_hat):

    mse = calculate_mse(
        x,
        x_hat,
    )

    if mse == 0:
        return float("inf")

    return 10 * math.log10(
        1.0 / mse
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
):

    model.eval()

    total_mse = 0.0
    total_psnr = 0.0

    count = 0

    for images, names in dataloader:

        images = images.to(DEVICE)

        reconstructed = model(images)

        reconstructed = reconstructed.clamp(
            0.0,
            1.0
        )

        mse = calculate_mse(
            images,
            reconstructed,
        )

        psnr = calculate_psnr(
            images,
            reconstructed,
        )

        total_mse += mse
        total_psnr += psnr

        count += 1

    return (
        total_mse / count,
        total_psnr / count,
    )


# ============================================================
# SAVE RECONSTRUCTIONS
# ============================================================

@torch.no_grad()
def save_reconstructions(
    model,
    dataloader,
    output_dir,
    prefix,
):

    model.eval()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for images, names in dataloader:

        images = images.to(DEVICE)

        reconstructed = model(images)

        reconstructed = reconstructed.clamp(
            0.0,
            1.0,
        )

        output = reconstructed.cpu().squeeze(0)

        output_image = transforms.ToPILImage()(
            output
        )

        output_image.save(
            output_dir /
            f"{prefix}_{names[0]}"
        )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model,
    model_name,
    train_loader,
    test_loader,
):

    print("\n" + "=" * 70)

    print(
        f"TRAINING: {model_name}"
    )

    print("=" * 70)

    model = model.to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_psnr = -float("inf")

    best_state = None

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0

        for images, _ in train_loader:

            images = images.to(
                DEVICE
            )

            optimizer.zero_grad()

            reconstructed = model(
                images
            )

            loss = criterion(
                reconstructed,
                images,
            )

            loss.backward()

            # Protect against unstable gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += loss.item()

        train_loss = (
            running_loss /
            len(train_loader)
        )

        test_mse, test_psnr = evaluate(
            model,
            test_loader,
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Loss {train_loss:.6f} | "
            f"Test MSE {test_mse:.6f} | "
            f"Test PSNR {test_psnr:.4f} dB"
        )

        # Save best model according to PSNR.
        if test_psnr > best_psnr:

            best_psnr = test_psnr

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

    # Restore best state.
    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # Save checkpoint.
    checkpoint_path = (
        RESULT_DIR /
        f"{model_name}_best.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    # Final evaluation.
    final_mse, final_psnr = evaluate(
        model,
        test_loader,
    )

    reconstruction_dir = (
        RESULT_DIR /
        model_name /
        "reconstructions"
    )

    save_reconstructions(
        model,
        test_loader,
        reconstruction_dir,
        model_name,
    )

    print(
        f"\nBest Test MSE : {final_mse:.6f}"
    )

    print(
        f"Best Test PSNR: {final_psnr:.4f} dB"
    )

    print(
        f"Checkpoint saved to:\n"
        f"{checkpoint_path}"
    )

    return {
        "model": model_name,
        "mse": final_mse,
        "psnr": final_psnr,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MYGFA RECONSTRUCTION EXPERIMENT")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    image_paths = sorted(
        [
            p
            for p in IMAGE_DIR.iterdir()
            if p.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }
        ]
    )

    if len(image_paths) < 3:

        raise RuntimeError(
            "Need at least 3 images."
        )

    # --------------------------------------------------------
    # Shuffle deterministically.
    # --------------------------------------------------------

    random.shuffle(
        image_paths
    )

    split_index = max(
        1,
        int(
            len(image_paths) *
            TRAIN_RATIO
        ),
    )

    train_paths = image_paths[
        :split_index
    ]

    test_paths = image_paths[
        split_index:
    ]

    # Ensure test set isn't empty.
    if not test_paths:

        test_paths = [
            train_paths.pop()
        ]

    print("\nTraining images:")

    for path in train_paths:

        print(
            f"  {path.name}"
        )

    print("\nTesting images:")

    for path in test_paths:

        print(
            f"  {path.name}"
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = ImageDataset(
        train_paths
    )

    test_dataset = ImageDataset(
        test_paths
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Model 1: CNN baseline
    # --------------------------------------------------------

    baseline_results = train_model(
        CNNBaseline(),
        "cnn_baseline",
        train_loader,
        test_loader,
    )

    # --------------------------------------------------------
    # Model 2: CNN + MyGFA
    # --------------------------------------------------------

    gfa_results = train_model(
        GLICPrototype(),
        "cnn_mygfa",
        train_loader,
        test_loader,
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<20}"
        f"{'MSE':>15}"
        f"{'PSNR (dB)':>15}"
    )

    print("-" * 70)

    for result in [
        baseline_results,
        gfa_results,
    ]:

        print(
            f"{result['model']:<20}"
            f"{result['mse']:>15.6f}"
            f"{result['psnr']:>15.4f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()