
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.snn_autoencoder import SNNAutoencoder
from src.dataset import image_to_tensor


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path(
        "checkpoints/snn_t8_best.pth"
    )

    image_path = Path(
        "data/Kodak/kodim01.png"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = SNNAutoencoder(
        time_steps=checkpoint["time_steps"],
        beta=checkpoint["beta"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with Image.open(image_path) as im:
        image = image_to_tensor(
            im.convert("RGB")
        )

    image = image.unsqueeze(0).to(device)

    height, width = image.shape[-2:]

    # Pad dimensions to a multiple of four
    pad_h = (-height) % 4
    pad_w = (-width) % 4

    image = F.pad(
        image,
        (0, pad_w, 0, pad_h),
        mode="replicate"
    )

    with torch.inference_mode():

        spike_counts = model.encode_latent(image)

        reconstructed = model.decode_latent(
            spike_counts
        )

    reconstructed = reconstructed[
        :, :, :height, :width
    ].clamp(0, 1)

    mse = F.mse_loss(
        reconstructed,
        image[:, :, :height, :width]
    ).item()

    print("Device:", device)
    print("Input shape:", image.shape)
    print("Latent shape:", spike_counts.shape)
    print("Latent minimum:", spike_counts.min().item())
    print("Latent maximum:", spike_counts.max().item())
    print("Reconstructed shape:", reconstructed.shape)
    print("Reconstruction MSE:", mse)

    assert spike_counts.min().item() >= 0
    assert spike_counts.max().item() <= model.time_steps

    print("\nSNN codec components working successfully!")


if __name__ == "__main__":
    main()
