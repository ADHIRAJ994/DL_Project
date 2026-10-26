"""
Visualize the GFA behavior of the official GLIC implementation.

Outputs:
    original.png
    reconstructed.png
    stage1_complexity.png
    stage1_local_degree.png
    stage1_global_degree.png
    stage2_complexity.png
    stage2_local_degree.png
    stage2_global_degree.png
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from glic_model import GLICModel
from glic_model.utils.graph_utils import compute_sobel_gradients


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = Path("0.05checkpoint_best.pth.tar")
IMAGE_PATH = Path("test_images/landscape.jpg")
OUTPUT_DIR = Path("results/gfa_visualization")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# IMAGE UTILITIES
# ============================================================

def load_image(path: Path):
    """
    Load an RGB image and convert it to a PyTorch tensor.

    Output shape:
        [1, 3, H, W]

    Pixel values are normalized to [0, 1].
    """
    image = Image.open(path).convert("RGB")

    tensor = (
        torch.from_numpy(__import__("numpy").array(image))
        .float()
        / 255.0
    )

    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return tensor


def pad_to_multiple(x, multiple=128):
    """
    GLIC's test script pads images to a multiple of 128.

    We use replicate padding so the image content is not
    artificially padded with black pixels.
    """
    _, _, h, w = x.shape

    padded_h = ((h + multiple - 1) // multiple) * multiple
    padded_w = ((w + multiple - 1) // multiple) * multiple

    pad_bottom = padded_h - h
    pad_right = padded_w - w

    x = F.pad(
        x,
        (0, pad_right, 0, pad_bottom),
        mode="replicate"
    )

    return x, (h, w)


def crop_to_original(x, original_size):
    """
    Remove padding after reconstruction.
    """
    h, w = original_size
    return x[..., :h, :w]


# ============================================================
# VISUALIZATION
# ============================================================

def save_image_tensor(tensor, path):
    """
    Save a tensor with shape [1, 3, H, W] as an RGB PNG.
    """
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = tensor.clamp(0, 1)

    image = tensor.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_heatmap(values, path, title):
    """
    Save a 2D tensor as a heatmap.
    """
    values = values.detach().cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(values)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# GRAPH VISUALIZATION
# ============================================================

def graph_degree_map(graph, batch_size, height, width, window_size):
    """
    Convert the GFA graph mask into a spatial degree map.

    graph shape is approximately:

        [number_of_windows, 1, window_size^2, candidates]

    For every query node, count the number of selected
    neighbors.
    """

    # Remove attention-head dimension.
    graph = graph.squeeze(1)

    # Count selected neighbors.
    degree = graph.sum(dim=-1)

    # Number of windows in H and W.
    numh = height // window_size
    numw = width // window_size

    # Restore:
    #
    # [windows, window_nodes]
    #
    # into:
    #
    # [H, W]
    degree = degree.view(
        batch_size,
        numh,
        numw,
        window_size,
        window_size,
    )

    degree = degree.permute(
        0, 1, 3, 2, 4
    ).contiguous()

    degree = degree.view(
        batch_size,
        height,
        width
    )

    return degree


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("GLIC GFA VISUALIZATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Image: {IMAGE_PATH}")

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_image(IMAGE_PATH)

    original_h = image.shape[-2]
    original_w = image.shape[-1]

    print(
        f"Original image size: "
        f"{original_w} x {original_h}"
    )

    image_padded, original_size = pad_to_multiple(
        image,
        multiple=128
    )

    print(
        f"Padded image size: "
        f"{image_padded.shape[-1]} x "
        f"{image_padded.shape[-2]}"
    )

    image_padded = image_padded.to(DEVICE)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = GLICModel().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present.
    if state_dict and all(
        key.startswith("module.")
        for key in state_dict.keys()
    ):
        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.update()
    model.eval()

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    save_image_tensor(
        image,
        OUTPUT_DIR / "original.png"
    )

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    print("\nRunning GLIC reconstruction...")

    with torch.no_grad():

        output = model(image_padded)

        reconstruction = output["x_hat"].clamp(
            0.0,
            1.0
        )

        reconstruction = crop_to_original(
            reconstruction,
            original_size
        )

    save_image_tensor(
        reconstruction,
        OUTPUT_DIR / "reconstructed.png"
    )

    print("Reconstruction saved.")

    # --------------------------------------------------------
    # Get intermediate encoder features
    # --------------------------------------------------------

    print("\nExtracting encoder feature maps...")

    with torch.no_grad():

        (
            final_latent,
            stage0,
            stage1,
            stage2
        ) = model.g_a.forward_energy(
            image_padded
        )

    print(
        f"Stage 0 shape: {tuple(stage0.shape)}"
    )

    print(
        f"Stage 1 shape: {tuple(stage1.shape)}"
    )

    print(
        f"Stage 2 shape: {tuple(stage2.shape)}"
    )

    # --------------------------------------------------------
    # Visualize both GFA stages
    # --------------------------------------------------------

    stages = [
        ("stage1", stage1, model.g_a.g2),
        ("stage2", stage2, model.g_a.g3),
    ]

    for name, feature_map, gfa in stages:

        print("\n" + "-" * 60)
        print(f"Processing {name}")
        print("-" * 60)

        B, C, H, W = feature_map.shape

        # GFA expects:
        #
        # [B, H*W, C]
        #
        flattened = (
            feature_map
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        # ----------------------------------------------------
        # Complexity / RMS gradient
        # ----------------------------------------------------

        complexity = compute_sobel_gradients(
            flattened,
            shape=(H, W)
        )

        save_heatmap(
            complexity[0],
            OUTPUT_DIR / f"{name}_complexity.png",
            f"{name} RMS Gradient Complexity"
        )

        print(
            f"{name} complexity range: "
            f"{complexity.min().item():.4f} "
            f"to "
            f"{complexity.max().item():.4f}"
        )

        # ----------------------------------------------------
        # Construct local + global graphs
        # ----------------------------------------------------

        print(f"Building {name} graphs...")

        with torch.no_grad():

            graph_local, graph_global = (
                gfa._calc_graph(
                    flattened,
                    (H, W)
                )
            )

        # ----------------------------------------------------
        # Degree maps
        # ----------------------------------------------------

        local_degree = graph_degree_map(
            graph_local,
            B,
            H,
            W,
            gfa.window_size
        )

        global_degree = graph_degree_map(
            graph_global,
            B,
            H,
            W,
            gfa.window_size
        )

        save_heatmap(
            local_degree[0],
            OUTPUT_DIR / f"{name}_local_degree.png",
            f"{name} Local Graph Degree"
        )

        save_heatmap(
            global_degree[0],
            OUTPUT_DIR / f"{name}_global_degree.png",
            f"{name} Global Graph Degree"
        )

        print(
            f"Local degree: "
            f"{local_degree.min().item():.1f} "
            f"to "
            f"{local_degree.max().item():.1f}"
        )

        print(
            f"Global degree: "
            f"{global_degree.min().item():.1f} "
            f"to "
            f"{global_degree.max().item():.1f}"
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()