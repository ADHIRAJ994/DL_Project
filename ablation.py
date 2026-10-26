"""
GLIC Graph Ablation Experiment

Compares:

1. Official GLIC
   - alternating Local + Global graph operations

2. Local-only
   - every GFA block uses local graph sampling

3. Global-only
   - every GFA block uses global graph sampling

IMPORTANT:
These are inference-time ablations using the SAME pretrained
checkpoint. They are NOT separately trained models.
"""

import json
from pathlib import Path

import torch

from glic_model import GLICModel

from test import (
    ImageDataset,
    build_dataloader,
    evaluate_forward,
    evaluate_compress,
)


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = Path("0.05checkpoint_best.pth.tar")
IMAGE_DIR = Path("test_images")

RESULT_DIR = Path("results/ablation")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# GET ALL GFA MODULES
# ============================================================

def get_gfa_modules(model):
    """
    Return all GFA modules in both encoder and decoder.

    Encoder:
        g2
        g3

    Decoder:
        g1
        g2
    """

    gfas = [
        model.g_a.g2,
        model.g_a.g3,
        model.g_s.g1,
        model.g_s.g2,
    ]

    return gfas


# ============================================================
# CHANGE GRAPH SAMPLING MODE
# ============================================================

def set_graph_mode(model, mode):
    """
    Change how every GraphAttentionLayer samples graph nodes.

    mode = "official"
        Keep the original alternating GN / GS configuration.

    mode = "local"
        Force every graph layer to use local sampling.

    mode = "global"
        Force every graph layer to use global sampling.
    """

    if mode == "official":
        # Do absolutely nothing.
        # The original architecture is already configured.
        return

    if mode == "local":
        sampling_method = 0

    elif mode == "global":
        sampling_method = 1

    else:
        raise ValueError(
            f"Unknown graph mode: {mode}"
        )

    for gfa in get_gfa_modules(model):

        for block in gfa.residual_group.blocks:

            # 0 = local sampling
            # 1 = global sampling
            block.sampling_method = sampling_method


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading GLIC model...")

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

    # Remove DataParallel prefix if necessary.
    if state_dict and all(
        key.startswith("module.")
        for key in state_dict.keys()
    ):
        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    # IMPORTANT:
    # We load the ORIGINAL pretrained weights.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.update()
    model.eval()

    return model


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(mode, dataset):

    print("\n" + "=" * 70)

    if mode == "official":
        title = "OFFICIAL GLIC"

    elif mode == "local":
        title = "LOCAL-ONLY GLIC"

    else:
        title = "GLOBAL-ONLY GLIC"

    print(title)

    print("=" * 70)

    model = load_model()

    # Change only the graph sampling behavior.
    # The pretrained weights remain unchanged.
    set_graph_mode(
        model,
        mode,
    )

    print("Running forward evaluation...")

    forward_loader = build_dataloader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    forward_result = evaluate_forward(
        model=model,
        dataloader=forward_loader,
        device=DEVICE,
        pad_multiple=128,
    )

    print(
        f"Forward BPP : {forward_result.bpp:.6f}"
    )

    print(
        f"Forward PSNR: {forward_result.psnr:.4f}"
    )

    print("\nRunning actual compression...")

    compress_result = evaluate_compress(
        model=model,
        dataset=dataset,
        device=DEVICE,
        pad_multiple=128,
        num_workers=0,
    )

    print(
        f"Actual BPP   : {compress_result.bpp:.6f}"
    )

    print(
        f"Actual PSNR  : {compress_result.psnr:.4f}"
    )

    return {
        "mode": mode,
        "forward_bpp": forward_result.bpp,
        "forward_psnr": forward_result.psnr,
        "compress_bpp": compress_result.bpp,
        "compress_psnr": compress_result.psnr,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GLIC GRAPH ABLATION")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print(
        f"Images: {IMAGE_DIR}"
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = ImageDataset(
        str(IMAGE_DIR)
    )

    print(
        f"Found {len(dataset)} images."
    )

    results = []

    # --------------------------------------------------------
    # 1. Original GLIC
    # --------------------------------------------------------

    results.append(
        run_experiment(
            "official",
            dataset,
        )
    )

    # --------------------------------------------------------
    # 2. Local-only
    # --------------------------------------------------------

    results.append(
        run_experiment(
            "local",
            dataset,
        )
    )

    # --------------------------------------------------------
    # 3. Global-only
    # --------------------------------------------------------

    results.append(
        run_experiment(
            "global",
            dataset,
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_file = (
        RESULT_DIR /
        "graph_ablation.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Print final table
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("FINAL ABLATION RESULTS")
    print("=" * 90)

    print(
        f"{'Mode':<18}"
        f"{'BPP':>14}"
        f"{'PSNR':>14}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['mode']:<18}"
            f"{result['compress_bpp']:>14.6f}"
            f"{result['compress_psnr']:>14.4f}"
        )

    print("-" * 90)

    print(
        f"\nSaved to:\n"
        f"{output_file.resolve()}"
    )


if __name__ == "__main__":
    main()