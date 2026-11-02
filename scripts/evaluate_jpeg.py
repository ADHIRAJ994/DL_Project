import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage.metrics import structural_similarity


def calculate_metrics(original, reconstructed):
    original = np.asarray(original, dtype=np.float32) / 255.0
    reconstructed = np.asarray(reconstructed, dtype=np.float32) / 255.0

    mse = float(np.mean((original - reconstructed) ** 2))

    psnr = (
        float("inf")
        if mse == 0
        else float(10 * np.log10(1.0 / mse))
    )

    ssim = float(
        structural_similarity(
            original,
            reconstructed,
            channel_axis=2,
            data_range=1.0
        )
    )

    return mse, psnr, ssim


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kodak-dir",
        default="data/Kodak"
    )

    parser.add_argument(
        "--snn-csv",
        default="results/snn_codec_final/snn_codec_results.csv"
    )

    parser.add_argument(
        "--output-dir",
        default="results/jpeg_comparison"
    )

    parser.add_argument(
        "--qualities",
        nargs="+",
        type=int,
        default=[50, 60, 70, 80, 90]
    )

    args = parser.parse_args()

    kodak_dir = Path(args.kodak_dir)
    output_dir = Path(args.output_dir)
    plots_dir = Path("plots")

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load the existing SNN codec results.
    snn_df = pd.read_csv(args.snn_csv)
    snn_df = snn_df[snn_df["image"] != "AVERAGE"]

    snn_lookup = snn_df.set_index("image")

    image_files = sorted(kodak_dir.glob("*.png"))

    if len(image_files) == 0:
        raise FileNotFoundError(
            f"No PNG images found in {kodak_dir}"
        )

    results = []

    # Evaluate JPEG at every selected quality level.
    for quality in args.qualities:

        print(f"\nTesting JPEG quality: {quality}")

        for image_path in image_files:

            original = Image.open(image_path).convert("RGB")

            width, height = original.size

            # Encode JPEG entirely in memory.
            buffer = io.BytesIO()

            original.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True
            )

            jpeg_bytes = buffer.getvalue()

            # Decode the JPEG bitstream.
            buffer.seek(0)

            reconstructed = Image.open(buffer).convert("RGB")

            mse, psnr, ssim = calculate_metrics(
                original,
                reconstructed
            )

            bpp = (
                len(jpeg_bytes) * 8
                / (width * height)
            )

            results.append({
                "image": image_path.name,
                "codec": "JPEG",
                "quality": quality,
                "bitstream_bytes": len(jpeg_bytes),
                "bpp": bpp,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim
            })

            print(
                f"{image_path.name}: "
                f"BPP={bpp:.3f}, "
                f"PSNR={psnr:.3f}, "
                f"SSIM={ssim:.4f}"
            )

    # Store all JPEG per-image results.
    jpeg_df = pd.DataFrame(results)

    # Add the existing SNN measurements to the comparison.
    snn_results = []

    for _, row in snn_df.iterrows():

        snn_results.append({
            "image": row["image"],
            "codec": "SNN-8",
            "quality": None,
            "bitstream_bytes": None,
            "bpp": row["bpp"],
            "mse": row["mse"],
            "psnr": row["psnr"],
            "ssim": row["ssim"]
        })

    combined_df = pd.concat(
        [jpeg_df, pd.DataFrame(snn_results)],
        ignore_index=True
    )

    combined_df.to_csv(
        output_dir / "jpeg_snn_per_image.csv",
        index=False
    )

    # Calculate average metrics for each JPEG quality.
    summary = (
        jpeg_df.groupby("quality")
        [["bpp", "mse", "psnr", "ssim"]]
        .mean()
        .reset_index()
    )

    summary.insert(0, "codec", "JPEG")

    snn_summary = pd.DataFrame([{
        "codec": "SNN-8",
        "quality": None,
        "bpp": snn_df["bpp"].mean(),
        "mse": snn_df["mse"].mean(),
        "psnr": snn_df["psnr"].mean(),
        "ssim": snn_df["ssim"].mean()
    }])

    final_summary = pd.concat(
        [summary, snn_summary],
        ignore_index=True
    )

    final_summary.to_csv(
        output_dir / "jpeg_snn_summary.csv",
        index=False
    )

    # Plot average rate-distortion curves.
    plt.figure(figsize=(8, 6))

    plt.plot(
        summary["bpp"],
        summary["psnr"],
        marker="o",
        label="JPEG"
    )

    plt.scatter(
        snn_summary["bpp"],
        snn_summary["psnr"],
        marker="*",
        s=180,
        label="SNN-8"
    )

    plt.xlabel("Bits per pixel (BPP)")
    plt.ylabel("PSNR (dB)")
    plt.title("JPEG vs SNN: Rate-Distortion Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        plots_dir / "jpeg_vs_snn_rate_distortion.png",
        dpi=180
    )

    plt.show()
    plt.close()

    # Plot PSNR against JPEG quality.
    plt.figure(figsize=(8, 5))

    plt.plot(
        summary["quality"],
        summary["psnr"],
        marker="o"
    )

    plt.xlabel("JPEG Quality")
    plt.ylabel("Average PSNR (dB)")
    plt.title("JPEG Reconstruction Quality")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        plots_dir / "jpeg_quality_vs_psnr.png",
        dpi=180
    )

    plt.show()
    plt.close()

    print("\nFINAL COMPARISON")
    print(final_summary.to_string(index=False))

    print("\nResults saved in:", output_dir)
    print("Graphs saved in:", plots_dir)


if __name__ == "__main__":
    main()