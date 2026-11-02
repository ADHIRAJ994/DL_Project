
from pathlib import Path
import argparse

from PIL import Image


def count_images(folder):

    extensions = {
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tif", ".tiff"
    }

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Folder not found: {folder.resolve()}"
        )

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def main():

    parser = argparse.ArgumentParser(
        description="Verify DIV2K and Kodak datasets"
    )

    parser.add_argument(
        "--train",
        default="data/DIV2K/train_hr"
    )

    parser.add_argument(
        "--valid",
        default="data/DIV2K/valid_hr"
    )

    parser.add_argument(
        "--kodak",
        default="data/Kodak"
    )

    args = parser.parse_args()

    datasets = [
        ("DIV2K Training", Path(args.train)),
        ("DIV2K Validation", Path(args.valid)),
        ("Kodak Testing", Path(args.kodak))
    ]

    for label, folder in datasets:

        files = count_images(folder)

        print(
            f"\n{label}: {len(files)} images"
        )

        print(f"Folder: {folder.resolve()}")

        if files:

            with Image.open(files[0]) as image:

                print(
                    f"Sample image: {files[0].name}"
                )

                print(
                    f"Dimensions: {image.size}"
                )

                print(
                    f"Image mode: {image.mode}"
                )


if __name__ == "__main__":
    main()
