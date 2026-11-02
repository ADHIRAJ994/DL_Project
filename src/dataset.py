
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


# Supported image formats
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def get_image_files(root):
    """
    Find all image files inside a directory.
    Searches subdirectories as well.
    """

    root = Path(root)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Image directory does not exist: {root.resolve()}"
        )

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(
            f"No images found inside: {root.resolve()}"
        )

    return files


def image_to_tensor(image):
    """
    Convert PIL image into a PyTorch tensor.

    Output shape: [3, H, W]
    Pixel range: [0, 1]
    """

    array = np.asarray(
        image.convert("RGB"),
        dtype=np.float32
    ).copy() / 255.0

    tensor = torch.from_numpy(array)

    return tensor.permute(2, 0, 1).contiguous()


class DIV2KPatchDataset(Dataset):

    def __init__(
        self,
        root,
        patch_size=128,
        training=True
    ):

        if patch_size <= 0:
            raise ValueError("patch_size must be positive")

        self.root = Path(root)

        self.files = get_image_files(self.root)

        self.patch_size = patch_size
        self.training = training

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        path = self.files[index]

        # Load image safely
        with Image.open(path) as im:
            image = im.convert("RGB")

        width, height = image.size

        # Resize if image is smaller than required patch
        if width < self.patch_size or height < self.patch_size:

            scale = max(
                self.patch_size / width,
                self.patch_size / height
            )

            new_size = (
                max(self.patch_size, round(width * scale)),
                max(self.patch_size, round(height * scale))
            )

            image = image.resize(
                new_size,
                Image.Resampling.BICUBIC
            )

            width, height = image.size

        # Random crop for training
        if self.training:

            left = random.randint(
                0, width - self.patch_size
            )

            top = random.randint(
                0, height - self.patch_size
            )

        # Center crop for validation
        else:

            left = (width - self.patch_size) // 2
            top = (height - self.patch_size) // 2

        image = image.crop((
            left,
            top,
            left + self.patch_size,
            top + self.patch_size
        ))

        # Data augmentation
        if self.training:

            if random.random() < 0.5:
                image = ImageOps.mirror(image)

            if random.random() < 0.5:
                image = ImageOps.flip(image)

        return image_to_tensor(image)
