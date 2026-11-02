
import json
import struct
import zlib

from pathlib import Path

import numpy as np


MAGIC = b"SNN1"


def save_bitstream(latent, metadata, output_path):
    """
    Save integer SNN latent symbols in a compressed file.

    File format:
        4 bytes  : Magic signature
        4 bytes  : Metadata length
        N bytes  : JSON metadata
        Remaining: Zlib-compressed latent symbols
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    latent = np.asarray(latent)

    # Spike counts must be integers
    rounded = np.rint(latent)

    if not np.allclose(latent, rounded):
        raise ValueError(
            "Latent contains non-integer values."
        )

    latent = rounded.astype(np.int64)

    # Spike counts must fit in uint8
    if latent.min() < 0 or latent.max() > 255:
        raise ValueError(
            "Latent values are outside uint8 range."
        )

    latent = latent.astype(np.uint8)

    raw_data = latent.tobytes(order="C")

    # Lossless compression
    compressed_data = zlib.compress(
        raw_data,
        level=9
    )

    metadata = dict(metadata)

    metadata["version"] = 1
    metadata["latent_shape"] = list(latent.shape)
    metadata["raw_bytes"] = len(raw_data)
    metadata["coding"] = "zlib"

    metadata_bytes = json.dumps(
        metadata
    ).encode("utf-8")

    if len(metadata_bytes) > 10_000_000:
        raise ValueError("Metadata is unexpectedly large.")

    with open(output_path, "wb") as f:

        f.write(MAGIC)

        f.write(
            struct.pack(">I", len(metadata_bytes))
        )

        f.write(metadata_bytes)

        f.write(compressed_data)

    file_size = output_path.stat().st_size

    return {
        "file_size": file_size,
        "raw_size": len(raw_data),
        "compressed_payload_size": len(compressed_data),
        "metadata_size": len(metadata_bytes)
    }


def load_bitstream(input_path):
    """
    Read an SNN compressed file and recover
    the original integer spike counts.
    """

    input_path = Path(input_path)

    with open(input_path, "rb") as f:

        magic = f.read(4)

        if magic != MAGIC:
            raise ValueError(
                "Invalid SNN compressed file."
            )

        length_data = f.read(4)

        if len(length_data) != 4:
            raise ValueError("Incomplete file header.")

        metadata_length = struct.unpack(
            ">I", length_data
        )[0]

        if metadata_length > 10_000_000:
            raise ValueError("Invalid metadata length.")

        metadata_bytes = f.read(metadata_length)

        if len(metadata_bytes) != metadata_length:
            raise ValueError("Incomplete metadata.")

        metadata = json.loads(
            metadata_bytes.decode("utf-8")
        )

        compressed_data = f.read()

    if metadata.get("version") != 1:
        raise ValueError("Unsupported bitstream version.")

    if metadata.get("coding") != "zlib":
        raise ValueError("Unsupported coding method.")

    try:
        raw_data = zlib.decompress(compressed_data)
    except zlib.error as exc:
        raise ValueError("Corrupted compressed payload.") from exc

    expected_size = int(metadata["raw_bytes"])

    if len(raw_data) != expected_size:
        raise ValueError("Decoded payload size mismatch.")

    shape = tuple(
        int(value) for value in metadata["latent_shape"]
    )

    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError("Invalid latent dimensions.")

    expected_elements = int(np.prod(shape))

    if expected_elements * np.dtype(np.uint8).itemsize != expected_size:
        raise ValueError("Latent shape does not match payload size.")

    latent = np.frombuffer(
        raw_data,
        dtype=np.uint8
    ).copy()

    latent = latent.reshape(shape)

    return latent, metadata
