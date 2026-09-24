from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from temporal_evidence_snn.data.static_dataset import load_static_image_npz
from temporal_evidence_snn.data.static_images import as_image_batch


@dataclass(frozen=True)
class DatasetNPZValidation:
    path: str
    dataset: str | None
    train_samples: int
    test_samples: int
    train_shape: tuple[int, ...]
    test_shape: tuple[int, ...]
    channels: int
    num_classes: int
    finite_values: bool

    def to_dict(self) -> dict:
        return asdict(self)


def expected_channels(dataset_name: str | None) -> int | None:
    if dataset_name is None:
        return None
    normalized = str(dataset_name).lower().replace("-", "_")
    if normalized in {"mnist", "fashion_mnist", "fashionmnist"}:
        return 1
    if normalized in {"cifar10", "cifar_10"}:
        return 3
    if normalized in {"nmnist", "n_mnist"}:
        return 10
    return None


def validate_dataset_npz(
    path: Path,
    *,
    dataset_name: str | None = None,
) -> DatasetNPZValidation:
    """Validate the public NPZ dataset contract.

    The NPZ must contain train/test image arrays and train/test labels. Arrays
    may use either the explicit public names or the common x/y aliases accepted
    by the loader.
    """

    splits = load_static_image_npz(Path(path))
    if splits.train_images.shape[0] != splits.train_labels.shape[0]:
        raise ValueError(
            "train_images and train_labels have different sample counts: "
            f"{splits.train_images.shape[0]} vs {splits.train_labels.shape[0]}"
        )
    if splits.test_images.shape[0] != splits.test_labels.shape[0]:
        raise ValueError(
            "test_images and test_labels have different sample counts: "
            f"{splits.test_images.shape[0]} vs {splits.test_labels.shape[0]}"
        )
    if splits.train_labels.ndim != 1 or splits.test_labels.ndim != 1:
        raise ValueError("Labels must be one-dimensional arrays.")
    if splits.train_labels.size == 0 or splits.test_labels.size == 0:
        raise ValueError("Both train and test splits must contain at least one sample.")

    channels = expected_channels(dataset_name)
    train_images = as_image_batch(splits.train_images, channels=channels)
    test_images = as_image_batch(splits.test_images, channels=channels)
    if int(train_images.shape[1]) != int(test_images.shape[1]):
        raise ValueError(
            "train and test image arrays expose different channel counts: "
            f"{train_images.shape[1]} vs {test_images.shape[1]}"
        )
    if tuple(train_images.shape[2:]) != tuple(test_images.shape[2:]):
        raise ValueError(
            "train and test image arrays expose different spatial shapes: "
            f"{tuple(train_images.shape[2:])} vs {tuple(test_images.shape[2:])}"
        )

    labels = np.concatenate([splits.train_labels, splits.test_labels]).astype(np.int64)
    if np.any(labels < 0):
        raise ValueError("Labels must be non-negative class indices.")

    return DatasetNPZValidation(
        path=str(Path(path)),
        dataset=None if dataset_name is None else str(dataset_name),
        train_samples=int(train_images.shape[0]),
        test_samples=int(test_images.shape[0]),
        train_shape=tuple(int(value) for value in train_images.shape),
        test_shape=tuple(int(value) for value in test_images.shape),
        channels=int(train_images.shape[1]),
        num_classes=int(labels.max()) + 1,
        finite_values=bool(np.isfinite(train_images).all() and np.isfinite(test_images).all()),
    )
