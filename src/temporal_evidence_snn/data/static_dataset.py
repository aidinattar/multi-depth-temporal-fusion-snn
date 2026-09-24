from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StaticImageDatasetSplits:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray


def _first_present(payload: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in payload:
            return payload[name]
    raise KeyError(f"Missing one of required arrays: {', '.join(names)}")


def _limit(array: np.ndarray, limit: int) -> np.ndarray:
    count = int(limit)
    if count <= 0:
        return array
    return array[:count]


def load_static_image_npz(
    path: Path,
    *,
    train_limit: int = 0,
    test_limit: int = 0,
) -> StaticImageDatasetSplits:
    """Load a local static-image dataset from an NPZ file.

    Supported key pairs are ``train_images/train_labels`` and
    ``test_images/test_labels``. Common aliases such as ``x_train/y_train`` and
    ``x_test/y_test`` are accepted to make exported dataset files easy to use.
    """

    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}

    train_images = _first_present(payload, ("train_images", "x_train"))
    train_labels = _first_present(payload, ("train_labels", "y_train"))
    test_images = _first_present(payload, ("test_images", "x_test"))
    test_labels = _first_present(payload, ("test_labels", "y_test"))

    return StaticImageDatasetSplits(
        train_images=_limit(np.asarray(train_images), train_limit),
        train_labels=_limit(np.asarray(train_labels, dtype=np.int16), train_limit),
        test_images=_limit(np.asarray(test_images), test_limit),
        test_labels=_limit(np.asarray(test_labels, dtype=np.int16), test_limit),
    )


def _torchvision_arrays(dataset) -> tuple[np.ndarray, np.ndarray]:
    images = getattr(dataset, "data", None)
    labels = getattr(dataset, "targets", None)
    if images is None or labels is None:
        raise ValueError("Unsupported torchvision dataset object: missing data/targets")
    if hasattr(images, "numpy"):
        images = images.numpy()
    return np.asarray(images), np.asarray(labels, dtype=np.int16)


def _load_torchvision_static_dataset(
    name: str,
    *,
    dataset_root: Path,
    train_limit: int,
    test_limit: int,
    download: bool,
) -> StaticImageDatasetSplits:
    try:
        from torchvision import datasets
    except Exception as exc:  # pragma: no cover - depends on optional dependency.
        raise ImportError(
            "Loading MNIST, Fashion-MNIST, or CIFAR-10 by name requires torchvision. "
            "Install the optional torchvision dependency or pass --dataset-npz."
        ) from exc

    dataset_classes = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
    }
    if name not in dataset_classes:
        raise ValueError(f"No torchvision loader is defined for dataset {name!r}")

    dataset_class = dataset_classes[name]
    train_dataset = dataset_class(root=str(dataset_root), train=True, download=bool(download))
    test_dataset = dataset_class(root=str(dataset_root), train=False, download=bool(download))
    train_images, train_labels = _torchvision_arrays(train_dataset)
    test_images, test_labels = _torchvision_arrays(test_dataset)
    return StaticImageDatasetSplits(
        train_images=_limit(train_images, train_limit),
        train_labels=_limit(train_labels, train_limit),
        test_images=_limit(test_images, test_limit),
        test_labels=_limit(test_labels, test_limit),
    )


def _dataset_npz_candidates(name: str, dataset_root: Path) -> list[Path]:
    aliases = {
        "mnist": ["mnist.npz", "MNIST.npz"],
        "fashion_mnist": ["fashion_mnist.npz", "fashion-mnist.npz", "Fashion-MNIST.npz"],
        "cifar10": ["cifar10.npz", "cifar-10.npz", "CIFAR-10.npz"],
        "nmnist": ["nmnist.npz", "n-mnist.npz", "N-MNIST.npz"],
    }
    return [dataset_root / filename for filename in aliases.get(name, [])]


def load_dataset_splits(
    dataset_config: dict[str, Any],
    *,
    dataset_root: Path,
    dataset_npz: Path | None = None,
    train_limit: int = 0,
    test_limit: int = 0,
    download: bool = False,
) -> StaticImageDatasetSplits:
    """Load the dataset requested by a public experiment config.

    Static datasets can be loaded either from torchvision or from a local NPZ.
    N-MNIST is intentionally NPZ-based in this repository: the expected arrays
    are event-surface summaries with the same train/test key contract used by
    ``load_static_image_npz``.
    """

    name = str(dataset_config.get("name", ""))
    root = Path(dataset_root)
    if dataset_npz is not None:
        return load_static_image_npz(
            Path(dataset_npz),
            train_limit=train_limit,
            test_limit=test_limit,
        )

    for candidate in _dataset_npz_candidates(name, root):
        if candidate.is_file():
            return load_static_image_npz(
                candidate,
                train_limit=train_limit,
                test_limit=test_limit,
            )

    if name in {"mnist", "fashion_mnist", "cifar10"}:
        return _load_torchvision_static_dataset(
            name,
            dataset_root=root,
            train_limit=train_limit,
            test_limit=test_limit,
            download=download,
        )

    if name == "nmnist":
        candidates = ", ".join(str(path) for path in _dataset_npz_candidates(name, root))
        raise FileNotFoundError(
            "N-MNIST loading expects an event-surface NPZ export. "
            f"Place one of these files under --dataset-root: {candidates}"
        )

    raise ValueError(f"Unsupported dataset name: {name!r}")
