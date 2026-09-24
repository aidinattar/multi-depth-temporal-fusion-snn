from __future__ import annotations

import numpy as np
import pytest

from temporal_evidence_snn.data import load_dataset_splits


def test_load_dataset_splits_finds_npz_from_dataset_root(tmp_path) -> None:
    dataset_path = tmp_path / "mnist.npz"
    np.savez(
        dataset_path,
        train_images=np.zeros((3, 8, 8), dtype=np.uint8),
        train_labels=np.array([0, 1, 2], dtype=np.int16),
        test_images=np.zeros((2, 8, 8), dtype=np.uint8),
        test_labels=np.array([0, 1], dtype=np.int16),
    )

    splits = load_dataset_splits(
        {"name": "mnist"},
        dataset_root=tmp_path,
        train_limit=2,
        test_limit=1,
    )

    assert splits.train_images.shape == (2, 8, 8)
    assert splits.train_labels.tolist() == [0, 1]
    assert splits.test_images.shape == (1, 8, 8)


def test_nmnist_requires_event_surface_npz(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="event-surface NPZ"):
        load_dataset_splits({"name": "nmnist"}, dataset_root=tmp_path)


def test_nmnist_event_surface_npz_uses_npz_contract(tmp_path) -> None:
    dataset_path = tmp_path / "N-MNIST.npz"
    np.savez(
        dataset_path,
        train_images=np.zeros((2, 10, 8, 8), dtype=np.float32),
        train_labels=np.array([0, 1], dtype=np.int16),
        test_images=np.zeros((1, 10, 8, 8), dtype=np.float32),
        test_labels=np.array([1], dtype=np.int16),
    )

    splits = load_dataset_splits({"name": "nmnist"}, dataset_root=tmp_path)

    assert splits.train_images.shape == (2, 10, 8, 8)
    assert splits.test_labels.tolist() == [1]
