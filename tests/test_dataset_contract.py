from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from temporal_evidence_snn.data import validate_dataset_npz


def test_validate_dataset_npz_accepts_nmnist_event_surface_export(tmp_path: Path) -> None:
    path = tmp_path / "N-MNIST.npz"
    np.savez(
        path,
        train_images=np.zeros((3, 10, 8, 8), dtype=np.float32),
        train_labels=np.array([0, 1, 2], dtype=np.int16),
        test_images=np.zeros((2, 10, 8, 8), dtype=np.float32),
        test_labels=np.array([0, 1], dtype=np.int16),
    )

    result = validate_dataset_npz(path, dataset_name="nmnist")

    assert result.channels == 10
    assert result.train_samples == 3
    assert result.test_samples == 2
    assert result.num_classes == 3
    assert result.finite_values is True


def test_validate_dataset_npz_rejects_wrong_nmnist_channels(tmp_path: Path) -> None:
    path = tmp_path / "bad_nmnist.npz"
    np.savez(
        path,
        train_images=np.zeros((3, 2, 8, 8), dtype=np.float32),
        train_labels=np.array([0, 1, 2], dtype=np.int16),
        test_images=np.zeros((2, 2, 8, 8), dtype=np.float32),
        test_labels=np.array([0, 1], dtype=np.int16),
    )

    with pytest.raises(ValueError, match="expected channel axis"):
        validate_dataset_npz(path, dataset_name="nmnist")


def test_validate_dataset_npz_cli_prints_json(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    path = tmp_path / "mnist.npz"
    np.savez(
        path,
        train_images=np.zeros((2, 8, 8), dtype=np.uint8),
        train_labels=np.array([0, 1], dtype=np.int16),
        test_images=np.zeros((1, 8, 8), dtype=np.uint8),
        test_labels=np.array([1], dtype=np.int16),
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dataset_npz.py",
            str(path),
            "--dataset",
            "mnist",
            "--json",
        ],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert '"dataset": "mnist"' in completed.stdout
    assert '"channels": 1' in completed.stdout
