from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_full_backend_can_export_base_evidence_from_local_npz(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "data"
    dataset_root.mkdir()
    dataset_path = dataset_root / "mnist.npz"

    train_images = np.zeros((4, 8, 8), dtype=np.uint8)
    train_images[:, 2:6, 2:6] = 255
    test_images = np.zeros((2, 8, 8), dtype=np.uint8)
    test_images[:, 1:7, 3:5] = 255
    np.savez(
        dataset_path,
        train_images=train_images,
        train_labels=np.array([0, 1, 0, 1], dtype=np.int16),
        test_images=test_images,
        test_labels=np.array([0, 1], dtype=np.int16),
    )

    output_root = tmp_path / "runs" / "full_model"
    cmd = [
        sys.executable,
        "scripts/train_full_model.py",
        "--experiment",
        "configs/experiments/full_model_mnist.yaml",
        "--backend",
        "full",
        "--stop-after",
        "base-evidence",
        "--dataset-root",
        str(dataset_root),
        "--seed",
        "0",
        "--output-root",
        str(output_root),
    ]
    subprocess.run(cmd, cwd=repo, check=True)

    run_dir = output_root / "mnist" / "seed_0"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "base_evidence_completed"
    assert summary["backend"] == "full"
    assert summary["feature_dim"] > 0
    assert summary["mean_spikes_per_sample"] is not None
    assert (run_dir / "base_evidence" / "trainset.npy").is_file()
    assert (run_dir / "base_evidence" / "testset.npy").is_file()
    assert (run_dir / "base_evidence" / "spiking_dataset_summary.json").is_file()


def test_full_backend_can_continue_to_residual_evidence_from_local_npz(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    dataset_path = tmp_path / "tiny_mnist_like_early_branch.npz"

    train_images = np.zeros((4, 16, 16), dtype=np.uint8)
    train_images[:, 4:12, 4:12] = 255
    test_images = np.zeros((2, 16, 16), dtype=np.uint8)
    test_images[:, 3:13, 6:10] = 255
    np.savez(
        dataset_path,
        train_images=train_images,
        train_labels=np.array([0, 1, 0, 1], dtype=np.int16),
        test_images=test_images,
        test_labels=np.array([0, 1], dtype=np.int16),
    )

    output_root = tmp_path / "runs" / "full_model"
    cmd = [
        sys.executable,
        "scripts/train_full_model.py",
        "--experiment",
        "configs/experiments/full_model_mnist.yaml",
        "--backend",
        "full",
        "--stop-after",
        "residual-evidence",
        "--dataset-npz",
        str(dataset_path),
        "--seed",
        "0",
        "--output-root",
        str(output_root),
    ]
    subprocess.run(cmd, cwd=repo, check=True)

    run_dir = output_root / "mnist" / "seed_0"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "residual_evidence_completed"
    assert summary["last_completed_stage"] == "residual_evidence"
    assert summary["feature_dim"] > 0
    assert (run_dir / "base_evidence" / "trainset.npy").is_file()
    assert (run_dir / "residual_evidence" / "trainset.npy").is_file()
    assert (run_dir / "residual_evidence" / "spiking_dataset_summary.json").is_file()


def test_full_backend_can_build_fused_sparse_code_from_local_npz(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    dataset_path = tmp_path / "tiny_mnist_like_fused_code.npz"

    train_images = np.zeros((4, 16, 16), dtype=np.uint8)
    train_images[0::2, 3:11, 3:11] = 255
    train_images[1::2, 5:13, 5:13] = 255
    test_images = np.zeros((2, 16, 16), dtype=np.uint8)
    test_images[0, 3:11, 3:11] = 255
    test_images[1, 5:13, 5:13] = 255
    np.savez(
        dataset_path,
        train_images=train_images,
        train_labels=np.array([0, 1, 0, 1], dtype=np.int16),
        test_images=test_images,
        test_labels=np.array([0, 1], dtype=np.int16),
    )

    output_root = tmp_path / "runs" / "full_model"
    cmd = [
        sys.executable,
        "scripts/train_full_model.py",
        "--experiment",
        "configs/experiments/full_model_mnist.yaml",
        "--backend",
        "full",
        "--stop-after",
        "fused-sparse-code",
        "--dataset-npz",
        str(dataset_path),
        "--seed",
        "0",
        "--output-root",
        str(output_root),
    ]
    subprocess.run(cmd, cwd=repo, check=True)

    run_dir = output_root / "mnist" / "seed_0"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "fused_sparse_code_completed"
    assert summary["last_completed_stage"] == "fused_sparse_code"
    assert summary["feature_dim"] > 0
    for directory in (
        "base_evidence",
        "residual_evidence",
        "deep_transform",
        "deep_correction",
        "fused_sparse_code",
    ):
        assert (run_dir / directory / "trainset.npy").is_file()
        assert (run_dir / directory / "testset.npy").is_file()
        assert (run_dir / directory / "spiking_dataset_summary.json").is_file()
    assert set(summary["stage_summaries"]) == {
        "base_evidence",
        "residual_evidence",
        "deep_transform",
        "deep_correction",
        "fused_sparse_code",
    }


def test_full_backend_can_train_readout_from_local_npz(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    dataset_path = tmp_path / "tiny_mnist_like_full_readout.npz"

    train_images = np.zeros((4, 16, 16), dtype=np.uint8)
    train_images[0::2, 3:11, 3:11] = 255
    train_images[1::2, 5:13, 5:13] = 255
    test_images = np.zeros((2, 16, 16), dtype=np.uint8)
    test_images[0, 3:11, 3:11] = 255
    test_images[1, 5:13, 5:13] = 255
    np.savez(
        dataset_path,
        train_images=train_images,
        train_labels=np.array([0, 1, 0, 1], dtype=np.int16),
        test_images=test_images,
        test_labels=np.array([0, 1], dtype=np.int16),
    )

    output_root = tmp_path / "runs" / "full_model"
    cmd = [
        sys.executable,
        "scripts/train_full_model.py",
        "--experiment",
        "configs/experiments/full_model_mnist.yaml",
        "--backend",
        "full",
        "--dataset-npz",
        str(dataset_path),
        "--seed",
        "0",
        "--output-root",
        str(output_root),
        "--epochs",
        "2",
    ]
    subprocess.run(cmd, cwd=repo, check=True)

    run_dir = output_root / "mnist" / "seed_0"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["last_completed_stage"] == "readout"
    assert summary["epochs_completed"] > 0
    assert summary["best_test_accuracy"] is not None
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "best_readout.pt").is_file()
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "starting stage=base_evidence" in run_log
    assert "completed stage=fused_sparse_code" in run_log
    assert "completed stage=temporal_readout" in run_log
    assert "run completed status=completed" in run_log
    for checkpoint in (
        "base_evidence.pt",
        "residual_evidence.pt",
        "deep_transform.pt",
        "deep_correction.pt",
    ):
        assert (run_dir / "checkpoints" / checkpoint).is_file()
