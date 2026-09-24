from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from temporal_evidence_snn.readout.temporal_rstdp import (
    SparseSpikingFeatureDataset,
    TemporalReadout,
    build_readout_from_config,
)


def _shuffle_dataset_in_memory(
    dataset: SparseSpikingFeatureDataset,
    *,
    seed: int,
) -> None:
    """Shuffle the training split once with a seeded permutation."""

    order = np.random.default_rng(int(seed)).permutation(len(dataset))
    dataset.labels = dataset.labels[order]
    dataset.data = [dataset.data[int(index)] for index in order]


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prepare_config(config: dict[str, Any], *, epochs: int) -> dict[str, Any]:
    resolved = deepcopy(config)
    required = {"network", "optimizer", "regularizer", "trainer"}
    missing = required.difference(resolved)
    if missing:
        raise ValueError(
            "Readout config is incomplete; missing sections: " + ", ".join(sorted(missing))
        )
    resolved["trainer"]["epochs"] = int(epochs)
    return resolved


def build_temporal_readout(
    *,
    config: dict[str, Any],
    input_size: int,
    num_classes: int,
    max_time: float,
    device: str = "cpu",
    output_dir: Path | None = None,
) -> TemporalReadout:
    """Build a temporal R-STDP readout from its resolved configuration."""

    return build_readout_from_config(
        config,
        input_size=int(input_size),
        n_classes=int(num_classes),
        max_time=float(max_time),
        device=str(device),
        output_dir=output_dir,
    )


def train_rstdp_temporal_readout(
    *,
    feature_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    seed: int,
    epochs: int,
    train_limit: int = 0,
    test_limit: int = 0,
    shuffle_train: bool = False,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train a temporal R-STDP readout on a fused sparse latency representation."""

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    train_dataset = SparseSpikingFeatureDataset(
        feature_dir / "trainset.npy",
        max_samples=int(train_limit),
    )
    test_dataset = SparseSpikingFeatureDataset(
        feature_dir / "testset.npy",
        max_samples=int(test_limit),
    )
    if shuffle_train:
        _shuffle_dataset_in_memory(train_dataset, seed=int(seed))

    classes = np.unique(train_dataset.labels)
    if classes.size == 0:
        raise ValueError("The readout training set is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = _prepare_config(config, epochs=int(epochs))
    readout = build_temporal_readout(
        config=resolved_config,
        input_size=train_dataset.input_size,
        num_classes=int(classes.size),
        max_time=train_dataset.max_time,
        device=str(device),
        output_dir=output_dir,
    )
    training_summary = readout.fit(
        train_dataset,
        val_dataset=None,
        test_dataset=test_dataset,
    )
    metrics = _load_metrics(output_dir / "readout_metrics.jsonl")
    summary = {
        "status": training_summary["status"],
        "best_test_accuracy": training_summary["best_test_acc"],
        "final_test_accuracy": training_summary["final_test_acc"],
        "best_epoch": training_summary["best_monitor_epoch"],
        "epochs_completed": training_summary["epochs_completed"],
        "readout_checkpoint": str(output_dir / "checkpoints" / "best_readout.pt"),
    }
    with (output_dir / "readout_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics, summary


def evaluate_rstdp_temporal_readout(
    *,
    feature_dir: Path,
    checkpoint_path: Path,
    config: dict[str, Any],
    test_limit: int = 0,
    device: str = "cpu",
) -> float:
    """Evaluate a saved temporal R-STDP readout on fused test features."""

    test_dataset = SparseSpikingFeatureDataset(
        feature_dir / "testset.npy",
        max_samples=int(test_limit),
    )
    classes = np.unique(test_dataset.labels)
    if classes.size == 0:
        raise ValueError("The readout test set is empty.")
    readout = build_temporal_readout(
        config=config,
        input_size=test_dataset.input_size,
        num_classes=int(classes.size),
        max_time=test_dataset.max_time,
        device=str(device),
        output_dir=None,
    )
    readout.load_checkpoint(checkpoint_path)
    return readout.predict(test_dataset, name="test")
