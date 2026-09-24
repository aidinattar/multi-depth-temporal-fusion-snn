from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    """Read and validate a completed run directory."""

    root = Path(run_dir)
    summary_path = root / "summary.json"
    metrics_path = root / "metrics.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing run metrics: {metrics_path}")

    summary = load_json(summary_path)
    metrics = read_metrics_jsonl(metrics_path)
    checkpoint = summary.get("readout_checkpoint")
    checkpoint_exists = bool(checkpoint and Path(str(checkpoint)).is_file())
    if summary.get("status") == "completed" and not checkpoint_exists:
        checkpoint_exists = any(
            (root / "checkpoints" / name).is_file()
            for name in ("best_readout.pt", "best_readout.npz")
        )

    return {
        "run_dir": str(root),
        "dataset": summary.get("dataset"),
        "seed": summary.get("seed"),
        "status": summary.get("status"),
        "best_test_accuracy": summary.get("best_test_accuracy"),
        "final_test_accuracy": summary.get("final_test_accuracy"),
        "best_epoch": summary.get("best_epoch"),
        "epochs_completed": summary.get("epochs_completed"),
        "feature_dim": summary.get("feature_dim"),
        "mean_spikes_per_sample": summary.get("mean_spikes_per_sample"),
        "density_percent": summary.get("density_percent"),
        "metrics_records": len(metrics),
        "checkpoint_exists": checkpoint_exists,
    }


def format_evaluation(result: dict[str, Any]) -> str:
    fields = [
        ("run_dir", "run_dir"),
        ("dataset", "dataset"),
        ("seed", "seed"),
        ("status", "status"),
        ("best_test_accuracy", "best_test_accuracy"),
        ("final_test_accuracy", "final_test_accuracy"),
        ("best_epoch", "best_epoch"),
        ("epochs_completed", "epochs_completed"),
        ("feature_dim", "feature_dim"),
        ("mean_spikes_per_sample", "mean_spikes_per_sample"),
        ("density_percent", "density_percent"),
        ("metrics_records", "metrics_records"),
        ("checkpoint_exists", "checkpoint_exists"),
    ]
    lines = []
    for key, label in fields:
        value = result.get(key)
        if isinstance(value, float):
            value = f"{value:.6g}"
        lines.append(f"{label}: {value}")
    return "\n".join(lines)
