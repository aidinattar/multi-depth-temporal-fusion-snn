from __future__ import annotations

from pathlib import Path
from typing import Any

from temporal_evidence_snn.evaluate import evaluate_run


SUMMARY_COLUMNS = [
    "dataset",
    "seed",
    "status",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "epochs_completed",
    "feature_dim",
    "mean_spikes_per_sample",
    "density_percent",
    "run_dir",
]


def discover_run_dirs(root: Path) -> list[Path]:
    base = Path(root)
    if (base / "summary.json").is_file():
        return [base]
    return sorted(path.parent for path in base.glob("*/*/summary.json"))


def summarize_runs(root: Path) -> list[dict[str, Any]]:
    return [evaluate_run(run_dir) for run_dir in discover_run_dirs(root)]


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def rows_to_tsv(rows: list[dict[str, Any]]) -> str:
    lines = ["\t".join(SUMMARY_COLUMNS)]
    for row in rows:
        lines.append("\t".join(_format_value(row.get(column)) for column in SUMMARY_COLUMNS))
    return "\n".join(lines) + "\n"


def rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_format_value(row.get(column)) for column in SUMMARY_COLUMNS) + " |"
        )
    return "\n".join(lines) + "\n"
