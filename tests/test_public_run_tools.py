from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from temporal_evidence_snn.evaluate import evaluate_run
from temporal_evidence_snn.summarize import rows_to_tsv, summarize_runs


def _write_run(root: Path, dataset: str, seed: int, best: float) -> Path:
    run_dir = root / dataset / f"seed_{seed}"
    (run_dir / "checkpoints").mkdir(parents=True)
    checkpoint = run_dir / "checkpoints" / "best_readout.npz"
    checkpoint.write_bytes(b"checkpoint")
    summary = {
        "dataset": dataset,
        "seed": seed,
        "status": "completed",
        "best_test_accuracy": best,
        "final_test_accuracy": best - 0.01,
        "best_epoch": 1,
        "epochs_completed": 2,
        "feature_dim": 12,
        "mean_spikes_per_sample": 4.5,
        "density_percent": 37.5,
        "readout_checkpoint": str(checkpoint),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(
        '{"kind": "readout_epoch", "epoch": 0}\n{"kind": "readout_epoch", "epoch": 1}\n',
        encoding="utf-8",
    )
    return run_dir


def test_evaluate_run_reads_completed_run(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "runs" / "full_model", "mnist", 0, 0.95)

    result = evaluate_run(run_dir)

    assert result["dataset"] == "mnist"
    assert result["best_test_accuracy"] == 0.95
    assert result["metrics_records"] == 2
    assert result["checkpoint_exists"] is True


def test_summarize_runs_discovers_seed_directories(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "full_model"
    _write_run(root, "mnist", 0, 0.95)
    _write_run(root, "cifar10", 0, 0.61)

    rows = summarize_runs(root)
    tsv = rows_to_tsv(rows)

    assert len(rows) == 2
    assert "dataset\tseed\tstatus" in tsv
    assert "mnist\t0\tcompleted" in tsv
    assert "cifar10\t0\tcompleted" in tsv


def test_run_tool_clis(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "runs" / "full_model"
    run_dir = _write_run(root, "mnist", 0, 0.95)

    evaluate = subprocess.run(
        [sys.executable, "scripts/evaluate_run.py", str(run_dir), "--json"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert '"dataset": "mnist"' in evaluate.stdout

    summary = subprocess.run(
        [sys.executable, "scripts/summarize_runs.py", str(root), "--format", "markdown"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert "| dataset | seed | status |" in summary.stdout
