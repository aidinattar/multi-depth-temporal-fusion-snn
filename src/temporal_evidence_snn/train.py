from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from temporal_evidence_snn.backbone import (
    base_evidence_extractor_from_config,
    export_base_evidence_features,
    export_local_branch_features_from_sparse,
    local_branch_extractor_from_config,
)
from temporal_evidence_snn.config import (
    load_resolved_config,
    validate_experiment_config,
    write_yaml,
)
from temporal_evidence_snn.data import load_dataset_splits
from temporal_evidence_snn.frontend import (
    event_surface_frontend_from_config,
    static_latency_frontend_from_config,
)
from temporal_evidence_snn.fused_sparse_code import build_fused_sparse_feature_dir
from temporal_evidence_snn.readout import train_rstdp_temporal_readout


def _dataset_name(config: dict[str, Any]) -> str:
    dataset = config.get("dataset", {})
    name = dataset.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Experiment config must define dataset.name")
    return name


def _output_dir(config: dict[str, Any], seed: int, output_root: Path | None) -> Path:
    dataset = _dataset_name(config)
    root_value = config.get("output", {}).get("root", "runs/full_model")
    root = output_root or Path(root_value)
    return root / dataset / f"seed_{seed}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _configure_run_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"temporal_evidence_snn.{run_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _log_stage_summary(
    logger: logging.Logger,
    *,
    stage_name: str,
    summary: dict[str, Any],
) -> None:
    test = summary["splits"]["testset"]
    logger.info(
        "completed stage=%s feature_dim=%d test_mean_spikes=%.3f",
        stage_name,
        int(test["feature_dim"]),
        float(test["sample_spike_count_mean"]),
    )


def _frontend_output_channels(frontend) -> int:
    return int(frontend.output_channels)


def _fused_code_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {}).get("config", {})
    fused = model.get("fused_sparse_code", {})
    selected_early = fused.get("residual_evidence", {})
    selected_deep = fused.get("selected_deep_correction", {})
    return {
        "early_events_per_sample": int(selected_early.get("max_events_per_sample", 128)),
        "correction_events_per_sample": int(selected_deep.get("max_events_per_sample", 16)),
        "agreement_margin": float(selected_deep.get("agreement_margin", 0.001)),
    }


def _backbone_config(config: dict[str, Any]) -> dict[str, Any]:
    backbone = config.get("model", {}).get("backbone_config", {})
    if not isinstance(backbone, dict) or "stages" not in backbone:
        raise ValueError("Experiment config must resolve model.backbone_config")
    return backbone


def _readout_config(config: dict[str, Any]) -> dict[str, Any]:
    readout = config.get("readout", {})
    resolved = readout.get("config", {})
    if not isinstance(resolved, dict):
        raise ValueError("Experiment config must resolve readout.config")
    return resolved


def _metric_record(
    *,
    kind: str,
    config: dict[str, Any],
    seed: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    train_split = summary["splits"]["trainset"]
    test_split = summary["splits"]["testset"]
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "dataset": _dataset_name(config),
        "seed": int(seed),
        "train_samples": train_split["num_samples"],
        "test_samples": test_split["num_samples"],
        "feature_dim": train_split["feature_dim"],
        "train_mean_spikes_per_sample": train_split["sample_spike_count_mean"],
        "test_mean_spikes_per_sample": test_split["sample_spike_count_mean"],
    }


def _write_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _backbone_epochs_completed(
    *,
    config: dict[str, Any],
    stage_summaries: dict[str, Any],
) -> int:
    stages = _backbone_config(config).get("stages", {})
    return sum(
        int(stages.get(stage_name, {}).get("epochs", 0))
        for stage_name in stage_summaries
        if stage_name in stages
    )


def _run_full_pipeline(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    seed: int,
    run_dir: Path,
    logger: logging.Logger,
) -> None:
    frontend_cfg = config.get("frontend", {}).get("config", {})
    input_type = str(frontend_cfg.get("input_type", ""))
    if input_type not in {"grayscale_image", "rgb_image", "event_stream"}:
        raise NotImplementedError(f"Unsupported front-end input_type={input_type!r}.")

    logger.info("loading dataset=%s", _dataset_name(config))
    dataset = load_dataset_splits(
        config.get("dataset", {}),
        dataset_root=Path(args.dataset_root).resolve(),
        dataset_npz=Path(args.dataset_npz).resolve() if args.dataset_npz else None,
        train_limit=int(args.train_limit),
        test_limit=int(args.test_limit),
        download=bool(args.download_data),
    )
    if input_type == "event_stream":
        frontend = event_surface_frontend_from_config(frontend_cfg)
    else:
        frontend = static_latency_frontend_from_config(frontend_cfg)
    backbone_cfg = _backbone_config(config)
    base_extractor = base_evidence_extractor_from_config(
        frontend=frontend,
        backbone_config=backbone_cfg,
        input_channels=_frontend_output_channels(frontend),
        seed=int(seed),
    )

    metrics: list[dict[str, Any]] = []
    stage_summaries: dict[str, Any] = {}
    target_stage = args.stop_after or "readout"
    feature_batch_size = int(config.get("training", {}).get("feature_batch_size", 64))

    logger.info("starting stage=base_evidence")
    base_summary = export_base_evidence_features(
        output_dir=run_dir / "base_evidence",
        extractor=base_extractor,
        train_images=dataset.train_images,
        train_labels=dataset.train_labels,
        test_images=dataset.test_images,
        test_labels=dataset.test_labels,
        batch_size=feature_batch_size,
        progress=lambda message: logger.info("stage=base_evidence %s", message),
    )
    torch.save(
        base_extractor.state_dict(),
        run_dir / "checkpoints" / "base_evidence.pt",
    )
    metrics.append(
        _metric_record(
            kind="base_evidence_export",
            config=config,
            seed=seed,
            summary=base_summary,
        )
    )
    stage_summaries["base_evidence"] = base_summary["splits"]
    final_stage = "base_evidence"
    final_feature_summary = base_summary
    _log_stage_summary(logger, stage_name="base_evidence", summary=base_summary)

    if target_stage in {"residual-evidence", "deep-correction", "fused-sparse-code", "readout"}:
        logger.info("starting stage=residual_evidence")
        base_channels = int(backbone_cfg["stages"]["base_evidence"]["maps"])
        residual_evidence = local_branch_extractor_from_config(
            backbone_config=backbone_cfg,
            stage_name="residual_evidence",
            input_channels=base_channels,
            seed=int(seed),
        )
        residual_summary = export_local_branch_features_from_sparse(
            output_dir=run_dir / "residual_evidence",
            extractor=residual_evidence,
            input_dir=run_dir / "base_evidence",
            input_channels=base_channels,
            batch_size=feature_batch_size,
            progress=lambda message: logger.info("stage=residual_evidence %s", message),
        )
        torch.save(
            residual_evidence.state_dict(),
            run_dir / "checkpoints" / "residual_evidence.pt",
        )
        metrics.append(
            _metric_record(
                kind="residual_evidence_export",
                config=config,
                seed=seed,
                summary=residual_summary,
            )
        )
        stage_summaries["residual_evidence"] = residual_summary["splits"]
        final_stage = "residual_evidence"
        final_feature_summary = residual_summary
        _log_stage_summary(
            logger,
            stage_name="residual_evidence",
            summary=residual_summary,
        )

    if target_stage in {"deep-correction", "fused-sparse-code", "readout"}:
        current_dir = run_dir / "residual_evidence"
        current_channels = int(backbone_cfg["stages"]["residual_evidence"]["maps"])
        for stage_name in ("deep_transform", "deep_correction"):
            logger.info("starting stage=%s", stage_name)
            deep_stage = local_branch_extractor_from_config(
                backbone_config=backbone_cfg,
                stage_name=stage_name,
                input_channels=current_channels,
                seed=int(seed),
            )
            branch_summary = export_local_branch_features_from_sparse(
                output_dir=run_dir / stage_name,
                extractor=deep_stage,
                input_dir=current_dir,
                input_channels=current_channels,
                batch_size=feature_batch_size,
                progress=lambda message, stage=stage_name: logger.info(
                    "stage=%s %s", stage, message
                ),
            )
            torch.save(
                deep_stage.state_dict(),
                run_dir / "checkpoints" / f"{stage_name}.pt",
            )
            metrics.append(
                _metric_record(
                    kind=f"{stage_name}_export",
                    config=config,
                    seed=seed,
                    summary=branch_summary,
                )
            )
            stage_summaries[stage_name] = branch_summary["splits"]
            final_stage = stage_name
            final_feature_summary = branch_summary
            current_dir = run_dir / stage_name
            current_channels = int(backbone_cfg["stages"][stage_name]["maps"])
            _log_stage_summary(
                logger,
                stage_name=stage_name,
                summary=branch_summary,
            )

    if target_stage in {"fused-sparse-code", "readout"}:
        logger.info("starting stage=fused_sparse_code")
        routing = _fused_code_config(config)
        fused_summary = build_fused_sparse_feature_dir(
            output_dir=run_dir / "fused_sparse_code",
            base_evidence_dir=run_dir / "base_evidence",
            early_branch_dir=run_dir / "residual_evidence",
            deep_branch_dirs=[run_dir / "deep_correction"],
            early_events_per_sample=routing["early_events_per_sample"],
            correction_events_per_sample=routing["correction_events_per_sample"],
            agreement_margin=routing["agreement_margin"],
        )
        metrics.append(
            _metric_record(
                kind="fused_sparse_code_export",
                config=config,
                seed=seed,
                summary=fused_summary,
            )
        )
        stage_summaries["fused_sparse_code"] = fused_summary["splits"]
        final_stage = "fused_sparse_code"
        final_feature_summary = fused_summary
        _log_stage_summary(
            logger,
            stage_name="fused_sparse_code",
            summary=fused_summary,
        )

    readout_result: dict[str, Any] | None = None
    if target_stage == "readout":
        logger.info("starting stage=temporal_readout")
        training_cfg = config.get("training", {})
        readout_epochs = (
            int(args.epochs) if int(args.epochs) > 0 else int(training_cfg.get("epochs", 1))
        )
        readout_metrics, readout_result = train_rstdp_temporal_readout(
            feature_dir=run_dir / "fused_sparse_code",
            output_dir=run_dir,
            config=_readout_config(config),
            seed=int(seed),
            epochs=readout_epochs,
            train_limit=int(args.train_limit),
            test_limit=int(args.test_limit),
            shuffle_train=bool(
                args.shuffle_train or config.get("readout", {}).get("shuffle_train", False)
            ),
            device=str(training_cfg.get("device", "cpu")),
        )
        metrics.extend({"kind": "readout_epoch", **record} for record in readout_metrics)
        final_stage = "readout"
        logger.info(
            "completed stage=temporal_readout best_test_accuracy=%.4f final_test_accuracy=%.4f",
            float(readout_result["best_test_accuracy"]),
            float(readout_result["final_test_accuracy"]),
        )

    final_test_split = final_feature_summary["splits"]["testset"]
    epochs_completed = _backbone_epochs_completed(
        config=config,
        stage_summaries=stage_summaries,
    )
    if readout_result is not None:
        epochs_completed += int(readout_result.get("epochs_completed") or 0)

    run_summary = {
        "dataset": _dataset_name(config),
        "seed": int(seed),
        "model": config.get("model", {}).get("config", {}).get("name", "unknown"),
        "status": "completed" if final_stage == "readout" else f"{final_stage}_completed",
        "backend": "full",
        "last_completed_stage": final_stage,
        "best_test_accuracy": None
        if readout_result is None
        else readout_result.get("best_test_accuracy"),
        "final_test_accuracy": None
        if readout_result is None
        else readout_result.get("final_test_accuracy"),
        "best_epoch": None if readout_result is None else readout_result.get("best_epoch"),
        "epochs_completed": int(epochs_completed),
        "feature_dim": final_test_split["feature_dim"],
        "mean_spikes_per_sample": final_test_split["sample_spike_count_mean"],
        "density_percent": float(final_test_split["feature_density_mean"]) * 100.0,
        "stage_summaries": stage_summaries,
    }
    if readout_result is not None:
        run_summary["readout_checkpoint"] = readout_result.get("readout_checkpoint")

    _write_metrics(run_dir / "metrics.jsonl", metrics)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info("run completed status=%s", run_summary["status"])
    if readout_result is None:
        _write_text(run_dir / "checkpoints" / ".gitkeep", "")


def run(args: argparse.Namespace) -> Path:
    experiment = Path(args.experiment)
    config = load_resolved_config(experiment)
    validate_experiment_config(config)
    seed = int(args.seed if args.seed is not None else config.get("training", {}).get("seed", 0))
    config.setdefault("training", {})["seed"] = seed

    run_dir = _output_dir(config, seed, Path(args.output_root) if args.output_root else None)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    logger = _configure_run_logger(run_dir)

    command = " ".join(shlex.quote(part) for part in sys.argv)
    _write_text(run_dir / "command.txt", command + "\n")
    write_yaml(run_dir / "resolved_config.yaml", config)
    logger.info("command=%s", command)
    logger.info("run_dir=%s", run_dir.resolve())
    logger.info("seed=%d", seed)

    _run_full_pipeline(
        args=args,
        config=config,
        seed=seed,
        run_dir=run_dir,
        logger=logger,
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the full residual/agreement SNN.")
    parser.add_argument("--experiment", required=True, help="Path to experiment YAML.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    parser.add_argument("--backend", choices=["full"], default="full", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-npz", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dataset-root", default="data", help="Dataset root directory.")
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Allow torchvision to download MNIST, Fashion-MNIST, or CIFAR-10.",
    )
    parser.add_argument(
        "--stop-after",
        choices=["base-evidence", "residual-evidence", "deep-correction", "fused-sparse-code"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--shuffle-train", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = run(args)
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
