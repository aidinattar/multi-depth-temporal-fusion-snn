from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}")
    return data


def _resolve_config_reference(value: str, root: Path) -> Any:
    path = root / value
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return load_resolved_config(path, root)
    return value


def _resolve_node(node: Any, root: Path) -> Any:
    if isinstance(node, dict):
        return {key: _resolve_node(value, root) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_node(value, root) for value in node]
    if isinstance(node, str):
        return _resolve_config_reference(node, root)
    return node


def load_resolved_config(path: Path, root: Path | None = None) -> dict[str, Any]:
    root = root or path.parent.parent.parent
    raw = load_yaml(path)
    return _resolve_node(deepcopy(raw), root)


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validate an end-to-end experiment configuration before training."""

    dataset_name = config.get("dataset", {}).get("name")
    if dataset_name not in {"mnist", "fashion_mnist", "cifar10", "nmnist"}:
        raise ValueError(f"Unsupported dataset.name={dataset_name!r}.")

    frontend = config.get("frontend", {}).get("config", {})
    if frontend.get("input_type") not in {
        "grayscale_image",
        "rgb_image",
        "event_stream",
    }:
        raise ValueError("The experiment must define a supported front end.")

    model = config.get("model", {}).get("config", {})
    if model.get("name") != "full_residual_agreement_snn":
        raise ValueError("The experiment must use full_residual_agreement_snn.")
    backbone = config.get("model", {}).get("backbone_config", {})
    stages = backbone.get("stages", {})
    required_stages = {
        "base_evidence",
        "residual_evidence",
        "deep_transform",
        "deep_correction",
    }
    missing_stages = required_stages.difference(stages)
    if missing_stages:
        raise ValueError(
            "Backbone configuration is missing stages: " + ", ".join(sorted(missing_stages))
        )

    readout = config.get("readout", {}).get("config", {})
    optimizer = readout.get("optimizer", {})
    if optimizer.get("method") != "rstdp":
        raise ValueError("The readout must use R-STDP.")
    if optimizer.get("mode") != "temporal_target_hard_negative_multiproto":
        raise ValueError("The readout must use temporal multiprototype routing.")
    if int(config.get("training", {}).get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive.")


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
