from __future__ import annotations

from pathlib import Path

import pytest

from temporal_evidence_snn.config import (
    load_resolved_config,
    validate_experiment_config,
)


@pytest.mark.parametrize(
    "config_name",
    (
        "full_model_mnist.yaml",
        "full_model_fashion_mnist.yaml",
        "full_model_cifar10.yaml",
        "full_model_nmnist.yaml",
    ),
)
def test_full_model_config_is_complete(config_name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_resolved_config(root / "configs/experiments" / config_name)

    validate_experiment_config(config)
    assert tuple(config["model"]["backbone_config"]["stages"]) == (
        "base_evidence",
        "residual_evidence",
        "deep_transform",
        "deep_correction",
    )
