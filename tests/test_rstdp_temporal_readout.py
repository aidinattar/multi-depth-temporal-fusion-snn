from __future__ import annotations

import numpy as np

from temporal_evidence_snn.fused_sparse_code import save_sparse_latency_payload
from temporal_evidence_snn.readout import (
    evaluate_rstdp_temporal_readout,
    train_rstdp_temporal_readout,
)


def test_rstdp_temporal_readout_learns_separable_sparse_codes(tmp_path) -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.int16)
    train = np.full((4, 6), np.inf, dtype=np.float32)
    train[0, [0, 1]] = [0.1, 0.2]
    train[2, [0, 1]] = [0.2, 0.3]
    train[1, [4, 5]] = [0.1, 0.2]
    train[3, [4, 5]] = [0.2, 0.3]
    test = train.copy()

    feature_dir = tmp_path / "fused_sparse_code"
    save_sparse_latency_payload(feature_dir / "trainset.npy", train, labels)
    save_sparse_latency_payload(feature_dir / "testset.npy", test, labels)

    config = {
        "network": [
            {
                "n_neurons": 6,
                "firing_threshold": 0.8,
                "w_init_normal": True,
                "w_init_mean": 0.3,
                "w_init_std": 0.01,
                "w_norm": True,
                "w_min": 0.0,
                "w_max": 1.0,
                "forward_mode": "vectorized",
            }
        ],
        "optimizer": {
            "method": "rstdp",
            "mode": "temporal_target_hard_negative_multiproto",
            "stdp": "additive",
            "ap": 0.1,
            "am": -0.05,
            "adaptive_lr": False,
            "annealing": 0.75,
            "non_target_scale": 0.32,
            "update_scale": 1.0,
            "temporal_margin": 0.005,
            "temporal_scale_cap": 0.005,
            "temporal_contrast_target_scale": 2.0,
            "temporal_floor_scale": 0.0,
            "temporal_stability_min_spikes": 3.0,
            "temporal_guard_anti_scale": 0.0,
            "temporal_guard_target_scale": 1.25,
            "temporal_correct_update_scale": 0.1,
            "temporal_confident_correct_scale": 0.0,
            "temporal_hard_negative_k": 1,
            "temporal_target_prototypes": 1,
            "temporal_negative_prototypes": 1,
            "temporal_prototype_decay": 0.5,
        },
        "regularizer": {"thr_lr": 0.05, "thr_anneal": 0.5},
        "trainer": {
            "epochs": 4,
            "nt_neurons": 1,
            "early_stopping": 0,
            "progress_every": 0,
        },
    }
    metrics, summary = train_rstdp_temporal_readout(
        feature_dir=feature_dir,
        output_dir=tmp_path / "run",
        config=config,
        seed=0,
        epochs=4,
    )

    assert len(metrics) == 4
    assert summary["status"] == "completed"
    assert summary["best_test_accuracy"] >= 0.5
    assert (tmp_path / "run" / "checkpoints" / "best_readout.pt").is_file()
    checkpoint_accuracy = evaluate_rstdp_temporal_readout(
        feature_dir=feature_dir,
        checkpoint_path=tmp_path / "run" / "checkpoints" / "best_readout.pt",
        config=config,
    )
    assert checkpoint_accuracy == summary["best_test_accuracy"]
