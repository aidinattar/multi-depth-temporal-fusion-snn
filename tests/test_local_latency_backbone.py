from __future__ import annotations

import numpy as np
import torch

from temporal_evidence_snn.backbone import (
    BaseEvidenceConfig,
    BaseEvidenceExtractor,
    base_evidence_extractor_from_config,
    export_base_evidence_features,
)
from temporal_evidence_snn.frontend import StaticLatencyFrontend, StaticLatencyFrontendConfig
from temporal_evidence_snn.fused_sparse_code import sparse_payload_to_latency_matrix
from temporal_evidence_snn.layers import (
    LocalLatencyConvConfig,
    LocalLatencyConvLayer,
    flatten_latency_map,
    min_pool_latency,
)


def test_local_latency_conv_infers_threshold_crossing_times() -> None:
    layer = LocalLatencyConvLayer(
        LocalLatencyConvConfig(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            threshold=1.0,
            weight_init_mean=0.0,
            weight_init_std=0.0,
            seed=0,
        )
    )
    layer.weights[...] = 0.6
    latencies = np.array(
        [
            [
                [[0.10, 0.20], [np.inf, 0.80]],
            ],
        ],
        dtype=np.float32,
    )

    output = layer.infer_latency_map(latencies)

    assert output.shape == (1, 1, 1, 1)
    assert abs(float(output[0, 0, 0, 0]) - 0.20) < 1.0e-6


def test_local_latency_conv_fit_updates_weights_and_keeps_bounds() -> None:
    layer = LocalLatencyConvLayer(
        LocalLatencyConvConfig(
            in_channels=1,
            out_channels=2,
            kernel_size=2,
            threshold=0.5,
            weight_init_mean=0.5,
            weight_init_std=0.0,
            seed=1,
        )
    )
    before = layer.weights.clone()
    latencies = np.zeros((4, 1, 4, 4), dtype=np.float32)

    history = layer.fit(latencies, epochs=2)

    assert len(history) == 2
    assert not torch.equal(before, layer.weights)
    assert float(layer.weights.min()) >= 0.0
    assert float(layer.weights.max()) <= 1.0


def test_streamed_training_matches_dense_training() -> None:
    config = LocalLatencyConvConfig(
        in_channels=2,
        out_channels=4,
        kernel_size=2,
        threshold=0.9,
        weight_init_mean=0.3,
        weight_init_std=0.01,
        seed=9,
    )
    rng = np.random.default_rng(12)
    latencies = rng.uniform(0.0, 1.0, size=(7, 2, 5, 5)).astype(np.float32)
    latencies[rng.random(latencies.shape) < 0.2] = np.inf
    dense = LocalLatencyConvLayer(config)
    streamed = LocalLatencyConvLayer(config)

    dense.fit(latencies, epochs=2)
    streamed.fit_stream(
        lambda: (latencies[start : start + 3] for start in range(0, 7, 3)),
        epochs=2,
    )

    assert torch.equal(streamed.weights, dense.weights)
    assert torch.equal(streamed.thresholds, dense.thresholds)
    assert (
        streamed.state_dict()["learning_rate_potentiation"]
        == dense.state_dict()["learning_rate_potentiation"]
    )


def test_min_pool_and_flatten_latency_map() -> None:
    latencies = np.array(
        [
            [
                [[0.4, 0.2], [np.inf, 0.9]],
            ],
        ],
        dtype=np.float32,
    )

    pooled = min_pool_latency(latencies, kernel_size=2, stride=2)
    flattened = flatten_latency_map(pooled)

    assert pooled.shape == (1, 1, 1, 1)
    assert abs(float(pooled[0, 0, 0, 0]) - 0.2) < 1.0e-6
    assert flattened.shape == (1, 1)


def test_export_base_evidence_features_writes_sparse_contract(tmp_path) -> None:
    train_images = np.zeros((4, 1, 8, 8), dtype=np.float32)
    train_images[:, :, 2:6, 2:6] = 1.0
    test_images = np.zeros((2, 1, 8, 8), dtype=np.float32)
    test_images[:, :, 1:7, 3:5] = 1.0

    extractor = BaseEvidenceExtractor(
        frontend=StaticLatencyFrontend(
            StaticLatencyFrontendConfig(
                input_channels=1,
                whitening_patch_size=3,
                whitening_stride=1,
                whitening_max_samples=512,
                fit_batch_images=4,
                time_steps=11,
            )
        ),
        config=BaseEvidenceConfig(
            local_conv=LocalLatencyConvConfig(
                in_channels=2,
                out_channels=3,
                kernel_size=3,
                threshold=1.0,
                seed=2,
            ),
            pooling_kernel_size=2,
            pooling_stride=2,
            epochs=1,
        ),
    )

    summary = export_base_evidence_features(
        output_dir=tmp_path / "base_evidence",
        extractor=extractor,
        train_images=train_images,
        train_labels=np.array([0, 1, 0, 1], dtype=np.int16),
        test_images=test_images,
        test_labels=np.array([0, 1], dtype=np.int16),
    )

    output_dir = tmp_path / "base_evidence"
    assert (output_dir / "trainset.npy").is_file()
    assert (output_dir / "testset.npy").is_file()
    assert (output_dir / "spiking_dataset_summary.json").is_file()
    assert summary["representation"] == "base_evidence"
    features, labels = sparse_payload_to_latency_matrix(
        np.load(output_dir / "trainset.npy", allow_pickle=True).item()
    )
    assert features.shape[0] == 4
    assert labels.tolist() == [0, 1, 0, 1]


def test_base_evidence_extractor_builder_reads_backbone_config() -> None:
    frontend = StaticLatencyFrontend(StaticLatencyFrontendConfig(input_channels=1))
    extractor = base_evidence_extractor_from_config(
        frontend=frontend,
        input_channels=2,
        seed=3,
        backbone_config={
            "stages": {
                "base_evidence": {
                    "maps": 5,
                    "kernel_size": 3,
                    "threshold": 2.0,
                    "pooling_kernel_size": 2,
                    "pooling_stride": 2,
                    "epochs": 4,
                }
            }
        },
    )

    assert extractor.local_conv.config.in_channels == 2
    assert extractor.local_conv.config.out_channels == 5
    assert extractor.config.pooling_kernel_size == 2
    assert extractor.config.epochs == 4
