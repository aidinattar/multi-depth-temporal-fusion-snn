from __future__ import annotations

import numpy as np

from temporal_evidence_snn.fused_sparse_code import (
    build_fused_sparse_feature_dir,
    build_fused_sparse_code,
    latency_matrix_to_sparse_payload,
    save_sparse_latency_payload,
    select_agreement_events,
    select_earliest_events,
    sparse_payload_to_latency_matrix,
)


def test_select_earliest_events_keeps_earliest_finite_cells() -> None:
    latencies = np.array(
        [
            [0.30, 0.10, 0.20, np.inf],
            [np.inf, 0.90, 0.40, 0.20],
        ],
        dtype=np.float32,
    )

    selected = select_earliest_events(latencies, max_events_per_sample=2)

    assert np.isfinite(selected[0]).tolist() == [False, True, True, False]
    assert np.isfinite(selected[1]).tolist() == [False, False, True, True]
    np.testing.assert_allclose(selected[np.isfinite(selected)], [0.10, 0.20, 0.40, 0.20])


def test_select_agreement_events_keeps_only_temporally_consistent_cells() -> None:
    early = np.array(
        [
            [0.10, 0.30, 0.50],
            [0.20, 0.80, 0.70],
        ],
        dtype=np.float32,
    )
    deep = np.array(
        [
            [0.1005, 0.90, 0.20],
            [0.2010, 0.10, np.inf],
        ],
        dtype=np.float32,
    )

    correction = select_agreement_events(
        [early, deep],
        agreement_margin=0.002,
        max_events_per_sample=1,
    )

    assert np.isfinite(correction).tolist() == [
        [True, False, False],
        [True, False, False],
    ]
    np.testing.assert_allclose(correction[:, 0], [0.10, 0.20], atol=1.0e-6)


def test_build_fused_sparse_code_preserves_base_and_adds_sparse_correction() -> None:
    base = np.array(
        [
            [0.20, np.inf, 0.40],
            [np.inf, 0.90, np.inf],
        ],
        dtype=np.float32,
    )
    early = np.array(
        [
            [0.10, 0.30, 0.50],
            [0.20, 0.80, 0.70],
        ],
        dtype=np.float32,
    )
    deep = np.array(
        [
            [0.1005, 0.90, 0.20],
            [0.2010, 0.10, np.inf],
        ],
        dtype=np.float32,
    )

    code = build_fused_sparse_code(
        base_evidence=base,
        early_branch=early,
        deep_branches=[deep],
        early_events_per_sample=2,
        correction_events_per_sample=1,
        agreement_margin=0.002,
    )

    assert code.fused_sparse_code.shape == (2, 9)
    np.testing.assert_array_equal(code.base_evidence, base)
    assert np.isfinite(code.selected_early_evidence).sum(axis=1).tolist() == [2, 2]
    assert np.isfinite(code.selected_deep_correction).sum(axis=1).tolist() == [1, 1]
    np.testing.assert_array_equal(code.fused_sparse_code[:, :3], base)


def test_sparse_payload_round_trip_preserves_latencies_and_labels() -> None:
    latencies = np.array(
        [
            [np.inf, 0.15, 0.70, np.inf],
            [0.20, np.inf, np.inf, 0.10],
        ],
        dtype=np.float32,
    )
    labels = np.array([3, 1], dtype=np.int16)

    payload = latency_matrix_to_sparse_payload(latencies, labels)
    restored, restored_labels = sparse_payload_to_latency_matrix(payload)

    np.testing.assert_array_equal(np.isfinite(restored), np.isfinite(latencies))
    np.testing.assert_allclose(restored[np.isfinite(restored)], latencies[np.isfinite(latencies)])
    np.testing.assert_array_equal(restored_labels, labels)


def test_build_fused_sparse_feature_dir_writes_sparse_feature_contract(tmp_path) -> None:
    labels = np.array([0, 1], dtype=np.int16)
    base = np.array(
        [
            [0.20, np.inf],
            [np.inf, 0.90],
        ],
        dtype=np.float32,
    )
    early = np.array(
        [
            [0.10, 0.30, 0.50],
            [0.20, 0.80, 0.70],
        ],
        dtype=np.float32,
    )
    deep = np.array(
        [
            [0.1005, 0.90, 0.20],
            [0.2010, 0.10, np.inf],
        ],
        dtype=np.float32,
    )

    base_dir = tmp_path / "base_evidence"
    early_dir = tmp_path / "early_branch"
    deep_dir = tmp_path / "deep_branch"
    for feature_dir, matrix in (
        (base_dir, base),
        (early_dir, early),
        (deep_dir, deep),
    ):
        save_sparse_latency_payload(feature_dir / "trainset.npy", matrix, labels)
        save_sparse_latency_payload(feature_dir / "testset.npy", matrix, labels)

    summary = build_fused_sparse_feature_dir(
        output_dir=tmp_path / "fused_sparse_code",
        base_evidence_dir=base_dir,
        early_branch_dir=early_dir,
        deep_branch_dirs=[deep_dir],
        early_events_per_sample=2,
        correction_events_per_sample=1,
        agreement_margin=0.002,
    )

    output_dir = tmp_path / "fused_sparse_code"
    assert (output_dir / "trainset.npy").is_file()
    assert (output_dir / "testset.npy").is_file()
    assert (output_dir / "spiking_dataset_summary.json").is_file()
    assert summary["representation"] == "fused_sparse_code"
    assert summary["splits"]["trainset"]["feature_dim"] == 8

    restored, restored_labels = sparse_payload_to_latency_matrix(
        np.load(output_dir / "trainset.npy", allow_pickle=True).item()
    )
    assert restored.shape == (2, 8)
    np.testing.assert_array_equal(restored_labels, labels)
