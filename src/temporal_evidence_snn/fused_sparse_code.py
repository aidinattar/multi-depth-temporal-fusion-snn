from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FusedSparseCode:
    """Output of residual/agreement routing.

    All arrays are dense latency matrices with shape
    ``(num_samples, num_features)``. Silent cells are represented by ``np.inf``.
    """

    base_evidence: np.ndarray
    selected_early_evidence: np.ndarray
    selected_deep_correction: np.ndarray
    fused_sparse_code: np.ndarray


def _as_latency_matrix(name: str, value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D latency matrix, got shape {matrix.shape}")
    return matrix


def _check_same_shape(reference: np.ndarray, others: Iterable[tuple[str, np.ndarray]]) -> None:
    for name, value in others:
        if reference.shape != value.shape:
            raise ValueError(f"{name} must have shape {reference.shape}, got {value.shape}")


def select_earliest_events(latencies: np.ndarray, max_events_per_sample: int) -> np.ndarray:
    """Keep the earliest finite events in each sample.

    This preserves latency values and sets all non-selected cells to ``np.inf``.
    A non-positive ``max_events_per_sample`` keeps all finite events.
    """

    matrix = _as_latency_matrix("latencies", latencies)
    limit = int(max_events_per_sample)
    if limit <= 0:
        return matrix.copy()

    finite = np.isfinite(matrix)
    score = np.zeros_like(matrix, dtype=np.float32)
    score[finite] = 1.0 - matrix[finite]
    selected = np.zeros_like(finite, dtype=bool)
    for row_index in range(matrix.shape[0]):
        finite_index = np.nonzero(finite[row_index])[0]
        if finite_index.size <= limit:
            selected[row_index, finite_index] = True
            continue
        row_scores = score[row_index, finite_index]
        top = np.argpartition(row_scores, -limit)[-limit:]
        selected[row_index, finite_index[top]] = True
    output = matrix.copy()
    output[~selected] = np.inf
    return output


def select_agreement_events(
    latency_maps: Iterable[np.ndarray],
    *,
    agreement_margin: float,
    max_events_per_sample: int,
) -> np.ndarray:
    """Select sparse events that agree across all provided latency maps.

    A cell is retained only when every latency map fires and the temporal
    spread between the earliest and latest event is at most ``agreement_margin``.
    The retained latency is the earliest event in the agreeing group.
    """

    maps = [
        _as_latency_matrix(f"latency_maps[{index}]", value)
        for index, value in enumerate(latency_maps)
    ]
    if len(maps) < 2:
        raise ValueError("Agreement selection requires at least two latency maps.")
    _check_same_shape(
        maps[0],
        [(f"latency_maps[{index}]", value) for index, value in enumerate(maps[1:], 1)],
    )

    stack = np.stack(maps, axis=0)
    all_finite = np.isfinite(stack).all(axis=0)
    earliest = np.min(stack, axis=0)
    latest = np.max(stack, axis=0)
    spread = np.full_like(earliest, np.inf, dtype=np.float32)
    np.subtract(latest, earliest, out=spread, where=all_finite)

    output = earliest.astype(np.float32, copy=True)
    output[~(all_finite & (spread <= float(agreement_margin)))] = np.inf
    return select_earliest_events(output, max_events_per_sample)


def build_fused_sparse_code(
    *,
    base_evidence: np.ndarray,
    early_branch: np.ndarray,
    deep_branches: Iterable[np.ndarray],
    early_events_per_sample: int = 128,
    correction_events_per_sample: int = 16,
    agreement_margin: float = 0.001,
) -> FusedSparseCode:
    """Build the full model readout code from early and deep evidence.

    The fused representation concatenates three terms:

    1. the preserved base evidence;
    2. a sparse early branch selected by earliest latency;
    3. a sparse deep correction selected by agreement between the early branch
       and one or more deeper branches.
    """

    base = _as_latency_matrix("base_evidence", base_evidence)
    early = _as_latency_matrix("early_branch", early_branch)
    deep = [
        _as_latency_matrix(f"deep_branches[{index}]", value)
        for index, value in enumerate(deep_branches)
    ]
    if not deep:
        raise ValueError("At least one deep branch is required.")

    _check_same_shape(
        early,
        [(f"deep_branches[{index}]", value) for index, value in enumerate(deep)],
    )
    if base.shape[0] != early.shape[0]:
        raise ValueError(
            "base_evidence and early_branch must have the same number of samples, "
            f"got {base.shape[0]} and {early.shape[0]}"
        )

    selected_early = select_earliest_events(early, early_events_per_sample)
    selected_deep = select_agreement_events(
        [early, *deep],
        agreement_margin=agreement_margin,
        max_events_per_sample=correction_events_per_sample,
    )
    fused = np.concatenate([base, selected_early, selected_deep], axis=1)
    return FusedSparseCode(
        base_evidence=base.copy(),
        selected_early_evidence=selected_early,
        selected_deep_correction=selected_deep,
        fused_sparse_code=fused,
    )


def sparse_payload_to_latency_matrix(
    payload: dict,
    *,
    max_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a sparse latency payload to a dense latency matrix."""

    num_samples_total = int(payload["shape"][0])
    num_features = int(payload["shape"][1])
    num_samples = (
        num_samples_total if int(max_samples) <= 0 else min(num_samples_total, int(max_samples))
    )
    matrix = np.full((num_samples, num_features), np.inf, dtype=np.float32)
    for sample_index, sample in enumerate(payload["data"][:num_samples]):
        try:
            indices = sample["indices"]
            timestamps = sample["timestamps"]
        except Exception:
            indices = sample.indices
            timestamps = sample.timestamps
        indices = np.asarray(indices, dtype=np.int64)
        timestamps = np.asarray(timestamps, dtype=np.float32)
        if indices.size:
            matrix[sample_index, indices] = timestamps
    labels = np.asarray(payload["labels"][:num_samples], dtype=np.int16)
    return matrix, labels


def latency_matrix_to_sparse_payload(latencies: np.ndarray, labels: np.ndarray) -> dict:
    """Convert a dense latency matrix into the sparse feature format."""

    matrix = _as_latency_matrix("latencies", latencies)
    labels_array = np.asarray(labels, dtype=np.int16)
    if labels_array.ndim != 1 or labels_array.shape[0] != matrix.shape[0]:
        raise ValueError(
            "labels must be a 1D array with one label per sample, "
            f"got shape {labels_array.shape} for {matrix.shape[0]} samples"
        )

    index_dtype = np.int16 if int(matrix.shape[1]) <= np.iinfo(np.int16).max else np.int32
    records = []
    for row in matrix:
        finite = np.isfinite(row)
        indices = np.nonzero(finite)[0].astype(index_dtype, copy=False)
        timestamps = row[finite].astype(np.float32, copy=False)
        if timestamps.size:
            order = np.argsort(timestamps, kind="stable")
            indices = indices[order]
            timestamps = timestamps[order]
        records.append(np.rec.fromarrays([indices, timestamps], names="indices,timestamps"))

    return {
        "data": records,
        "labels": labels_array,
        "shape": tuple(int(value) for value in matrix.shape),
        "max_time": 1.0,
    }


def load_sparse_latency_payload(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


def save_sparse_latency_payload(path: Path, latencies: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, latency_matrix_to_sparse_payload(latencies, labels), allow_pickle=True)


def _resolve_feature_dir(path: Path) -> Path:
    if (path / "trainset.npy").is_file() and (path / "testset.npy").is_file():
        return path
    nested = path / "features" / "single"
    if (nested / "trainset.npy").is_file() and (nested / "testset.npy").is_file():
        return nested
    raise FileNotFoundError(f"Missing trainset.npy/testset.npy under {path}")


def _load_feature_split(
    feature_dir: Path,
    split_name: str,
    *,
    max_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    payload = load_sparse_latency_payload(feature_dir / f"{split_name}.npy")
    return sparse_payload_to_latency_matrix(payload, max_samples=max_samples)


def _check_labels(reference: np.ndarray, labels: np.ndarray, *, split_name: str) -> None:
    if reference.shape != labels.shape or not np.array_equal(reference, labels):
        raise ValueError(f"Labels do not match for split {split_name}")


def _split_summary(path: Path, latencies: np.ndarray, labels: np.ndarray) -> dict:
    finite = np.isfinite(latencies)
    counts = finite.sum(axis=1).astype(np.int64)
    values = latencies[finite]
    num_classes = int(labels.max()) + 1 if labels.size else 0
    class_counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(int)

    class_mean_spike_count = []
    for class_index in range(num_classes):
        mask = labels == class_index
        class_mean_spike_count.append(float(counts[mask].mean()) if bool(mask.any()) else None)

    if values.size:
        timestamp_summary = {
            "timestamp_mean": float(values.mean()),
            "timestamp_std": float(values.std()),
            "timestamp_min": float(values.min()),
            "timestamp_max": float(values.max()),
        }
    else:
        timestamp_summary = {
            "timestamp_mean": None,
            "timestamp_std": None,
            "timestamp_min": None,
            "timestamp_max": None,
        }

    density = counts / max(int(latencies.shape[1]), 1)
    return {
        "path": str(path),
        "num_samples": int(latencies.shape[0]),
        "feature_dim": int(latencies.shape[1]),
        "max_time": 1.0,
        "sample_spike_count_mean": float(counts.mean()) if counts.size else 0.0,
        "sample_spike_count_std": float(counts.std()) if counts.size else 0.0,
        "sample_spike_count_min": int(counts.min()) if counts.size else 0,
        "sample_spike_count_max": int(counts.max()) if counts.size else 0,
        "feature_density_mean": float(density.mean()) if density.size else 0.0,
        "feature_density_std": float(density.std()) if density.size else 0.0,
        "non_silent_sample_ratio": float((counts > 0).mean()) if counts.size else 0.0,
        "total_spikes": int(counts.sum()),
        "class_sample_counts": class_counts.tolist(),
        "class_mean_spike_count": class_mean_spike_count,
        **timestamp_summary,
    }


def _sparse_split_summary(path: Path, payload: dict) -> dict:
    labels = np.asarray(payload["labels"], dtype=np.int16)
    records = payload["data"]
    feature_dim = int(payload["shape"][1])
    counts = np.asarray([len(record) for record in records], dtype=np.int64)
    value_count = 0
    value_sum = 0.0
    value_sum_sq = 0.0
    value_min = float("inf")
    value_max = -float("inf")
    for record in records:
        timestamps = np.asarray(record["timestamps"], dtype=np.float32)
        if timestamps.size == 0:
            continue
        values64 = timestamps.astype(np.float64)
        value_count += int(values64.size)
        value_sum += float(values64.sum())
        value_sum_sq += float(np.square(values64).sum())
        value_min = min(value_min, float(values64.min()))
        value_max = max(value_max, float(values64.max()))
    if value_count:
        mean = value_sum / value_count
        variance = max(0.0, value_sum_sq / value_count - mean * mean)
        timestamp_summary = {
            "timestamp_mean": mean,
            "timestamp_std": variance**0.5,
            "timestamp_min": value_min,
            "timestamp_max": value_max,
        }
    else:
        timestamp_summary = {
            "timestamp_mean": None,
            "timestamp_std": None,
            "timestamp_min": None,
            "timestamp_max": None,
        }
    num_classes = int(labels.max()) + 1 if labels.size else 0
    class_counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(int)
    class_mean_spike_count = [
        float(counts[labels == class_index].mean()) if bool((labels == class_index).any()) else None
        for class_index in range(num_classes)
    ]
    density = counts / max(feature_dim, 1)
    return {
        "path": str(path),
        "num_samples": int(labels.shape[0]),
        "feature_dim": feature_dim,
        "max_time": 1.0,
        "sample_spike_count_mean": float(counts.mean()) if counts.size else 0.0,
        "sample_spike_count_std": float(counts.std()) if counts.size else 0.0,
        "sample_spike_count_min": int(counts.min()) if counts.size else 0,
        "sample_spike_count_max": int(counts.max()) if counts.size else 0,
        "feature_density_mean": float(density.mean()) if density.size else 0.0,
        "feature_density_std": float(density.std()) if density.size else 0.0,
        "non_silent_sample_ratio": float((counts > 0).mean()) if counts.size else 0.0,
        "total_spikes": int(counts.sum()),
        "class_sample_counts": class_counts.tolist(),
        "class_mean_spike_count": class_mean_spike_count,
        **timestamp_summary,
    }


def _sparse_record_to_dense(record, feature_dim: int) -> np.ndarray:
    dense = np.full((1, int(feature_dim)), np.inf, dtype=np.float32)
    indices = np.asarray(record["indices"], dtype=np.int64)
    dense[0, indices] = np.asarray(record["timestamps"], dtype=np.float32)
    return dense


def build_fused_sparse_feature_dir(
    *,
    output_dir: Path,
    base_evidence_dir: Path,
    early_branch_dir: Path,
    deep_branch_dirs: Iterable[Path],
    early_events_per_sample: int = 128,
    correction_events_per_sample: int = 16,
    agreement_margin: float = 0.001,
    max_train_samples: int = 0,
    max_test_samples: int = 0,
) -> dict:
    """Build train/test fused sparse feature files from branch feature dirs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = _resolve_feature_dir(base_evidence_dir)
    early_dir = _resolve_feature_dir(early_branch_dir)
    deep_dirs = [_resolve_feature_dir(path) for path in deep_branch_dirs]
    if not deep_dirs:
        raise ValueError("At least one deep branch directory is required.")

    split_limits = {
        "trainset": int(max_train_samples),
        "testset": int(max_test_samples),
    }
    split_summaries = {}
    for split_name, max_samples in split_limits.items():
        payloads = [
            load_sparse_latency_payload(directory / f"{split_name}.npy")
            for directory in (base_dir, early_dir, *deep_dirs)
        ]
        labels = np.asarray(payloads[0]["labels"], dtype=np.int16)
        if max_samples > 0:
            labels = labels[: int(max_samples)]
        for payload in payloads[1:]:
            other_labels = np.asarray(payload["labels"], dtype=np.int16)[: len(labels)]
            _check_labels(labels, other_labels, split_name=split_name)

        feature_dims = [int(payload["shape"][1]) for payload in payloads]
        if any(dim != feature_dims[1] for dim in feature_dims[2:]):
            raise ValueError(
                f"Early and deep feature dimensions differ for {split_name}: {feature_dims[1:]}"
            )
        fused_records = []
        fused_feature_dim = feature_dims[0] + feature_dims[1] + feature_dims[1]
        for sample_index in range(len(labels)):
            dense_rows = [
                _sparse_record_to_dense(
                    payload["data"][sample_index],
                    feature_dim,
                )
                for payload, feature_dim in zip(payloads, feature_dims)
            ]
            code = build_fused_sparse_code(
                base_evidence=dense_rows[0],
                early_branch=dense_rows[1],
                deep_branches=dense_rows[2:],
                early_events_per_sample=early_events_per_sample,
                correction_events_per_sample=correction_events_per_sample,
                agreement_margin=agreement_margin,
            )
            fused_records.extend(
                latency_matrix_to_sparse_payload(
                    code.fused_sparse_code,
                    labels[sample_index : sample_index + 1],
                )["data"]
            )

        split_path = output_dir / f"{split_name}.npy"
        fused_payload = {
            "data": fused_records,
            "labels": labels,
            "shape": (len(labels), fused_feature_dim),
            "max_time": 1.0,
        }
        np.save(split_path, fused_payload, allow_pickle=True)
        split_summaries[split_name] = _sparse_split_summary(split_path, fused_payload)

    summary = {
        "representation": "fused_sparse_code",
        "routing": {
            "early_events_per_sample": int(early_events_per_sample),
            "correction_events_per_sample": int(correction_events_per_sample),
            "agreement_margin": float(agreement_margin),
            "num_deep_branches": len(deep_dirs),
        },
        "sources": {
            "base_evidence_dir": str(base_dir),
            "early_branch_dir": str(early_dir),
            "deep_branch_dirs": [str(path) for path in deep_dirs],
        },
        "splits": split_summaries,
    }
    with (output_dir / "spiking_dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
