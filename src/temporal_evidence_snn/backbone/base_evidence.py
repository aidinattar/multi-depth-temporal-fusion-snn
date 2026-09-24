from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from temporal_evidence_snn.frontend import StaticLatencyFrontend
from temporal_evidence_snn.fused_sparse_code import (
    latency_matrix_to_sparse_payload,
    load_sparse_latency_payload,
    save_sparse_latency_payload,
)
from temporal_evidence_snn.layers import (
    LocalLatencyConvConfig,
    LocalLatencyConvLayer,
    flatten_latency_map,
    min_pool_latency,
)


@dataclass(frozen=True)
class BaseEvidenceConfig:
    local_conv: LocalLatencyConvConfig
    pooling_kernel_size: int = 4
    pooling_stride: int = 4
    epochs: int = 1


@dataclass(frozen=True)
class LocalBranchConfig:
    local_conv: LocalLatencyConvConfig
    representation: str
    pooling_kernel_size: int = 1
    pooling_stride: int = 1
    epochs: int = 1


class SparseLatencyMapSource:
    """Repeatable batched view over a sparse latency-map export."""

    def __init__(self, path: Path, *, channels: int, batch_size: int = 64) -> None:
        self.path = Path(path)
        self.payload = load_sparse_latency_payload(self.path)
        self.labels = np.asarray(self.payload["labels"], dtype=np.int16)
        self.feature_dim = int(self.payload["shape"][1])
        self.channels = int(channels)
        spatial_area, remainder = divmod(self.feature_dim, self.channels)
        spatial_size = int(round(spatial_area**0.5))
        if remainder or spatial_size * spatial_size != spatial_area:
            raise ValueError(
                f"Cannot reshape {self.feature_dim} features into {self.channels} square maps"
            )
        self.spatial_size = spatial_size
        self.batch_size = max(1, int(batch_size))

    def iter_batches(self) -> Iterator[np.ndarray]:
        records = self.payload["data"]
        index_dtype = np.int64
        for start in range(0, len(self.labels), self.batch_size):
            stop = min(start + self.batch_size, len(self.labels))
            dense = np.full(
                (stop - start, self.feature_dim),
                np.inf,
                dtype=np.float32,
            )
            for local_index, record in enumerate(records[start:stop]):
                indices = np.asarray(record["indices"], dtype=index_dtype)
                dense[local_index, indices] = np.asarray(
                    record["timestamps"],
                    dtype=np.float32,
                )
            yield dense.reshape(
                stop - start,
                self.channels,
                self.spatial_size,
                self.spatial_size,
            )


def _array_batch_factory(
    values: np.ndarray,
    *,
    batch_size: int,
) -> Callable[[], Iterator[np.ndarray]]:
    size = max(1, int(batch_size))

    def batches() -> Iterator[np.ndarray]:
        for start in range(0, int(values.shape[0]), size):
            yield values[start : start + size]

    return batches


def _write_transformed_batches(
    *,
    path: Path,
    labels: np.ndarray,
    batch_factory: Callable[[], Iterator[np.ndarray]],
    transform: Callable[[np.ndarray], np.ndarray],
) -> dict:
    records = []
    feature_dim = None
    for batch in batch_factory():
        transformed = transform(batch)
        payload = latency_matrix_to_sparse_payload(
            transformed,
            np.zeros((transformed.shape[0],), dtype=np.int16),
        )
        records.extend(payload["data"])
        if feature_dim is None:
            feature_dim = int(payload["shape"][1])
    if feature_dim is None:
        raise ValueError(f"Cannot export an empty split to {path}")

    labels_array = np.asarray(labels, dtype=np.int16)
    payload = {
        "data": records,
        "labels": labels_array,
        "shape": (len(records), int(feature_dim)),
        "max_time": 1.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, payload, allow_pickle=True)
    return _feature_summary_from_sparse_payload(path, payload)


def _feature_summary_from_sparse_payload(path: Path, payload: dict) -> dict:
    labels = np.asarray(payload["labels"], dtype=np.int16)
    counts = np.asarray([len(record) for record in payload["data"]], dtype=np.int64)
    feature_dim = int(payload["shape"][1])
    value_count = 0
    value_sum = 0.0
    value_sum_sq = 0.0
    value_min = float("inf")
    value_max = -float("inf")
    for record in payload["data"]:
        timestamps = np.asarray(record["timestamps"], dtype=np.float32)
        if timestamps.size == 0:
            continue
        values64 = timestamps.astype(np.float64)
        value_count += int(values64.size)
        value_sum += float(values64.sum())
        value_sum_sq += float(np.square(values64).sum())
        value_min = min(value_min, float(values64.min()))
        value_max = max(value_max, float(values64.max()))
    num_classes = int(labels.max()) + 1 if labels.size else 0
    class_counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(int)
    density = counts / max(feature_dim, 1)
    return {
        "path": str(path),
        "num_samples": int(labels.shape[0]),
        "feature_dim": feature_dim,
        "sample_spike_count_mean": float(counts.mean()) if counts.size else 0.0,
        "sample_spike_count_std": float(counts.std()) if counts.size else 0.0,
        "sample_spike_count_min": int(counts.min()) if counts.size else 0,
        "sample_spike_count_max": int(counts.max()) if counts.size else 0,
        "feature_density_mean": float(density.mean()) if density.size else 0.0,
        "feature_density_std": float(density.std()) if density.size else 0.0,
        "non_silent_sample_ratio": float((counts > 0).mean()) if counts.size else 0.0,
        "total_spikes": int(counts.sum()),
        "class_sample_counts": class_counts.tolist(),
        "timestamp_mean": value_sum / value_count if value_count else None,
        "timestamp_std": (
            max(0.0, value_sum_sq / value_count - (value_sum / value_count) ** 2) ** 0.5
            if value_count
            else None
        ),
        "timestamp_min": value_min if value_count else None,
        "timestamp_max": value_max if value_count else None,
    }


def _feature_summary(path: Path, features: np.ndarray, labels: np.ndarray) -> dict:
    finite = np.isfinite(features)
    counts = finite.sum(axis=1).astype(np.int64)
    values = features[finite]
    num_classes = int(labels.max()) + 1 if labels.size else 0
    class_counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(int)
    density = counts / max(int(features.shape[1]), 1)
    return {
        "path": str(path),
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "sample_spike_count_mean": float(counts.mean()) if counts.size else 0.0,
        "sample_spike_count_std": float(counts.std()) if counts.size else 0.0,
        "sample_spike_count_min": int(counts.min()) if counts.size else 0,
        "sample_spike_count_max": int(counts.max()) if counts.size else 0,
        "feature_density_mean": float(density.mean()) if density.size else 0.0,
        "feature_density_std": float(density.std()) if density.size else 0.0,
        "non_silent_sample_ratio": float((counts > 0).mean()) if counts.size else 0.0,
        "total_spikes": int(counts.sum()),
        "class_sample_counts": class_counts.tolist(),
        "timestamp_mean": float(values.mean()) if values.size else None,
        "timestamp_std": float(values.std()) if values.size else None,
        "timestamp_min": float(values.min()) if values.size else None,
        "timestamp_max": float(values.max()) if values.size else None,
    }


def _write_sparse_feature_dir(
    *,
    output_dir: Path,
    representation: str,
    training_history: list[dict[str, float]],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_labels_array = np.asarray(train_labels, dtype=np.int16)
    test_labels_array = np.asarray(test_labels, dtype=np.int16)

    train_path = output_dir / "trainset.npy"
    test_path = output_dir / "testset.npy"
    save_sparse_latency_payload(train_path, train_features, train_labels_array)
    save_sparse_latency_payload(test_path, test_features, test_labels_array)

    summary = {
        "representation": representation,
        "training_history": training_history,
        "splits": {
            "trainset": _feature_summary(train_path, train_features, train_labels_array),
            "testset": _feature_summary(test_path, test_features, test_labels_array),
        },
    }
    with (output_dir / "spiking_dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


class BaseEvidenceExtractor:
    """Static front-end plus first local latency-convolution stage."""

    def __init__(
        self,
        *,
        frontend: StaticLatencyFrontend,
        config: BaseEvidenceConfig,
    ) -> None:
        self.frontend = frontend
        self.config = config
        self.local_conv = LocalLatencyConvLayer(config.local_conv)
        self.training_history_: list[dict[str, float]] = []

    def fit(
        self,
        images: np.ndarray,
        *,
        batch_size: int = 64,
        progress: Callable[[str], None] | None = None,
    ) -> "BaseEvidenceExtractor":
        self.frontend.fit(images)
        image_batches = _array_batch_factory(images, batch_size=batch_size)
        self.training_history_ = self.local_conv.fit_stream(
            lambda: (self.frontend.transform_latency(batch) for batch in image_batches()),
            epochs=int(self.config.epochs),
            progress=progress,
        )
        return self

    def transform_latency(self, images: np.ndarray) -> np.ndarray:
        frontend_latency = self.frontend.transform_latency(images)
        conv_latency = self.local_conv.infer_latency_map(frontend_latency)
        return min_pool_latency(
            conv_latency,
            kernel_size=int(self.config.pooling_kernel_size),
            stride=int(self.config.pooling_stride),
        )

    def transform_features(self, images: np.ndarray) -> np.ndarray:
        return flatten_latency_map(self.transform_latency(images))

    def state_dict(self) -> dict:
        frontend_state = {}
        for name in ("whitening_weight_", "scaling_min_", "scaling_max_"):
            value = getattr(self.frontend, name, None)
            frontend_state[name] = None if value is None else value.clone()
        return {
            "frontend": frontend_state,
            "local_conv": self.local_conv.state_dict(),
        }


class LocalBranchExtractor:
    """Locally trained branch operating on previously exported latency maps."""

    def __init__(self, *, config: LocalBranchConfig) -> None:
        self.config = config
        self.local_conv = LocalLatencyConvLayer(config.local_conv)
        self.training_history_: list[dict[str, float]] = []

    def fit(self, latency_maps: np.ndarray) -> "LocalBranchExtractor":
        self.training_history_ = self.local_conv.fit(
            latency_maps,
            epochs=int(self.config.epochs),
        )
        return self

    def fit_batches(
        self,
        batch_factory: Callable[[], Iterator[np.ndarray]],
        *,
        progress: Callable[[str], None] | None = None,
    ) -> "LocalBranchExtractor":
        self.training_history_ = self.local_conv.fit_stream(
            batch_factory,
            epochs=int(self.config.epochs),
            progress=progress,
        )
        return self

    def transform_latency(self, latency_maps: np.ndarray) -> np.ndarray:
        conv_latency = self.local_conv.infer_latency_map(latency_maps)
        return min_pool_latency(
            conv_latency,
            kernel_size=int(self.config.pooling_kernel_size),
            stride=int(self.config.pooling_stride),
        )

    def transform_features(self, latency_maps: np.ndarray) -> np.ndarray:
        return flatten_latency_map(self.transform_latency(latency_maps))

    def state_dict(self) -> dict:
        return {"local_conv": self.local_conv.state_dict()}


def export_base_evidence_features(
    *,
    output_dir: Path,
    extractor: BaseEvidenceExtractor,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 64,
    progress: Callable[[str], None] | None = None,
) -> dict:
    extractor.fit(
        train_images,
        batch_size=int(batch_size),
        progress=progress,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_summary = _write_transformed_batches(
        path=output_dir / "trainset.npy",
        labels=train_labels,
        batch_factory=_array_batch_factory(train_images, batch_size=batch_size),
        transform=extractor.transform_features,
    )
    test_summary = _write_transformed_batches(
        path=output_dir / "testset.npy",
        labels=test_labels,
        batch_factory=_array_batch_factory(test_images, batch_size=batch_size),
        transform=extractor.transform_features,
    )
    summary = {
        "representation": "base_evidence",
        "training_history": extractor.training_history_,
        "splits": {"trainset": train_summary, "testset": test_summary},
    }
    with (output_dir / "spiking_dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def export_local_branch_features(
    *,
    output_dir: Path,
    extractor: LocalBranchExtractor,
    train_latencies: np.ndarray,
    train_labels: np.ndarray,
    test_latencies: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    extractor.fit(train_latencies)
    train_features = extractor.transform_features(train_latencies)
    test_features = extractor.transform_features(test_latencies)
    return _write_sparse_feature_dir(
        output_dir=output_dir,
        representation=extractor.config.representation,
        training_history=extractor.training_history_,
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
    )


def export_local_branch_features_from_sparse(
    *,
    output_dir: Path,
    extractor: LocalBranchExtractor,
    input_dir: Path,
    input_channels: int,
    batch_size: int = 64,
    progress: Callable[[str], None] | None = None,
) -> dict:
    train_source = SparseLatencyMapSource(
        input_dir / "trainset.npy",
        channels=int(input_channels),
        batch_size=int(batch_size),
    )
    test_source = SparseLatencyMapSource(
        input_dir / "testset.npy",
        channels=int(input_channels),
        batch_size=int(batch_size),
    )
    extractor.fit_batches(train_source.iter_batches, progress=progress)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_summary = _write_transformed_batches(
        path=output_dir / "trainset.npy",
        labels=train_source.labels,
        batch_factory=train_source.iter_batches,
        transform=extractor.transform_features,
    )
    test_summary = _write_transformed_batches(
        path=output_dir / "testset.npy",
        labels=test_source.labels,
        batch_factory=test_source.iter_batches,
        transform=extractor.transform_features,
    )
    summary = {
        "representation": extractor.config.representation,
        "training_history": extractor.training_history_,
        "splits": {"trainset": train_summary, "testset": test_summary},
    }
    with (output_dir / "spiking_dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def base_evidence_extractor_from_config(
    *,
    frontend: StaticLatencyFrontend,
    backbone_config: dict,
    input_channels: int,
    seed: int = 0,
) -> BaseEvidenceExtractor:
    stages = backbone_config.get("stages", {})
    stage = stages.get("base_evidence")
    if not isinstance(stage, dict):
        raise ValueError("backbone config must define stages.base_evidence")

    layer_config = LocalLatencyConvConfig(
        in_channels=int(input_channels),
        out_channels=int(stage.get("maps", 128)),
        kernel_size=int(stage.get("kernel_size", 5)),
        stride=int(stage.get("stride", 1)),
        padding=int(stage.get("padding", 0)),
        threshold=float(stage.get("threshold", 5.0)),
        weight_init_mean=float(stage.get("weight_init_mean", 0.5)),
        weight_init_std=float(stage.get("weight_init_std", 0.01)),
        weight_init_mode=str(stage.get("weight_init_mode", "normal")),
        identity_gain=float(stage.get("identity_gain", 1.0)),
        identity_noise_std=float(stage.get("identity_noise_std", 0.0)),
        learning_rate_potentiation=float(stage.get("learning_rate_potentiation", 0.1)),
        learning_rate_depression=float(stage.get("learning_rate_depression", 0.1)),
        beta=float(stage.get("beta", 1.0)),
        threshold_target_time=float(stage.get("threshold_target_time", 0.95)),
        threshold_learning_rate=float(stage.get("threshold_learning_rate", 1.0)),
        min_threshold=float(stage.get("min_threshold", 2.0)),
        annealing=float(stage.get("annealing", 0.95)),
        inference_winner_take_all=bool(stage.get("inference_winner_take_all", False)),
        inference_top_k=int(stage.get("inference_top_k", 0)),
        patch_sampling_space=str(stage.get("patch_sampling_space", "reference_output")),
        seed=int(stage.get("initialization_seed", seed)),
    )
    return BaseEvidenceExtractor(
        frontend=frontend,
        config=BaseEvidenceConfig(
            local_conv=layer_config,
            pooling_kernel_size=int(stage.get("pooling_kernel_size", 4)),
            pooling_stride=int(stage.get("pooling_stride", 4)),
            epochs=int(stage.get("epochs", 1)),
        ),
    )


def local_branch_extractor_from_config(
    *,
    backbone_config: dict,
    stage_name: str,
    input_channels: int,
    seed: int = 0,
) -> LocalBranchExtractor:
    stages = backbone_config.get("stages", {})
    stage = stages.get(stage_name)
    if not isinstance(stage, dict):
        raise ValueError(f"backbone config must define stages.{stage_name}")

    layer_config = LocalLatencyConvConfig(
        in_channels=int(input_channels),
        out_channels=int(stage.get("maps", 256)),
        kernel_size=int(stage.get("kernel_size", 1)),
        stride=int(stage.get("stride", 1)),
        padding=int(stage.get("padding", 0)),
        threshold=float(stage.get("threshold", 5.0)),
        weight_init_mean=float(stage.get("weight_init_mean", 0.5)),
        weight_init_std=float(stage.get("weight_init_std", 0.01)),
        weight_init_mode=str(stage.get("weight_init_mode", "normal")),
        identity_gain=float(stage.get("identity_gain", 1.0)),
        identity_noise_std=float(stage.get("identity_noise_std", 0.0)),
        learning_rate_potentiation=float(stage.get("learning_rate_potentiation", 0.1)),
        learning_rate_depression=float(stage.get("learning_rate_depression", 0.1)),
        beta=float(stage.get("beta", 1.0)),
        threshold_target_time=float(stage.get("threshold_target_time", 0.95)),
        threshold_learning_rate=float(stage.get("threshold_learning_rate", 1.0)),
        min_threshold=float(stage.get("min_threshold", 2.0)),
        annealing=float(stage.get("annealing", 0.95)),
        inference_winner_take_all=bool(stage.get("inference_winner_take_all", False)),
        inference_top_k=int(stage.get("inference_top_k", 0)),
        patch_sampling_space=str(stage.get("patch_sampling_space", "input_valid")),
        seed=int(stage.get("initialization_seed", seed)),
    )
    return LocalBranchExtractor(
        config=LocalBranchConfig(
            local_conv=layer_config,
            representation=str(stage.get("representation", stage_name)),
            pooling_kernel_size=int(stage.get("pooling_kernel_size", 1)),
            pooling_stride=int(stage.get("pooling_stride", 1)),
            epochs=int(stage.get("epochs", 1)),
        )
    )
