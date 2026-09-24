from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LocalLatencyConvConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    threshold: float = 5.0
    weight_init_mean: float = 0.5
    weight_init_std: float = 0.01
    weight_init_mode: str = "normal"
    identity_gain: float = 1.0
    identity_noise_std: float = 0.0
    learning_rate_potentiation: float = 0.1
    learning_rate_depression: float = 0.1
    beta: float = 1.0
    weight_min: float = 0.0
    weight_max: float = 1.0
    threshold_target_time: float = 0.95
    threshold_learning_rate: float = 1.0
    min_threshold: float = 2.0
    annealing: float = 0.95
    inference_winner_take_all: bool = False
    inference_top_k: int = 0
    patch_sampling_space: str = "input_valid"
    seed: int = 0


def _as_latency_tensor(latencies: np.ndarray | torch.Tensor) -> torch.Tensor:
    maps = torch.as_tensor(latencies, dtype=torch.float32)
    if maps.ndim != 4:
        raise ValueError(f"Expected latency maps with shape (N,C,H,W), got {tuple(maps.shape)}")
    return maps


def _output_size(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
    return (int(input_size) + 2 * int(padding) - int(kernel_size)) // int(stride) + 1


def _weight_flat_hwc(weight: torch.Tensor) -> torch.Tensor:
    chunks = []
    for row in range(int(weight.shape[-2])):
        for col in range(int(weight.shape[-1])):
            chunks.append(weight[:, :, row, col])
    return torch.cat(chunks, dim=1)


def _patch_flat_hwc(patch: torch.Tensor) -> torch.Tensor:
    return patch.permute(1, 2, 0).contiguous().view(-1)


@torch.no_grad()
def _winner_from_patch(
    patch: torch.Tensor,
    weights: torch.Tensor,
    thresholds: torch.Tensor,
) -> tuple[bool, int, float]:
    flat = _patch_flat_hwc(patch)
    active = torch.isfinite(flat)
    if not bool(active.any().item()):
        return False, -1, float("inf")

    flat_indices = torch.nonzero(active, as_tuple=False).flatten()
    spike_times = flat[flat_indices]
    order = torch.argsort(spike_times, stable=True)
    sorted_indices = flat_indices[order]
    sorted_times = spike_times[order]
    sorted_weights = _weight_flat_hwc(weights).index_select(1, sorted_indices)
    reached = sorted_weights.cumsum(dim=1) >= thresholds.unsqueeze(1)
    any_reached = reached.any(dim=1)
    if not bool(any_reached.any().item()):
        return False, -1, float("inf")

    first_indices = reached.to(torch.int64).argmax(dim=1)
    sentinel = torch.full_like(first_indices, int(sorted_times.numel()))
    first_indices = torch.where(any_reached, first_indices, sentinel)
    winner = int(first_indices.argmin().item())
    winner_position = int(first_indices[winner].item())
    return True, winner, float(sorted_times[winner_position].item())


class LocalLatencyConvLayer:
    """Event-driven convolution with local first-spike STDP."""

    def __init__(self, config: LocalLatencyConvConfig) -> None:
        self.config = config
        if int(config.inference_top_k) < 0:
            raise ValueError("inference_top_k must be non-negative")

        shape = (
            int(config.out_channels),
            int(config.in_channels),
            int(config.kernel_size),
            int(config.kernel_size),
        )
        generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
        mode = str(config.weight_init_mode).strip().lower()
        if mode in {"normal", "reference"}:
            weights = torch.empty(shape, dtype=torch.float32)
            weights.normal_(
                mean=float(config.weight_init_mean),
                std=float(config.weight_init_std),
                generator=generator,
            )
        elif mode in {"identity", "diagonal"}:
            if int(config.kernel_size) != 1:
                raise ValueError("identity initialization requires kernel_size=1")
            if int(config.in_channels) != int(config.out_channels):
                raise ValueError("identity initialization requires equal input and output channels")
            weights = torch.zeros(shape, dtype=torch.float32)
            diagonal = torch.arange(int(config.in_channels))
            weights[diagonal, diagonal, 0, 0] = float(config.identity_gain)
            if float(config.identity_noise_std) > 0.0:
                noise = torch.randn(shape, generator=generator, dtype=torch.float32)
                weights.add_(noise * float(config.identity_noise_std))
        else:
            raise ValueError(f"Unsupported weight_init_mode={config.weight_init_mode!r}")

        self.weight = weights.clamp(float(config.weight_min), float(config.weight_max))
        self.threshold = torch.full(
            (int(config.out_channels),), float(config.threshold), dtype=torch.float32
        )
        self._random = random.Random(int(config.seed))
        self._potentiation = float(config.learning_rate_potentiation)
        self._depression = abs(float(config.learning_rate_depression))
        self._threshold_learning_rate = float(config.threshold_learning_rate)

    @property
    def weights(self) -> torch.Tensor:
        return self.weight

    @property
    def thresholds(self) -> torch.Tensor:
        return self.threshold

    def output_shape(self, height: int, width: int) -> tuple[int, int, int]:
        return (
            int(self.config.out_channels),
            _output_size(height, self.config.kernel_size, self.config.stride, self.config.padding),
            _output_size(width, self.config.kernel_size, self.config.stride, self.config.padding),
        )

    def _patches_flat_hwc_single(
        self, latency_map: torch.Tensor, output_height: int, output_width: int
    ) -> torch.Tensor:
        stride = int(self.config.stride)
        padding = int(self.config.padding)
        if padding > 0:
            latency_map = F.pad(
                latency_map,
                (padding, padding, padding, padding),
                mode="constant",
                value=float("inf"),
            )

        chunks = []
        for row in range(int(self.config.kernel_size)):
            row_slice = slice(row, row + output_height * stride, stride)
            for col in range(int(self.config.kernel_size)):
                col_slice = slice(col, col + output_width * stride, stride)
                chunks.append(latency_map[:, row_slice, col_slice].permute(1, 2, 0).contiguous())
        return torch.cat(chunks, dim=-1).view(output_height * output_width, -1)

    @torch.no_grad()
    def _infer_tensor(self, latencies: torch.Tensor) -> torch.Tensor:
        samples, channels, height, width = latencies.shape
        if int(channels) != int(self.config.in_channels):
            raise ValueError(
                f"Expected {self.config.in_channels} input channels, got {int(channels)}"
            )

        _, output_height, output_width = self.output_shape(int(height), int(width))
        output = torch.full(
            (int(samples), int(self.config.out_channels), output_height, output_width),
            float("inf"),
            dtype=torch.float32,
            device=latencies.device,
        )
        weight = self.weight.to(latencies.device)
        threshold = self.threshold.to(latencies.device)
        weight_flat = _weight_flat_hwc(weight)
        positions = output_height * output_width

        for sample in range(int(samples)):
            patches = self._patches_flat_hwc_single(latencies[sample], output_height, output_width)
            order = torch.argsort(patches, dim=-1, stable=True)
            sorted_latencies = torch.gather(patches, dim=-1, index=order)
            sorted_weights = torch.take_along_dim(
                weight_flat.unsqueeze(1).expand(-1, positions, -1),
                order.unsqueeze(0).expand(int(self.config.out_channels), -1, -1),
                dim=-1,
            )
            reached = sorted_weights.cumsum(dim=-1) >= threshold.view(-1, 1, 1)
            any_reached = reached.any(dim=-1)
            first_indices = reached.to(torch.int64).argmax(dim=-1)
            first_times = torch.gather(
                sorted_latencies.unsqueeze(0).expand(int(self.config.out_channels), -1, -1),
                dim=-1,
                index=first_indices.unsqueeze(-1),
            ).squeeze(-1)
            first_times = torch.where(
                any_reached,
                first_times,
                torch.full_like(first_times, float("inf")),
            )
            output[sample] = first_times.view(
                int(self.config.out_channels), output_height, output_width
            )
        return self._apply_pointwise_competition(output)

    def _apply_pointwise_competition(self, output: torch.Tensor) -> torch.Tensor:
        top_k = int(self.config.inference_top_k)
        if bool(self.config.inference_winner_take_all) and top_k <= 0:
            top_k = 1
        if top_k <= 0 or top_k >= int(self.config.out_channels):
            return output

        times = output.permute(0, 2, 3, 1).contiguous()
        order = torch.argsort(times, dim=-1, stable=True)
        selected = order[..., :top_k]
        selected_times = torch.gather(times, dim=-1, index=selected)
        keep = torch.zeros_like(times, dtype=torch.bool)
        keep.scatter_(-1, selected, torch.isfinite(selected_times))
        filtered = torch.where(keep, times, torch.full_like(times, float("inf")))
        return filtered.permute(0, 3, 1, 2).contiguous()

    def infer_latency_map(self, latencies: np.ndarray | torch.Tensor) -> np.ndarray:
        return self._infer_tensor(_as_latency_tensor(latencies)).cpu().numpy()

    def _sample_patch_origin(
        self,
        *,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
    ) -> tuple[int, int]:
        kernel = int(self.config.kernel_size)
        space = str(self.config.patch_sampling_space).strip().lower()
        if space == "reference_output":
            max_height = max(0, int(output_height) - kernel)
            max_width = max(0, int(output_width) - kernel)
        elif space == "input_valid":
            max_height = max(0, int(input_height) - kernel)
            max_width = max(0, int(input_width) - kernel)
        else:
            raise ValueError("patch_sampling_space must be 'reference_output' or 'input_valid'")
        row = self._random.randint(0, max_height) if max_height > 0 else 0
        col = self._random.randint(0, max_width) if max_width > 0 else 0
        return row, col

    @torch.no_grad()
    def fit_one_epoch(
        self,
        latencies: np.ndarray | torch.Tensor,
        *,
        anneal: bool = True,
    ) -> dict[str, float]:
        maps = _as_latency_tensor(latencies)
        samples, channels, height, width = maps.shape
        if int(channels) != int(self.config.in_channels):
            raise ValueError(
                f"Expected {self.config.in_channels} input channels, got {int(channels)}"
            )

        _, output_height, output_width = self.output_shape(int(height), int(width))
        updates = 0
        nonempty = 0
        no_winner = 0
        spike_counts: list[int] = []
        winner_times: list[float] = []
        mean_weight_changes: list[float] = []
        for sample in range(int(samples)):
            row, col = self._sample_patch_origin(
                input_height=int(height),
                input_width=int(width),
                output_height=output_height,
                output_width=output_width,
            )
            kernel = int(self.config.kernel_size)
            patch = maps[sample, :, row : row + kernel, col : col + kernel].contiguous()
            spike_count = int(torch.isfinite(patch).sum().item())
            spike_counts.append(spike_count)
            if spike_count <= 0:
                mean_weight_changes.append(0.0)
                continue

            nonempty += 1
            updated, winner, post_time = _winner_from_patch(patch, self.weight, self.threshold)
            if not updated:
                no_winner += 1
                mean_weight_changes.append(0.0)
                continue

            updates += 1
            winner_times.append(post_time)
            self._update_thresholds(winner, post_time)
            mean_weight_changes.append(self._update_weights(winner, patch, post_time))

        if anneal:
            self._anneal_learning_rates()
        return {
            "sample_count": float(samples),
            "nonempty_patch_ratio": float(nonempty / max(int(samples), 1)),
            "winner_update_ratio": float(updates / max(int(samples), 1)),
            "no_winner_ratio": float(no_winner / max(int(samples), 1)),
            "patch_spike_count_mean": float(np.mean(spike_counts)) if spike_counts else 0.0,
            "winner_time_mean": float(np.mean(winner_times)) if winner_times else -1.0,
            "winner_time_std": float(np.std(winner_times)) if winner_times else 0.0,
            "mean_dW": float(np.mean(mean_weight_changes)) if mean_weight_changes else 0.0,
            "threshold_mean": float(self.threshold.mean().item()),
            "threshold_min": float(self.threshold.min().item()),
            "threshold_max": float(self.threshold.max().item()),
            "learning_rate_potentiation": float(self._potentiation),
            "learning_rate_depression": -float(self._depression),
            "threshold_learning_rate": float(self._threshold_learning_rate),
        }

    def fit(self, latencies: np.ndarray | torch.Tensor, *, epochs: int) -> list[dict[str, float]]:
        return [self.fit_one_epoch(latencies) for _ in range(int(epochs))]

    def fit_stream(
        self,
        batch_factory: Callable[[], Iterable[np.ndarray | torch.Tensor]],
        *,
        epochs: int,
        progress: Callable[[str], None] | None = None,
        progress_every_batches: int = 100,
    ) -> list[dict[str, float]]:
        """Train sample-wise from repeatable batches without dense materialization."""

        history = []
        for epoch in range(int(epochs)):
            batch_stats = []
            for batch_index, batch in enumerate(batch_factory(), start=1):
                batch_stats.append(self.fit_one_epoch(batch, anneal=False))
                if (
                    progress is not None
                    and int(progress_every_batches) > 0
                    and batch_index % int(progress_every_batches) == 0
                ):
                    samples = int(sum(item["sample_count"] for item in batch_stats))
                    progress(
                        f"epoch={epoch + 1}/{int(epochs)} batches={batch_index} samples={samples}"
                    )
            self._anneal_learning_rates()
            epoch_stats = self._merge_batch_stats(batch_stats)
            history.append(epoch_stats)
            if progress is not None:
                progress(
                    f"epoch={epoch + 1}/{int(epochs)} completed "
                    f"winner_update_ratio={epoch_stats['winner_update_ratio']:.4f} "
                    f"threshold_mean={epoch_stats['threshold_mean']:.4f}"
                )
        return history

    def _anneal_learning_rates(self) -> None:
        factor = float(self.config.annealing)
        self._potentiation *= factor
        self._depression *= factor
        self._threshold_learning_rate *= factor

    def _merge_batch_stats(
        self,
        batch_stats: list[dict[str, float]],
    ) -> dict[str, float]:
        sample_count = sum(item["sample_count"] for item in batch_stats)
        weighted_fields = (
            "nonempty_patch_ratio",
            "winner_update_ratio",
            "no_winner_ratio",
            "patch_spike_count_mean",
            "winner_time_mean",
            "winner_time_std",
            "mean_dW",
        )
        merged = {"sample_count": float(sample_count)}
        for field in weighted_fields:
            merged[field] = sum(item[field] * item["sample_count"] for item in batch_stats) / max(
                sample_count, 1.0
            )
        merged.update(
            {
                "threshold_mean": float(self.threshold.mean().item()),
                "threshold_min": float(self.threshold.min().item()),
                "threshold_max": float(self.threshold.max().item()),
                "learning_rate_potentiation": float(self._potentiation),
                "learning_rate_depression": -float(self._depression),
                "threshold_learning_rate": float(self._threshold_learning_rate),
            }
        )
        return merged

    @torch.no_grad()
    def _update_thresholds(self, winner: int, post_time: float) -> None:
        learning_rate = float(self._threshold_learning_rate)
        self.threshold.add_(
            -learning_rate * (float(post_time) - float(self.config.threshold_target_time))
        )
        if int(self.config.out_channels) > 1:
            self.threshold.add_(-learning_rate / float(int(self.config.out_channels) - 1))
        self.threshold[int(winner)] += learning_rate
        self.threshold.clamp_(min=float(self.config.min_threshold))

    @torch.no_grad()
    def _update_weights(self, winner: int, patch: torch.Tensor, post_time: float) -> float:
        winner_weights = self.weight[int(winner)]
        pre_before_post = patch <= float(post_time)
        potentiation = self._potentiation * torch.exp(-float(self.config.beta) * winner_weights)
        depression = self._depression * torch.exp(float(self.config.beta) * (winner_weights - 1.0))
        updated = torch.where(
            pre_before_post,
            winner_weights + potentiation,
            winner_weights - depression,
        ).clamp(float(self.config.weight_min), float(self.config.weight_max))
        mean_change = float((updated - winner_weights).abs().mean().item())
        winner_weights.copy_(updated)
        return mean_change

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "weight": self.weight.clone(),
            "threshold": self.threshold.clone(),
            "learning_rate_potentiation": float(self._potentiation),
            "learning_rate_depression": float(self._depression),
            "threshold_learning_rate": float(self._threshold_learning_rate),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | float]) -> None:
        self.weight.copy_(torch.as_tensor(state["weight"], dtype=torch.float32))
        self.threshold.copy_(torch.as_tensor(state["threshold"], dtype=torch.float32))
        self._potentiation = float(state.get("learning_rate_potentiation", self._potentiation))
        self._depression = float(state.get("learning_rate_depression", self._depression))
        self._threshold_learning_rate = float(
            state.get("threshold_learning_rate", self._threshold_learning_rate)
        )


def min_pool_latency(
    latencies: np.ndarray | torch.Tensor, *, kernel_size: int, stride: int
) -> np.ndarray:
    maps = _as_latency_tensor(latencies)
    samples, channels, height, width = maps.shape
    output_height = _output_size(int(height), int(kernel_size), int(stride), 0)
    output_width = _output_size(int(width), int(kernel_size), int(stride), 0)
    pooled = torch.full(
        (int(samples), int(channels), output_height, output_width),
        float("inf"),
        dtype=torch.float32,
    )
    for row in range(output_height):
        row_start = row * int(stride)
        for col in range(output_width):
            col_start = col * int(stride)
            window = maps[
                :,
                :,
                row_start : row_start + int(kernel_size),
                col_start : col_start + int(kernel_size),
            ]
            pooled[:, :, row, col] = window.amin(dim=(-2, -1))
    return pooled.numpy()


def flatten_latency_map(latencies: np.ndarray | torch.Tensor) -> np.ndarray:
    maps = _as_latency_tensor(latencies)
    return maps.reshape((maps.shape[0], -1)).numpy().astype(np.float32, copy=False)
