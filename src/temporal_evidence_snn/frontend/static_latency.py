from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


def _resolve_patch_size(value: Any) -> tuple[int, int]:
    if value is None:
        return 0, 0
    if isinstance(value, (int, float)):
        size = int(value)
        return size, size
    values = tuple(int(item) for item in value)
    if len(values) != 2:
        raise ValueError(f"Patch size must contain two values, got {values}")
    return values


def _as_raw_uint8(images: np.ndarray, *, color_mode: str) -> torch.Tensor:
    array = np.asarray(images)
    if array.ndim == 2:
        array = array[None]
    if array.ndim == 3:
        if color_mode != "grayscale":
            raise ValueError(f"RGB input requires shape (N,H,W,3), got {array.shape}")
    elif array.ndim == 4:
        if int(array.shape[-1]) not in {1, 3}:
            if int(array.shape[1]) in {1, 3}:
                array = np.moveaxis(array, 1, -1)
            else:
                raise ValueError(f"Cannot identify the channel axis in {array.shape}")
    else:
        raise ValueError(f"Expected an image batch, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        maximum = float(finite.max()) if finite.size else 0.0
        scale = 255.0 if maximum <= 1.0 else 1.0
        array = np.rint(np.clip(array * scale, 0.0, 255.0))
    return torch.as_tensor(array, dtype=torch.uint8)


def _uint8_to_perceptual_channels(images: torch.Tensor, color_mode: str) -> torch.Tensor:
    if images.dim() == 3:
        if color_mode != "grayscale":
            raise ValueError(f"Expected RGB input for color_mode={color_mode!r}")
        return images.to(dtype=torch.float32).unsqueeze(1) / 255.0
    if images.dim() != 4:
        raise ValueError(f"Expected (N,H,W) or (N,H,W,C), got {tuple(images.shape)}")
    if int(images.shape[-1]) == 1:
        if color_mode != "grayscale":
            raise ValueError(f"Expected RGB input for color_mode={color_mode!r}")
        return images.to(dtype=torch.float32).permute(0, 3, 1, 2).contiguous() / 255.0
    x = images.to(dtype=torch.float32).permute(0, 3, 1, 2).contiguous() / 255.0
    if color_mode == "rgb":
        return x
    if color_mode == "grayscale":
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    raise ValueError(f"Unsupported color_mode={color_mode!r}")


def _signed_context_competition(
    x: torch.Tensor,
    *,
    patch_size: tuple[int, int],
    alpha: float,
    loser_attenuation: float,
    mixed_ratio_start: float,
    mixed_ratio_end: float,
    energy_gate_start: float,
    energy_gate_end: float,
    channel_gate_start: float,
    channel_gate_end: float,
) -> torch.Tensor:
    alpha = min(max(float(alpha), 0.0), 1.0)
    if alpha <= 0.0:
        return x
    patch_h, patch_w = patch_size
    if patch_h <= 0 or patch_w <= 0 or patch_h % 2 == 0 or patch_w % 2 == 0:
        raise ValueError(f"Signed-context patch must be positive and odd, got {patch_size}")
    positive = torch.clamp_min(x, 0.0)
    negative = torch.clamp_min(-x, 0.0)
    positive_support = F.avg_pool2d(
        positive, (patch_h, patch_w), stride=1, padding=(patch_h // 2, patch_w // 2)
    )
    negative_support = F.avg_pool2d(
        negative, (patch_h, patch_w), stride=1, padding=(patch_h // 2, patch_w // 2)
    )
    maximum = torch.maximum(positive_support, negative_support).clamp_min(1e-12)
    mixed_ratio = torch.minimum(positive_support, negative_support) / maximum
    ratio_range = max(float(mixed_ratio_end) - float(mixed_ratio_start), 1e-6)
    gate = ((mixed_ratio - float(mixed_ratio_start)) / ratio_range).clamp(0.0, 1.0)
    if float(energy_gate_end) > float(energy_gate_start):
        energy = positive_support + negative_support
        relative = energy / energy.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        gate *= ((relative - energy_gate_start) / (energy_gate_end - energy_gate_start)).clamp(
            0.0, 1.0
        )
    if float(channel_gate_end) > float(channel_gate_start):
        mass = x.abs().sum(dim=(-2, -1), keepdim=True)
        relative = mass / mass.amax(dim=1, keepdim=True).clamp_min(1e-12)
        gate *= ((relative - channel_gate_start) / (channel_gate_end - channel_gate_start)).clamp(
            0.0, 1.0
        )
    gate *= alpha
    aligned = (positive_support >= negative_support) == (x >= 0.0)
    candidate = torch.where(aligned, x, x * float(loser_attenuation))
    original_mass = x.abs().sum(dim=(-1, -2), keepdim=True)
    candidate_mass = candidate.abs().sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    candidate = torch.where(
        original_mass > 0.0,
        candidate * (original_mass / candidate_mass),
        torch.zeros_like(candidate),
    )
    return torch.lerp(x, candidate, gate)


def _patch_view(pair: torch.Tensor, patch_size: tuple[int, int]):
    patch_h, patch_w = patch_size
    _, _, height, width = pair.shape
    pad_h = (patch_h - height % patch_h) % patch_h
    pad_w = (patch_w - width % patch_w) % patch_w
    padded = F.pad(pair, (0, pad_w, 0, pad_h))
    batch, _, padded_h, padded_w = padded.shape
    grid_h, grid_w = padded_h // patch_h, padded_w // patch_w
    patches = padded.view(batch, 2, grid_h, patch_h, grid_w, patch_w).permute(0, 2, 4, 1, 3, 5)
    return patches, height, width, padded_h, padded_w


def _restore_patches(
    patches: torch.Tensor,
    *,
    height: int,
    width: int,
    padded_h: int,
    padded_w: int,
) -> torch.Tensor:
    batch = int(patches.shape[0])
    restored = patches.permute(0, 3, 1, 4, 2, 5).contiguous().view(batch, 2, padded_h, padded_w)
    return restored[:, :, :height, :width]


def _polarity_patch_reweight(
    x: torch.Tensor,
    pair_slices: Iterable[slice],
    *,
    patch_size: tuple[int, int],
    positive_dominant_compress: float,
    positive_dominant_negative_boost: float,
    negative_dominant_compress: float,
    negative_dominant_positive_boost: float,
    dominance_margin: float,
) -> torch.Tensor:
    if (
        max(
            positive_dominant_compress,
            positive_dominant_negative_boost,
            negative_dominant_compress,
            negative_dominant_positive_boost,
        )
        <= 0.0
        or min(patch_size) <= 0
    ):
        return x
    margin = min(max(float(dominance_margin), 0.0), 0.99)
    out = x.clone()
    for pair_slice in pair_slices:
        patches, height, width, padded_h, padded_w = _patch_view(out[:, pair_slice], patch_size)
        energy = patches.sum(dim=(-1, -2))
        relative = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        positive_gate = ((relative[..., 0] - relative[..., 1] - margin) / (1.0 - margin)).clamp(
            0.0, 1.0
        )
        negative_gate = ((relative[..., 1] - relative[..., 0] - margin) / (1.0 - margin)).clamp(
            0.0, 1.0
        )
        scale = torch.ones_like(patches)
        scale[..., 0, :, :] *= (1.0 - positive_dominant_compress * positive_gate)[..., None, None]
        scale[..., 1, :, :] *= (1.0 + positive_dominant_negative_boost * positive_gate)[
            ..., None, None
        ]
        scale[..., 1, :, :] *= (1.0 - negative_dominant_compress * negative_gate)[..., None, None]
        scale[..., 0, :, :] *= (1.0 + negative_dominant_positive_boost * negative_gate)[
            ..., None, None
        ]
        candidate = patches * scale.clamp_min(0.0)
        original_total = patches.sum(dim=(-1, -2, -3), keepdim=True)
        candidate_total = candidate.sum(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-12)
        candidate *= original_total / candidate_total
        out[:, pair_slice] = _restore_patches(
            candidate, height=height, width=width, padded_h=padded_h, padded_w=padded_w
        )
    return out


def _pair_compensation(
    x: torch.Tensor,
    pair_slices: Iterable[slice],
    *,
    patch_size: tuple[int, int],
    alpha: float,
    loser_attenuation: float,
    negative_bias: float,
    dominance_margin: float,
    energy_gate_start: float,
    energy_gate_end: float,
    peak_gate_start: float,
    peak_gate_end: float,
) -> torch.Tensor:
    alpha = min(max(float(alpha), 0.0), 1.0)
    if alpha <= 0.0 or min(patch_size) <= 0:
        return x
    margin = min(max(float(dominance_margin), 0.0), 0.99)
    patch_h, patch_w = patch_size
    out = x.clone()
    for pair_slice in pair_slices:
        patches, height, width, padded_h, padded_w = _patch_view(out[:, pair_slice], patch_size)
        energy = patches.sum(dim=(-1, -2))
        total = energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        relative = energy / total
        positive_gate = ((relative[..., 0] - relative[..., 1] - margin) / (1.0 - margin)).clamp(
            0.0, 1.0
        )
        if float(energy_gate_end) > float(energy_gate_start):
            patch_energy = total[..., 0]
            energy_relative = patch_energy / patch_energy.amax(
                dim=(-1, -2), keepdim=True
            ).clamp_min(1e-12)
            positive_gate *= (
                (energy_relative - energy_gate_start) / (energy_gate_end - energy_gate_start)
            ).clamp(0.0, 1.0)
        if float(peak_gate_end) > float(peak_gate_start):
            positive_patch = patches[..., 0, :, :]
            peak_relative = (
                positive_patch.amax(dim=(-1, -2))
                / positive_patch.mean(dim=(-1, -2)).clamp_min(1e-12)
            ) / float(patch_h * patch_w)
            positive_gate *= (
                (peak_relative - peak_gate_start) / (peak_gate_end - peak_gate_start)
            ).clamp(0.0, 1.0)
        score = patches.clone()
        if float(negative_bias) > 0.0:
            score[..., 1, :, :] *= (1.0 + negative_bias * positive_gate)[..., None, None]
        winners = score.argmax(dim=3, keepdim=True)
        scale = torch.full_like(patches, min(max(float(loser_attenuation), 0.0), 1.0))
        scale.scatter_(3, winners, 1.0)
        candidate = patches * scale
        original_total = patches.sum(dim=(-1, -2, -3), keepdim=True)
        candidate_total = candidate.sum(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-12)
        candidate *= original_total / candidate_total
        gate = (positive_gate * alpha)[..., None, None, None]
        patches = patches * (1.0 - gate) + candidate * gate
        out[:, pair_slice] = _restore_patches(
            patches, height=height, width=width, padded_h=padded_h, padded_w=padded_w
        )
    return out


@dataclass(frozen=True)
class StaticLatencyFrontendConfig:
    input_channels: int = 1
    color_mode: str = "grayscale"
    split_polarities: bool = True
    whitening_patch_size: int = 7
    whitening_stride: int = 2
    whitening_eps: float = 1.0e-2
    whitening_pca_compress: float = 1.0
    whitening_max_samples: int = 1_000_000
    fit_batch_images: int = 64
    calibration_mode: str = "dataset_feature"
    calibration_oracle_compat: bool = True
    signed_context_patch_size: tuple[int, int] = (0, 0)
    signed_context_alpha: float = 0.0
    signed_context_loser_attenuation: float = 0.25
    signed_context_mixed_ratio_start: float = 0.2
    signed_context_mixed_ratio_end: float = 0.7
    signed_context_energy_gate_start: float = 0.0
    signed_context_energy_gate_end: float = 0.0
    signed_context_channel_gate_start: float = 0.0
    signed_context_channel_gate_end: float = 0.0
    polarity_reweight_patch_size: tuple[int, int] = (0, 0)
    positive_dominant_compress: float = 0.0
    positive_dominant_negative_boost: float = 0.0
    negative_dominant_compress: float = 0.0
    negative_dominant_positive_boost: float = 0.0
    polarity_dominance_margin: float = 0.1
    pair_compensation_patch_size: tuple[int, int] = (0, 0)
    pair_compensation_alpha: float = 0.0
    pair_compensation_loser_attenuation: float = 0.25
    pair_compensation_negative_bias: float = 0.0
    pair_compensation_dominance_margin: float = 0.1
    pair_compensation_energy_gate_start: float = 0.0
    pair_compensation_energy_gate_end: float = 0.0
    pair_compensation_peak_gate_start: float = 0.0
    pair_compensation_peak_gate_end: float = 0.0
    response_floor: float = 0.0
    response_gamma: float = 1.0
    time_steps: int = 40


def response_to_latency(
    responses: np.ndarray,
    *,
    response_floor: float = 0.0,
    response_gamma: float = 1.0,
) -> np.ndarray:
    response = torch.as_tensor(responses, dtype=torch.float32)
    if float(response_floor) > 0.0:
        response = torch.where(response < float(response_floor), 0.0, response)
    if float(response_gamma) != 1.0:
        response = response.pow(float(response_gamma))
    timestamp = torch.clamp_min(1.0 - response, 0.0)
    latency = torch.full_like(response, float("inf"))
    active = timestamp != 1.0
    latency[active] = timestamp[active]
    return latency.numpy()


def latency_to_spike_bins(latencies: np.ndarray, *, time_steps: int) -> np.ndarray:
    latency = torch.as_tensor(latencies, dtype=torch.float32)
    steps = int(time_steps)
    if latency.dim() != 4 or steps <= 0:
        raise ValueError(
            f"Expected (N,C,H,W) latencies and positive time_steps, got {latency.shape}"
        )
    output = torch.zeros((steps, *latency.shape), dtype=latency.dtype)
    active = torch.isfinite(latency)
    if bool(active.any()):
        bins = torch.round(latency.clamp(0.0, 1.0) * float(steps - 1)).to(torch.long)
        indices = active.nonzero(as_tuple=True)
        output[(bins[indices], *indices)] = 1.0
    return output.numpy()


class StaticLatencyFrontend:
    """Static-image latency front end with local whitening and polarity processing."""

    def __init__(self, config: StaticLatencyFrontendConfig) -> None:
        self.config = config
        self.whitening_weight_: torch.Tensor | None = None
        self.scaling_min_: torch.Tensor | None = None
        self.scaling_max_: torch.Tensor | None = None

    @property
    def output_channels(self) -> int:
        return int(self.config.input_channels) * (2 if self.config.split_polarities else 1)

    def _fit_whitening(self, images: torch.Tensor) -> None:
        config = self.config
        patch_size = int(config.whitening_patch_size)
        patch_dim = patch_size * patch_size * int(config.input_channels)
        sum_vector = torch.zeros(patch_dim, dtype=torch.float64)
        sum_outer = torch.zeros((patch_dim, patch_dim), dtype=torch.float64)
        total = 0
        for start in range(0, int(images.shape[0]), int(config.fit_batch_images)):
            if total >= int(config.whitening_max_samples):
                break
            x = _uint8_to_perceptual_channels(
                images[start : start + int(config.fit_batch_images)], config.color_mode
            )
            patches = F.unfold(x, kernel_size=patch_size, stride=int(config.whitening_stride))
            patches = patches.transpose(1, 2).reshape(-1, patch_dim)
            patches = patches[: int(config.whitening_max_samples) - total]
            if not patches.numel():
                continue
            patches64 = patches.to(torch.float64)
            sum_vector += patches64.sum(dim=0)
            sum_outer += patches64.T @ patches64
            total += int(patches64.shape[0])
        if total == 0:
            raise RuntimeError("No patches were available for whitening.")
        mean = sum_vector / float(total)
        covariance = sum_outer / float(total) - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        inverse_sqrt = (eigenvalues.clamp_min(0.0) + float(config.whitening_eps)).rsqrt()
        keep = int(float(patch_dim) * float(config.whitening_pca_compress))
        if keep < patch_dim:
            inverse_sqrt[: patch_dim - keep] = 0.0
        whitening = eigenvectors @ torch.diag(inverse_sqrt) @ eigenvectors.T
        filters = []
        center = patch_size // 2
        for channel in range(int(config.input_channels)):
            impulse = torch.zeros(
                (config.input_channels, patch_size, patch_size), dtype=torch.float64
            )
            impulse[channel, center, center] = 1.0
            response = (impulse.flatten() @ whitening).view(
                config.input_channels, patch_size, patch_size
            )
            filters.append((response - response.mean()).to(torch.float32))
        self.whitening_weight_ = torch.stack(filters).contiguous()

    def _whiten(self, x: torch.Tensor) -> torch.Tensor:
        if self.whitening_weight_ is None:
            raise RuntimeError("The front end must be fitted before transformation.")
        pad = int(self.config.whitening_patch_size) // 2
        return F.conv2d(
            F.pad(x, (pad, pad, pad, pad), mode="replicate"), self.whitening_weight_.to(x.dtype)
        )

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.split_polarities:
            return torch.clamp_min(x, 0.0)
        return torch.stack((torch.clamp_min(x, 0.0), torch.clamp_min(-x, 0.0)), dim=2).flatten(1, 2)

    def fit(self, images: np.ndarray) -> "StaticLatencyFrontend":
        raw = _as_raw_uint8(images, color_mode=self.config.color_mode)
        self._fit_whitening(raw)
        if self.config.calibration_mode != "dataset_feature":
            raise ValueError("Static latency encoding requires dataset_feature calibration.")
        minimum = maximum = None
        for start in range(0, int(raw.shape[0]), int(self.config.fit_batch_images)):
            x = _uint8_to_perceptual_channels(
                raw[start : start + int(self.config.fit_batch_images)], self.config.color_mode
            )
            split = self._split(self._whiten(x))
            batch_minimum = split.amin(dim=0, keepdim=True)
            batch_maximum = split.amax(dim=0, keepdim=True)
            if minimum is None:
                if self.config.calibration_oracle_compat:
                    info = torch.finfo(batch_minimum.dtype)
                    minimum = torch.full_like(batch_minimum, info.max)
                    maximum = torch.full_like(batch_maximum, info.tiny)
                else:
                    minimum, maximum = batch_minimum, batch_maximum
            minimum = torch.minimum(minimum, batch_minimum)
            maximum = torch.maximum(maximum, batch_maximum)
        self.scaling_min_ = minimum.contiguous()
        self.scaling_max_ = maximum.contiguous()
        return self

    def _calibrate(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaling_min_ is None or self.scaling_max_ is None:
            raise RuntimeError("The front end must be fitted before transformation.")
        delta = self.scaling_max_ - self.scaling_min_
        mask = delta == 0 if self.config.calibration_oracle_compat else delta.abs() <= 1e-12
        return torch.where(mask, torch.zeros_like(x), (x - self.scaling_min_) / delta)

    def _pair_slices(self) -> list[slice]:
        width = 2 if self.config.split_polarities else 1
        return [
            slice(index * width, (index + 1) * width) for index in range(self.config.input_channels)
        ]

    def _response_tensor(self, images: np.ndarray) -> torch.Tensor:
        raw = _as_raw_uint8(images, color_mode=self.config.color_mode)
        x = self._whiten(_uint8_to_perceptual_channels(raw, self.config.color_mode))
        x = _signed_context_competition(
            x,
            patch_size=self.config.signed_context_patch_size,
            alpha=self.config.signed_context_alpha,
            loser_attenuation=self.config.signed_context_loser_attenuation,
            mixed_ratio_start=self.config.signed_context_mixed_ratio_start,
            mixed_ratio_end=self.config.signed_context_mixed_ratio_end,
            energy_gate_start=self.config.signed_context_energy_gate_start,
            energy_gate_end=self.config.signed_context_energy_gate_end,
            channel_gate_start=self.config.signed_context_channel_gate_start,
            channel_gate_end=self.config.signed_context_channel_gate_end,
        )
        x = self._calibrate(self._split(x))
        pairs = self._pair_slices()
        x = _polarity_patch_reweight(
            x,
            pairs,
            patch_size=self.config.polarity_reweight_patch_size,
            positive_dominant_compress=self.config.positive_dominant_compress,
            positive_dominant_negative_boost=self.config.positive_dominant_negative_boost,
            negative_dominant_compress=self.config.negative_dominant_compress,
            negative_dominant_positive_boost=self.config.negative_dominant_positive_boost,
            dominance_margin=self.config.polarity_dominance_margin,
        )
        x = _pair_compensation(
            x,
            pairs,
            patch_size=self.config.pair_compensation_patch_size,
            alpha=self.config.pair_compensation_alpha,
            loser_attenuation=self.config.pair_compensation_loser_attenuation,
            negative_bias=self.config.pair_compensation_negative_bias,
            dominance_margin=self.config.pair_compensation_dominance_margin,
            energy_gate_start=self.config.pair_compensation_energy_gate_start,
            energy_gate_end=self.config.pair_compensation_energy_gate_end,
            peak_gate_start=self.config.pair_compensation_peak_gate_start,
            peak_gate_end=self.config.pair_compensation_peak_gate_end,
        )
        if self.config.response_floor > 0.0:
            x = torch.where(x < self.config.response_floor, 0.0, x)
        if self.config.response_gamma != 1.0:
            x = x.pow(self.config.response_gamma)
        return x

    def transform_response(self, images: np.ndarray) -> np.ndarray:
        return self._response_tensor(images).detach().cpu().numpy()

    def transform_latency(self, images: np.ndarray) -> np.ndarray:
        response = self._response_tensor(images)
        timestamp = torch.clamp_min(1.0 - response, 0.0)
        latency = torch.full_like(response, float("inf"))
        active = timestamp != 1.0
        latency[active] = timestamp[active]
        return latency.detach().cpu().numpy()

    def transform_spikes(self, images: np.ndarray) -> np.ndarray:
        return latency_to_spike_bins(
            self.transform_latency(images), time_steps=self.config.time_steps
        )


def static_latency_frontend_from_config(frontend_config: dict) -> StaticLatencyFrontend:
    input_type = str(frontend_config.get("input_type", "grayscale_image"))
    if input_type == "event_stream":
        raise ValueError("Event streams require EventSurfaceLatencyFrontend.")
    color_mode = "rgb" if input_type == "rgb_image" else "grayscale"
    whitening = frontend_config.get("local_whitening", {})
    calibration = frontend_config.get("calibration", {})
    signed = frontend_config.get("signed_context", {})
    reweight = frontend_config.get("polarity_reweighting", {})
    pair = frontend_config.get("pair_compensation", {})
    response = frontend_config.get("response", {})
    return StaticLatencyFrontend(
        StaticLatencyFrontendConfig(
            input_channels=3 if color_mode == "rgb" else 1,
            color_mode=color_mode,
            split_polarities=bool(frontend_config.get("polarity_split", {}).get("enabled", True)),
            whitening_patch_size=int(whitening.get("patch_size", 9 if color_mode == "rgb" else 7)),
            whitening_stride=int(whitening.get("stride", 2)),
            whitening_eps=float(whitening.get("eps", 1e-2)),
            whitening_pca_compress=float(whitening.get("pca_compress", 1.0)),
            whitening_max_samples=int(whitening.get("max_samples", 1_000_000)),
            fit_batch_images=int(whitening.get("fit_batch_images", 64)),
            calibration_mode=str(calibration.get("mode", "dataset_feature")),
            calibration_oracle_compat=bool(calibration.get("oracle_compat", True)),
            signed_context_patch_size=_resolve_patch_size(signed.get("patch_size", 0)),
            signed_context_alpha=float(signed.get("alpha", 0.0)),
            signed_context_loser_attenuation=float(signed.get("loser_attenuation", 0.25)),
            signed_context_mixed_ratio_start=float(signed.get("mixed_ratio_start", 0.2)),
            signed_context_mixed_ratio_end=float(signed.get("mixed_ratio_end", 0.7)),
            signed_context_energy_gate_start=float(signed.get("energy_gate_start", 0.0)),
            signed_context_energy_gate_end=float(signed.get("energy_gate_end", 0.0)),
            signed_context_channel_gate_start=float(signed.get("channel_gate_start", 0.0)),
            signed_context_channel_gate_end=float(signed.get("channel_gate_end", 0.0)),
            polarity_reweight_patch_size=_resolve_patch_size(reweight.get("patch_size", 0)),
            positive_dominant_compress=float(reweight.get("positive_dominant_compress", 0.0)),
            positive_dominant_negative_boost=float(
                reweight.get("positive_dominant_negative_boost", 0.0)
            ),
            negative_dominant_compress=float(reweight.get("negative_dominant_compress", 0.0)),
            negative_dominant_positive_boost=float(
                reweight.get("negative_dominant_positive_boost", 0.0)
            ),
            polarity_dominance_margin=float(reweight.get("dominance_margin", 0.1)),
            pair_compensation_patch_size=_resolve_patch_size(pair.get("patch_size", 0)),
            pair_compensation_alpha=float(pair.get("alpha", 0.0)),
            pair_compensation_loser_attenuation=float(pair.get("loser_attenuation", 0.25)),
            pair_compensation_negative_bias=float(pair.get("negative_bias", 0.0)),
            pair_compensation_dominance_margin=float(pair.get("dominance_margin", 0.1)),
            pair_compensation_energy_gate_start=float(pair.get("energy_gate_start", 0.0)),
            pair_compensation_energy_gate_end=float(pair.get("energy_gate_end", 0.0)),
            pair_compensation_peak_gate_start=float(pair.get("peak_gate_start", 0.0)),
            pair_compensation_peak_gate_end=float(pair.get("peak_gate_end", 0.0)),
            response_floor=float(response.get("floor", 0.0)),
            response_gamma=float(response.get("gamma", 1.0)),
            time_steps=int(frontend_config.get("time_steps", 40)),
        )
    )
