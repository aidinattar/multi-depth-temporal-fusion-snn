from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EventStream:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray


@dataclass(frozen=True)
class PreparedNMNIST:
    output_path: str
    train_samples: int
    test_samples: int
    surface_shape: tuple[int, int, int]


@dataclass(frozen=True)
class LabeledEventFile:
    path: Path
    label: int


def load_nmnist_bin_events(path: Path) -> EventStream:
    """Decode a standard N-MNIST binary event file.

    Each event is stored as five bytes: ``x``, ``y``, one byte containing the
    polarity bit and high timestamp bits, then two timestamp bytes.
    """

    raw = np.fromfile(Path(path), dtype=np.uint8)
    if raw.size % 5 != 0:
        raise ValueError(f"N-MNIST file has {raw.size} bytes, not a multiple of five: {path}")
    if raw.size == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return EventStream(x=empty, y=empty, t=empty.astype(np.float64), p=empty)

    data = raw.reshape(-1, 5).astype(np.uint32, copy=False)
    x = data[:, 0]
    y = data[:, 1]
    p = (data[:, 2] >> 7) & 1
    t = ((data[:, 2] & 0x7F) << 16) | (data[:, 3] << 8) | data[:, 4]

    # N-MNIST records timestamp overflows as rows with y=240. Tonic's
    # canonical reader adds 2**13 to this and every subsequent timestamp for
    # each marker, then removes the marker rows.
    for overflow_index in np.where(y == 240)[0]:
        t[overflow_index:] += 2**13
    event_rows = y != 240
    x = x[event_rows]
    y = y[event_rows]
    p = p[event_rows]
    t = t[event_rows]
    return EventStream(
        x=x.astype(np.int64, copy=False),
        y=y.astype(np.int64, copy=False),
        t=t.astype(np.float64, copy=False),
        p=p.astype(np.int64, copy=False),
    )


def _split_dir(root: Path, split: str) -> Path:
    names = {
        "train": ("Train", "train", "Training", "training"),
        "test": ("Test", "test", "Testing", "testing"),
    }[split]
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find N-MNIST {split} split under {root}. "
        f"Expected one of: {', '.join(names)}"
    )


def collect_nmnist_event_files(root: Path, *, split: str) -> list[LabeledEventFile]:
    """Collect event files from an N-MNIST split in deterministic order."""

    split_path = _split_dir(Path(root), split)
    samples: list[LabeledEventFile] = []
    for class_dir in sorted(split_path.iterdir(), key=lambda path: path.name):
        if not class_dir.is_dir():
            continue
        try:
            label = int(class_dir.name)
        except ValueError:
            continue
        for event_file in sorted(class_dir.glob("*.bin")):
            samples.append(LabeledEventFile(path=event_file, label=label))
    if not samples:
        raise FileNotFoundError(f"No N-MNIST .bin files found in {split_path}")
    # Match Tonic's deterministic os.walk order: sorted class directories,
    # followed by sorted files within each directory.
    return samples


def _limit_samples(samples: list[LabeledEventFile], limit: int) -> list[LabeledEventFile]:
    count = int(limit)
    if count <= 0:
        return samples
    return samples[:count]


def _valid_events(
    events: EventStream,
    *,
    width: int,
    height: int,
    polarities: int,
) -> EventStream:
    valid = (events.x >= 0) & (events.x < width)
    valid &= (events.y >= 0) & (events.y < height)
    valid &= (events.p >= 0) & (events.p < polarities)
    return EventStream(
        x=events.x[valid],
        y=events.y[valid],
        t=events.t[valid],
        p=events.p[valid],
    )


def apply_local_temporal_denoise(
    events: EventStream,
    *,
    width: int,
    height: int,
    polarities: int,
    filter_time_us: float,
) -> EventStream:
    """Apply the exact local temporal support rule used by Tonic Denoise."""

    filter_time = float(filter_time_us)
    if filter_time <= 0.0 or events.x.size == 0:
        return events

    del width, height, polarities
    dynamic_width = int(events.x.max()) + 1
    dynamic_height = int(events.y.max()) + 1
    keep = np.zeros(events.x.shape[0], dtype=bool)
    timestamp_memory = np.zeros((dynamic_width, dynamic_height)) + filter_time
    for index in range(events.x.shape[0]):
        x = int(events.x[index])
        y = int(events.y[index])
        t = float(events.t[index])
        timestamp_memory[x, y] = t + filter_time
        keep[index] = bool(
            (x > 0 and timestamp_memory[x - 1, y] > t)
            or (x < dynamic_width - 1 and timestamp_memory[x + 1, y] > t)
            or (y > 0 and timestamp_memory[x, y - 1] > t)
            or (y < dynamic_height - 1 and timestamp_memory[x, y + 1] > t)
        )

    return EventStream(
        x=events.x[keep],
        y=events.y[keep],
        t=events.t[keep],
        p=events.p[keep],
    )


def _normalize_surface(
    surface: np.ndarray,
    *,
    normalize: str,
    log1p: bool,
    gamma: float,
    local_norm_radius: int,
    local_norm_eps: float,
) -> np.ndarray:
    output = torch.from_numpy(surface.astype(np.float32, copy=False))
    if bool(log1p):
        output = torch.log1p(output)

    if int(local_norm_radius) > 0:
        radius = int(local_norm_radius)
        local = F.avg_pool2d(
            output.unsqueeze(0),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
            count_include_pad=False,
        ).squeeze(0)
        output = output / (local + float(local_norm_eps))

    mode = str(normalize).lower()
    if mode == "sample":
        output = output / output.amax().clamp_min(1.0e-6)
    elif mode == "channel":
        denom = output.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        output = output / denom
    elif mode in {"none", "off"}:
        output = output.clamp(0.0, 1.0)
    else:
        raise ValueError("normalize must be one of: sample, channel, none")

    if float(gamma) != 1.0:
        output = output.clamp(0.0, 1.0).pow(float(gamma))
    return output.clamp(0.0, 1.0).numpy()


def nmnist_events_to_surface(
    events: EventStream,
    *,
    width: int = 34,
    height: int = 34,
    polarities: int = 2,
    bins: int = 5,
    normalize: str = "sample",
    log1p: bool = True,
    gamma: float = 1.0,
    local_norm_radius: int = 2,
    local_norm_eps: float = 1.0e-4,
    denoise_filter_time_us: float = 10000.0,
) -> np.ndarray:
    """Convert one N-MNIST event stream to a polarity-by-time-bin surface."""

    if int(polarities) != 2:
        raise ValueError(f"N-MNIST export expects two polarities, got {polarities}")
    if int(bins) <= 0:
        raise ValueError(f"bins must be positive, got {bins}")

    valid = _valid_events(events, width=int(width), height=int(height), polarities=int(polarities))
    supported = apply_local_temporal_denoise(
        valid,
        width=int(width),
        height=int(height),
        polarities=int(polarities),
        filter_time_us=float(denoise_filter_time_us),
    )

    channels = int(bins) * int(polarities)
    surface = np.zeros((channels, int(height), int(width)), dtype=np.float32)
    if supported.x.size == 0:
        return surface

    t_min = float(np.min(supported.t))
    t_span = max(float(np.max(supported.t) - t_min), 1.0)
    bin_index = np.floor((supported.t - t_min) / t_span * int(bins)).astype(np.int64)
    bin_index = np.clip(bin_index, 0, int(bins) - 1)
    channel = bin_index * int(polarities) + supported.p
    np.add.at(surface, (channel, supported.y, supported.x), 1.0)

    return _normalize_surface(
        surface,
        normalize=normalize,
        log1p=bool(log1p),
        gamma=float(gamma),
        local_norm_radius=int(local_norm_radius),
        local_norm_eps=float(local_norm_eps),
    )


def _build_split(
    samples: list[LabeledEventFile],
    *,
    width: int,
    height: int,
    polarities: int,
    bins: int,
    normalize: str,
    log1p: bool,
    gamma: float,
    local_norm_radius: int,
    local_norm_eps: float,
    denoise_filter_time_us: float,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for sample in samples:
        events = load_nmnist_bin_events(sample.path)
        images.append(
            nmnist_events_to_surface(
                events,
                width=width,
                height=height,
                polarities=polarities,
                bins=bins,
                normalize=normalize,
                log1p=log1p,
                gamma=gamma,
                local_norm_radius=local_norm_radius,
                local_norm_eps=local_norm_eps,
                denoise_filter_time_us=denoise_filter_time_us,
            )
        )
        labels.append(int(sample.label))
    return np.stack(images).astype(np.float32, copy=False), np.asarray(labels, dtype=np.int16)


def prepare_nmnist_npz(
    input_root: Path,
    output_path: Path,
    *,
    width: int = 34,
    height: int = 34,
    polarities: int = 2,
    bins: int = 5,
    normalize: str = "sample",
    log1p: bool = True,
    gamma: float = 1.0,
    local_norm_radius: int = 2,
    local_norm_eps: float = 1.0e-4,
    denoise_filter_time_us: float = 10000.0,
    train_limit: int = 0,
    test_limit: int = 0,
    overwrite: bool = False,
) -> PreparedNMNIST:
    """Prepare a public N-MNIST event-surface NPZ from raw binary event files."""

    root = Path(input_root)
    output = Path(output_path)
    if output.exists() and not bool(overwrite):
        raise FileExistsError(f"Output file already exists: {output}")

    train_files = _limit_samples(
        collect_nmnist_event_files(root, split="train"),
        int(train_limit),
    )
    test_files = _limit_samples(
        collect_nmnist_event_files(root, split="test"),
        int(test_limit),
    )

    common = dict(
        width=int(width),
        height=int(height),
        polarities=int(polarities),
        bins=int(bins),
        normalize=normalize,
        log1p=bool(log1p),
        gamma=float(gamma),
        local_norm_radius=int(local_norm_radius),
        local_norm_eps=float(local_norm_eps),
        denoise_filter_time_us=float(denoise_filter_time_us),
    )
    train_images, train_labels = _build_split(train_files, **common)
    test_images, test_labels = _build_split(test_files, **common)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )
    return PreparedNMNIST(
        output_path=str(output),
        train_samples=int(train_images.shape[0]),
        test_samples=int(test_images.shape[0]),
        surface_shape=tuple(int(value) for value in train_images.shape[1:]),
    )


__all__ = [
    "EventStream",
    "LabeledEventFile",
    "PreparedNMNIST",
    "apply_local_temporal_denoise",
    "collect_nmnist_event_files",
    "load_nmnist_bin_events",
    "nmnist_events_to_surface",
    "prepare_nmnist_npz",
]
