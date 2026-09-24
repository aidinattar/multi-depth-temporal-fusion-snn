from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from temporal_evidence_snn.data import validate_dataset_npz
from temporal_evidence_snn.data.nmnist_events import (
    EventStream,
    load_nmnist_bin_events,
    nmnist_events_to_surface,
)


def _write_nmnist_bin(path: Path, events: list[tuple[int, int, int, int]]) -> None:
    payload = bytearray()
    for x, y, t, p in events:
        payload.append(int(x) & 0xFF)
        payload.append(int(y) & 0xFF)
        payload.append(((int(p) & 1) << 7) | ((int(t) >> 16) & 0x7F))
        payload.append((int(t) >> 8) & 0xFF)
        payload.append(int(t) & 0xFF)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(payload))


def _make_tiny_nmnist_tree(root: Path) -> None:
    _write_nmnist_bin(root / "Train" / "0" / "sample_0.bin", [(2, 3, 10, 0), (2, 3, 20, 0)])
    _write_nmnist_bin(root / "Train" / "1" / "sample_0.bin", [(5, 7, 11, 1), (5, 8, 30, 1)])
    _write_nmnist_bin(root / "Test" / "0" / "sample_0.bin", [(1, 2, 15, 0), (1, 3, 25, 0)])
    _write_nmnist_bin(root / "Test" / "1" / "sample_0.bin", [(9, 4, 12, 1), (10, 4, 40, 1)])


def test_load_nmnist_bin_events_decodes_standard_event_format(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    _write_nmnist_bin(path, [(4, 8, 12345, 1), (6, 9, 54321, 0)])

    events = load_nmnist_bin_events(path)

    assert events.x.tolist() == [4, 6]
    assert events.y.tolist() == [8, 9]
    assert events.p.tolist() == [1, 0]
    assert events.t.tolist() == [12345.0, 54321.0]


def test_nmnist_events_to_surface_builds_ten_channel_summary() -> None:
    events = EventStream(
        x=np.array([2, 2, 5], dtype=np.int64),
        y=np.array([3, 3, 6], dtype=np.int64),
        t=np.array([10.0, 20.0, 30.0], dtype=np.float64),
        p=np.array([0, 0, 1], dtype=np.int64),
    )

    surface = nmnist_events_to_surface(
        events,
        bins=5,
        denoise_filter_time_us=0.0,
        local_norm_radius=0,
    )

    assert surface.shape == (10, 34, 34)
    assert surface.dtype == np.float32
    assert float(surface.max()) == 1.0
    assert np.count_nonzero(surface) == 3


def test_prepare_nmnist_npz_cli_writes_valid_dataset(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    input_root = tmp_path / "N-MNIST"
    output = tmp_path / "N-MNIST.npz"
    _make_tiny_nmnist_tree(input_root)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_nmnist_npz.py",
            "--input-root",
            str(input_root),
            "--output",
            str(output),
            "--denoise-filter-time-us",
            "0",
            "--json",
        ],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert '"train_samples": 2' in completed.stdout
    result = validate_dataset_npz(output, dataset_name="nmnist")
    assert result.channels == 10
    assert result.train_samples == 2
    assert result.test_samples == 2
