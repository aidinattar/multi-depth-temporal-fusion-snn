#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal_evidence_snn.data.dataset_contract import validate_dataset_npz  # noqa: E402
from temporal_evidence_snn.data.nmnist_events import prepare_nmnist_npz  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare an N-MNIST event-surface NPZ from raw binary event files."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Raw N-MNIST root.")
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ file.")
    parser.add_argument("--width", type=int, default=34, help="Sensor width.")
    parser.add_argument("--height", type=int, default=34, help="Sensor height.")
    parser.add_argument("--bins", type=int, default=5, help="Temporal event-surface bins.")
    parser.add_argument(
        "--normalize",
        choices=["sample", "channel", "none"],
        default="sample",
        help="Surface normalization mode.",
    )
    parser.add_argument("--log1p", dest="log1p", action="store_true", default=True)
    parser.add_argument("--no-log1p", dest="log1p", action="store_false")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--local-norm-radius", type=int, default=2)
    parser.add_argument("--local-norm-eps", type=float, default=1.0e-4)
    parser.add_argument("--denoise-filter-time-us", type=float, default=10000.0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    prepared = prepare_nmnist_npz(
        args.input_root,
        args.output,
        width=args.width,
        height=args.height,
        bins=args.bins,
        normalize=args.normalize,
        log1p=args.log1p,
        gamma=args.gamma,
        local_norm_radius=args.local_norm_radius,
        local_norm_eps=args.local_norm_eps,
        denoise_filter_time_us=args.denoise_filter_time_us,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        overwrite=args.overwrite,
    )
    validation = validate_dataset_npz(args.output, dataset_name="nmnist")
    payload = {
        "prepared": asdict(prepared),
        "validation": validation.to_dict(),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"output_path: {prepared.output_path}")
        print(f"train_samples: {prepared.train_samples}")
        print(f"test_samples: {prepared.test_samples}")
        print(f"surface_shape: {prepared.surface_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
