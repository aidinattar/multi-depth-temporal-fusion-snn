#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal_evidence_snn.data import validate_dataset_npz  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a dataset NPZ export.")
    parser.add_argument("path", type=Path, help="Dataset NPZ file.")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist", "cifar10", "nmnist"],
        default=None,
        help="Optional dataset name used to validate the expected channel count.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    result = validate_dataset_npz(args.path, dataset_name=args.dataset).to_dict()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
