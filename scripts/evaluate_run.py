#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal_evidence_snn.evaluate import evaluate_run, format_evaluation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a completed run.")
    parser.add_argument("run_dir", type=Path, help="Run directory to inspect.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    result = evaluate_run(args.run_dir)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(format_evaluation(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
