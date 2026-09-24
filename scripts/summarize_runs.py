#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal_evidence_snn.summarize import (  # noqa: E402
    rows_to_markdown,
    rows_to_tsv,
    summarize_runs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize one or more runs.")
    parser.add_argument("root", type=Path, help="Run directory or run root.")
    parser.add_argument(
        "--format",
        choices=["tsv", "markdown", "json"],
        default="tsv",
        help="Output format.",
    )
    args = parser.parse_args(argv)

    rows = summarize_runs(args.root)
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.format == "markdown":
        print(rows_to_markdown(rows), end="")
    else:
        print(rows_to_tsv(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
