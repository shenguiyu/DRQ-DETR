#!/usr/bin/env python3
"""Run an experiment from scripts/experiments.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "experiments.json"


def load_experiments() -> dict[str, dict[str, str]]:
    with MANIFEST.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--experiment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def print_experiments(experiments: dict[str, dict[str, str]]) -> None:
    width = max(len(name) for runs in experiments.values() for name in runs)
    for dataset, runs in experiments.items():
        print(dataset)
        for name, config in runs.items():
            print(f"  {name:<{width}}  {config}")


def main() -> int:
    args = parse_args()
    experiments = load_experiments()
    if args.list:
        print_experiments(experiments)
        return 0

    if args.dataset not in experiments:
        raise SystemExit(f"Unknown dataset: {args.dataset}. Use --list.")
    if args.experiment not in experiments[args.dataset]:
        raise SystemExit(
            f"Unknown experiment: {args.experiment}. Use --list."
        )

    config = experiments[args.dataset][args.experiment]
    command = [
        sys.executable,
        "train.py",
        "-c",
        config,
        "--seed",
        str(args.seed),
    ]
    if args.checkpoint:
        command.extend(["-r", args.checkpoint])
    if args.test_only:
        command.append("--test-only")
    if args.set:
        command.extend(["-u", *args.set])

    print(" ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
