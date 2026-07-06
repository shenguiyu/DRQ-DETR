#!/usr/bin/env python3
"""Validate public DRQ-DETR experiment configs before a release."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
MAIN_CONFIGS = (
    "configs/experiments/sard/drq_detr.yml",
    "configs/experiments/seadronessee_odv2/drq_detr.yml",
    "configs/experiments/visdrone2019/drq_detr.yml",
)
EXPERIMENT_MANIFEST = REPO_ROOT / "scripts" / "experiments.json"
LOCAL_PATH_PATTERN = re.compile(r"^(?:[A-Za-z]:[\\/]|/home/|/mnt/[A-Za-z]/)")


def merge_dict(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            merge_dict(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML value must be a mapping: {path}")
    return data


def load_merged(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Config include cycle: {chain}")
    if not path.exists():
        raise FileNotFoundError(path)

    data = read_yaml(path)
    merged: dict[str, Any] = {}
    includes = data.get("__include__", [])
    if isinstance(includes, str):
        includes = [includes]
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        merge_dict(merged, load_merged(include_path, (*stack, path)))
    merge_dict(merged, data)
    return merged


def walk_values(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_values(child)
    else:
        yield value


def validate_experiment(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        merged = load_merged(path)
    except Exception as exc:
        return [f"{path.relative_to(REPO_ROOT)}: {type(exc).__name__}: {exc}"]

    for value in walk_values(read_yaml(path)):
        if isinstance(value, str) and LOCAL_PATH_PATTERN.match(value):
            errors.append(
                f"{path.relative_to(REPO_ROOT)} contains a machine-specific path: {value}"
            )

    model_cfg = merged.get("DRQ_DETR", {})
    if isinstance(model_cfg, dict) and "yaml_path" in model_cfg:
        architecture_path = REPO_ROOT / str(model_cfg["yaml_path"])
        if not architecture_path.exists():
            errors.append(
                f"{path.relative_to(REPO_ROOT)} references missing architecture "
                f"{model_cfg['yaml_path']}"
            )

    if merged.get("model") not in {"DEIM", "DRQ_DETR"}:
        errors.append(
            f"{path.relative_to(REPO_ROOT)} resolves unsupported model={merged.get('model')}"
        )
    return errors


def validate_main_configs() -> list[str]:
    errors: list[str] = []
    for relative in MAIN_CONFIGS:
        path = REPO_ROOT / relative
        merged = load_merged(path)
        architecture = merged.get("DRQ_DETR", {}).get("yaml_path")
        if architecture != "configs/models/drq_detr_p2_64.yml":
            errors.append(f"{relative} does not select the public P2-64 architecture")

        losses = merged.get("DEIMCriterion", {}).get("losses", [])
        weights = merged.get("DEIMCriterion", {}).get("weight_dict", {})
        if "dga" in losses or "loss_dga" in weights:
            errors.append(f"{relative} unexpectedly enables DGA")

        architecture_data = read_yaml(REPO_ROOT / architecture)
        decoder_args = architecture_data["decoder"][0][2]
        if decoder_args.get("sdq_pre_topk") != 1024:
            errors.append(f"{relative} has unexpected sdq_pre_topk")
        if decoder_args.get("sdq_query_topk") != 64:
            errors.append(f"{relative} has unexpected sdq_query_topk")
    return errors


def validate_manifests() -> list[str]:
    errors: list[str] = []
    for path in sorted((REPO_ROOT / "scripts").glob("fps_benchmark_manifest*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            runs = payload.get("runs", [])
            if not isinstance(runs, list) or not runs:
                errors.append(f"{path.relative_to(REPO_ROOT)} has no benchmark runs")
            for run in runs:
                config = run.get("config")
                if config and not (REPO_ROOT / config).exists():
                    errors.append(
                        f"{path.relative_to(REPO_ROOT)} references missing config {config}"
                    )
                for key in ("config", "checkpoint", "package_path"):
                    value = run.get(key)
                    if isinstance(value, str) and LOCAL_PATH_PATTERN.match(value):
                        errors.append(
                            f"{path.relative_to(REPO_ROOT)} contains a local {key}: {value}"
                        )
        except Exception as exc:
            errors.append(f"{path.relative_to(REPO_ROOT)}: {type(exc).__name__}: {exc}")
    return errors


def load_experiment_configs() -> list[str]:
    with EXPERIMENT_MANIFEST.open(encoding="utf-8") as handle:
        experiments = json.load(handle)
    return sorted({config for runs in experiments.values() for config in runs.values()})


def validate_experiment_manifest() -> list[str]:
    errors: list[str] = []
    try:
        configs = load_experiment_configs()
    except Exception as exc:
        return [f"{EXPERIMENT_MANIFEST.relative_to(REPO_ROOT)}: {type(exc).__name__}: {exc}"]
    for config in configs:
        if not (REPO_ROOT / config).exists():
            errors.append(f"Experiment manifest references missing config {config}")
    return errors


def build_experiment_models() -> list[str]:
    errors: list[str] = []
    snippet = (
        "from engine.core import YAMLConfig; "
        "cfg=YAMLConfig({config!r}); "
        "model=cfg.model; "
        "print(sum(p.numel() for p in model.parameters()))"
    )
    for relative in load_experiment_configs():
        command = [sys.executable, "-c", snippet.format(config=relative)]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip().splitlines()
            errors.append(f"{relative} model build failed: {' | '.join(details[-4:])}")
        else:
            last_line = result.stdout.strip().splitlines()[-1]
            print(f"BUILD {relative}: {int(last_line):,} parameters")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-model",
        action="store_true",
        help="Instantiate every model listed in scripts/experiments.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments = sorted((CONFIG_ROOT / "experiments").rglob("*.yml"))
    errors: list[str] = []
    for path in experiments:
        if path.name.startswith("_"):
            continue
        errors.extend(validate_experiment(path))
    errors.extend(validate_main_configs())
    errors.extend(validate_manifests())
    errors.extend(validate_experiment_manifest())
    if args.build_model:
        errors.extend(build_experiment_models())

    if errors:
        print("\nConfiguration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        f"OK: validated {len(experiments)} experiment configs, "
        f"{len(load_experiment_configs())} runnable entries, and all FPS manifests."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
