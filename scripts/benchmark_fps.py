#!/usr/bin/env python3
"""Benchmark inference latency and FPS for DRQ-DETR paper experiments.

The script is intentionally standalone: it does not modify training/evaluation
code and can be re-run with different checkpoints, GPUs, image sizes, or
iteration counts. The default workflow is driven by a JSON manifest.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import importlib
import json
import os
import re
import statistics
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure FPS/latency for DRQ-DETR and comparison detectors."
    )
    parser.add_argument(
        "--manifest",
        default="scripts/fps_benchmark_manifest.json",
        help="JSON manifest containing benchmark runs.",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmarks/fps_results.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/benchmarks/fps_results.json",
        help="JSON output path.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Synthetic square input size.")
    parser.add_argument("--batch-size", type=int, default=1, help="Images per timing step.")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations.")
    parser.add_argument(
        "--precision",
        choices=("fp32", "amp", "fp16"),
        default="fp32",
        help="Inference precision. fp32 is the most comparable default.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="For DRQ/DEIM-style models, exclude the postprocessor from total FPS.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run entries whose dataset/model/name contains any provided substring.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Emit disabled manifest entries as skipped rows.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed run instead of writing an error row.",
    )
    return parser.parse_args()


def normalize_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip().strip('"')
    if not text:
        return None
    text = text.replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", text)
    if match and os.name != "nt":
        drive = match.group(1).lower()
        rest = match.group(2)
        return Path(f"/mnt/{drive}/{rest}")
    return Path(text)


def resolve_path(repo_root: Path, value: str | os.PathLike[str] | None) -> Path | None:
    path = normalize_path(value)
    if path is None:
        return None
    if path.is_absolute():
        return path
    return repo_root / path


def prepare_import_path(repo_root: Path, value: str | os.PathLike[str] | None, package: str) -> None:
    path = resolve_path(repo_root, value)
    if path is None:
        return
    if not path.exists():
        raise FileNotFoundError(f"Package path not found: {path}")
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

    loaded_paths = []
    for name, module in list(sys.modules.items()):
        if name == package or name.startswith(package + "."):
            module_file = getattr(module, "__file__", "")
            if module_file:
                loaded_paths.append(str(module_file))
            del sys.modules[name]
    importlib.invalidate_caches()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("runs"), list):
        return data["runs"]
    raise ValueError(f"Unsupported manifest format: {path}")


def entry_label(entry: Mapping[str, Any]) -> str:
    return " ".join(
        str(entry.get(key, "")) for key in ("dataset", "model", "name", "backend", "arch")
    ).lower()


def entry_selected(entry: Mapping[str, Any], filters: list[str] | None) -> bool:
    if not filters:
        return True
    label = entry_label(entry)
    return any(item.lower() in label for item in filters)


def ensure_repo_import(repo_root: Path) -> None:
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)


def count_params(model: torch.nn.Module) -> tuple[float, float]:
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total, trainable


def is_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return any(torch.is_tensor(v) for v in value.values())


def choose_state_dict(checkpoint: Any, prefer_ema: bool = True) -> tuple[Mapping[str, Any], str]:
    if is_state_dict(checkpoint):
        return checkpoint, "checkpoint"

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint is neither a state_dict nor a dict-like checkpoint.")

    if prefer_ema and "ema" in checkpoint:
        ema = checkpoint["ema"]
        if isinstance(ema, Mapping):
            if is_state_dict(ema.get("module")):
                return ema["module"], "ema.module"
            if is_state_dict(ema):
                return ema, "ema"

    for key in ("model", "state_dict", "module", "net"):
        value = checkpoint.get(key)
        if is_state_dict(value):
            return value, key

    raise KeyError(f"Could not find model weights in checkpoint keys: {list(checkpoint.keys())}")


def key_candidates(key: str) -> list[str]:
    candidates = [key]
    prefixes = ("module.", "model.", "ema.module.", "ema.")
    changed = True
    current = key
    while changed:
        changed = False
        for prefix in prefixes:
            if current.startswith(prefix):
                current = current[len(prefix) :]
                candidates.append(current)
                changed = True
    if key.startswith("model.module."):
        candidates.append(key[len("model.module.") :])
    return list(dict.fromkeys(candidates))


def load_weights(
    model: torch.nn.Module, checkpoint_path: Path, prefer_ema: bool = True
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state, source_name = choose_state_dict(checkpoint, prefer_ema=prefer_ema)
    target_state = model.state_dict()

    loadable: dict[str, torch.Tensor] = {}
    skipped = 0
    seen_sources: set[str] = set()
    for raw_key, value in source_state.items():
        if not torch.is_tensor(value):
            skipped += 1
            continue
        matched_key = None
        for candidate in key_candidates(raw_key):
            if candidate in target_state and target_state[candidate].shape == value.shape:
                matched_key = candidate
                break
        if matched_key is None:
            skipped += 1
            continue
        loadable[matched_key] = value
        seen_sources.add(raw_key)

    missing = [key for key in target_state if key not in loadable]
    incompatible = model.load_state_dict(loadable, strict=False)
    return {
        "weight_source": source_name,
        "loaded_keys": len(loadable),
        "missing_keys": len(missing),
        "unexpected_keys": len(incompatible.unexpected_keys),
        "skipped_keys": skipped,
        "source_keys": len(source_state),
        "used_source_keys": len(seen_sources),
    }


def materialize_config(
    repo_root: Path,
    config_value: str | None,
    temp_files: list[Path],
    package_path: Path | None = None,
) -> Path:
    import yaml

    config_path = resolve_path(repo_root, config_value)
    if config_path is None:
        raise ValueError("A DEIM/DRQ entry requires a config path.")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if config_path.suffix.lower() != ".json":
        return config_path

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    yaml_cfg = data.get("yaml_cfg")
    if not isinstance(yaml_cfg, Mapping):
        raise ValueError(f"JSON config does not contain yaml_cfg: {config_path}")

    yaml_cfg = copy.deepcopy(dict(yaml_cfg))
    yaml_cfg.pop("__include__", None)
    if package_path is not None:
        for value in yaml_cfg.values():
            if not isinstance(value, dict):
                continue
            yaml_path = value.get("yaml_path")
            if not isinstance(yaml_path, str) or Path(yaml_path).is_absolute():
                continue
            candidate = package_path / yaml_path
            if candidate.exists():
                value["yaml_path"] = str(candidate)

    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".yml", prefix="fps_cfg_", delete=False
    )
    with handle:
        yaml.safe_dump(yaml_cfg, handle, sort_keys=False, allow_unicode=True)
    temp_path = Path(handle.name)
    temp_files.append(temp_path)
    return temp_path


@contextlib.contextmanager
def autocast_context(device: torch.device, precision: str):
    if precision == "amp" and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mean_or_zero(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return float(ordered[idx])


def fps_from_latency(latency_ms: float) -> float:
    return 1000.0 / latency_ms if latency_ms > 0 else 0.0


def apply_precision(model: torch.nn.Module, precision: str) -> torch.nn.Module:
    if precision == "fp16":
        return model.half()
    return model


def build_deim_model(
    repo_root: Path, entry: Mapping[str, Any], device: torch.device, temp_files: list[Path]
) -> tuple[torch.nn.Module, torch.nn.Module | None, Path, dict[str, Any]]:
    package_path = resolve_path(repo_root, entry.get("package_path"))
    if package_path is not None:
        prepare_import_path(repo_root, entry.get("package_path"), "engine")
        code_path = package_path
    else:
        ensure_repo_import(repo_root)
        code_path = repo_root
    import engine  # noqa: F401  Registers modules.
    from engine.core import YAMLConfig

    config_path = materialize_config(repo_root, entry.get("config"), temp_files, package_path)
    overrides = dict(entry.get("overrides", {}))
    cfg = YAMLConfig(str(config_path), **overrides)
    model = cfg.model.to(device).eval()
    postprocessor = None
    if cfg.postprocessor is not None:
        postprocessor = cfg.postprocessor.to(device).eval()

    load_info: dict[str, Any] = {}
    checkpoint_path = resolve_path(repo_root, entry.get("checkpoint"))
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        load_info = load_weights(
            model, checkpoint_path, prefer_ema=bool(entry.get("prefer_ema", True))
        )
    load_info["code_path"] = str(code_path)
    return model, postprocessor, config_path, load_info


def build_rtdetr_official_model(
    repo_root: Path, entry: Mapping[str, Any], device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Module | None, Path, dict[str, Any]]:
    package_path = resolve_path(repo_root, entry.get("package_path"))
    if package_path is None or not package_path.exists():
        raise FileNotFoundError("RT-DETR entry requires package_path pointing to rtdetr_pytorch.")
    package_text = str(package_path)
    if package_text not in sys.path:
        sys.path.insert(0, package_text)

    import src.zoo.rtdetr  # noqa: F401  Registers RT-DETR modules.
    from src.core import YAMLConfig

    config_path = resolve_path(repo_root, entry.get("config"))
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"RT-DETR config not found: {config_path}")

    overrides = dict(entry.get("overrides", {}))
    cfg = YAMLConfig(str(config_path), **overrides)
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval() if cfg.postprocessor is not None else None

    load_info: dict[str, Any] = {}
    checkpoint_path = resolve_path(repo_root, entry.get("checkpoint"))
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        load_info = load_weights(
            model, checkpoint_path, prefer_ema=bool(entry.get("prefer_ema", True))
        )
    return model, postprocessor, config_path, load_info


def benchmark_deim(
    repo_root: Path,
    entry: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    temp_files: list[Path],
) -> dict[str, Any]:
    model, postprocessor, config_path, load_info = build_deim_model(
        repo_root, entry, device, temp_files
    )
    model = apply_precision(model, args.precision)

    images = torch.rand(args.batch_size, 3, args.imgsz, args.imgsz, device=device)
    if args.precision == "fp16":
        images = images.half()
    orig_sizes = torch.full(
        (args.batch_size, 2), args.imgsz, dtype=torch.float32, device=device
    )
    include_postprocess = not args.no_postprocess and postprocessor is not None

    with torch.inference_mode():
        for _ in range(args.warmup):
            with autocast_context(device, args.precision):
                outputs = model(images)
                if include_postprocess:
                    _ = postprocessor(outputs, orig_sizes)
        sync_if_cuda(device)

        model_times: list[float] = []
        post_times: list[float] = []
        total_times: list[float] = []
        for _ in range(args.iters):
            if device.type == "cuda":
                model_start = torch.cuda.Event(enable_timing=True)
                model_end = torch.cuda.Event(enable_timing=True)
                post_start = torch.cuda.Event(enable_timing=True)
                post_end = torch.cuda.Event(enable_timing=True)

                model_start.record()
                with autocast_context(device, args.precision):
                    outputs = model(images)
                model_end.record()

                if include_postprocess:
                    post_start.record()
                    _ = postprocessor(outputs, orig_sizes)
                    post_end.record()
                sync_if_cuda(device)

                model_ms = model_start.elapsed_time(model_end) / args.batch_size
                post_ms = (
                    post_start.elapsed_time(post_end) / args.batch_size
                    if include_postprocess
                    else 0.0
                )
            else:
                start = time.perf_counter()
                with autocast_context(device, args.precision):
                    outputs = model(images)
                mid = time.perf_counter()
                if include_postprocess:
                    _ = postprocessor(outputs, orig_sizes)
                end = time.perf_counter()
                model_ms = (mid - start) * 1000.0 / args.batch_size
                post_ms = ((end - mid) * 1000.0 / args.batch_size) if include_postprocess else 0.0

            total_ms = model_ms + post_ms
            model_times.append(model_ms)
            post_times.append(post_ms)
            total_times.append(total_ms)

    params_m, trainable_m = count_params(model)
    checkpoint = resolve_path(repo_root, entry.get("checkpoint"))
    return {
        "config": str(config_path),
        "checkpoint": str(checkpoint) if checkpoint else "",
        "include_postprocess": include_postprocess,
        "params_m": params_m,
        "trainable_params_m": trainable_m,
        "latency_model_ms": mean_or_zero(model_times),
        "latency_postprocess_ms": mean_or_zero(post_times),
        "latency_total_ms": mean_or_zero(total_times),
        "latency_total_p50_ms": percentile(total_times, 0.50),
        "latency_total_p95_ms": percentile(total_times, 0.95),
        "fps_model": fps_from_latency(mean_or_zero(model_times)),
        "fps_total": fps_from_latency(mean_or_zero(total_times)),
        **load_info,
    }


def benchmark_rtdetr_official(
    repo_root: Path,
    entry: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model, postprocessor, config_path, load_info = build_rtdetr_official_model(
        repo_root, entry, device
    )
    model = apply_precision(model, args.precision)

    images = torch.rand(args.batch_size, 3, args.imgsz, args.imgsz, device=device)
    if args.precision == "fp16":
        images = images.half()
    orig_sizes = torch.full(
        (args.batch_size, 2), args.imgsz, dtype=torch.float32, device=device
    )
    include_postprocess = not args.no_postprocess and postprocessor is not None

    with torch.inference_mode():
        for _ in range(args.warmup):
            with autocast_context(device, args.precision):
                outputs = model(images)
                if include_postprocess:
                    _ = postprocessor(outputs, orig_sizes)
        sync_if_cuda(device)

        model_times: list[float] = []
        post_times: list[float] = []
        total_times: list[float] = []
        for _ in range(args.iters):
            if device.type == "cuda":
                model_start = torch.cuda.Event(enable_timing=True)
                model_end = torch.cuda.Event(enable_timing=True)
                post_start = torch.cuda.Event(enable_timing=True)
                post_end = torch.cuda.Event(enable_timing=True)

                model_start.record()
                with autocast_context(device, args.precision):
                    outputs = model(images)
                model_end.record()

                if include_postprocess:
                    post_start.record()
                    _ = postprocessor(outputs, orig_sizes)
                    post_end.record()
                sync_if_cuda(device)

                model_ms = model_start.elapsed_time(model_end) / args.batch_size
                post_ms = (
                    post_start.elapsed_time(post_end) / args.batch_size
                    if include_postprocess
                    else 0.0
                )
            else:
                start = time.perf_counter()
                with autocast_context(device, args.precision):
                    outputs = model(images)
                mid = time.perf_counter()
                if include_postprocess:
                    _ = postprocessor(outputs, orig_sizes)
                end = time.perf_counter()
                model_ms = (mid - start) * 1000.0 / args.batch_size
                post_ms = ((end - mid) * 1000.0 / args.batch_size) if include_postprocess else 0.0

            total_ms = model_ms + post_ms
            model_times.append(model_ms)
            post_times.append(post_ms)
            total_times.append(total_ms)

    params_m, trainable_m = count_params(model)
    checkpoint = resolve_path(repo_root, entry.get("checkpoint"))
    return {
        "config": str(config_path),
        "checkpoint": str(checkpoint) if checkpoint else "",
        "include_postprocess": include_postprocess,
        "params_m": params_m,
        "trainable_params_m": trainable_m,
        "latency_model_ms": mean_or_zero(model_times),
        "latency_postprocess_ms": mean_or_zero(post_times),
        "latency_total_ms": mean_or_zero(total_times),
        "latency_total_p50_ms": percentile(total_times, 0.50),
        "latency_total_p95_ms": percentile(total_times, 0.95),
        "fps_model": fps_from_latency(mean_or_zero(model_times)),
        "fps_total": fps_from_latency(mean_or_zero(total_times)),
        **load_info,
    }


def build_torchvision_detector(
    entry: Mapping[str, Any], device: torch.device, imgsz: int
) -> torch.nn.Module:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn

    arch = str(entry.get("arch", "")).lower()
    num_classes = int(entry["num_classes"])
    common = {
        "weights": None,
        "weights_backbone": None,
        "min_size": imgsz,
        "max_size": imgsz,
    }
    if arch == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(num_classes=num_classes, **common)
    elif arch == "retinanet_resnet50_fpn":
        model = retinanet_resnet50_fpn(num_classes=num_classes, **common)
    else:
        raise ValueError(f"Unsupported torchvision detector arch: {arch}")
    return model.to(device).eval()


def benchmark_torchvision_detection(
    repo_root: Path,
    entry: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.precision != "fp32":
        raise ValueError("torchvision_detection backend currently supports fp32 only.")

    model = build_torchvision_detector(entry, device, args.imgsz)
    checkpoint_path = resolve_path(repo_root, entry.get("checkpoint"))
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    load_info = load_weights(model, checkpoint_path, prefer_ema=False)
    images = [
        torch.rand(3, args.imgsz, args.imgsz, device=device) for _ in range(args.batch_size)
    ]

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(images)
        sync_if_cuda(device)

        total_times: list[float] = []
        for _ in range(args.iters):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(images)
                end.record()
                sync_if_cuda(device)
                total_ms = start.elapsed_time(end) / args.batch_size
            else:
                tick = time.perf_counter()
                _ = model(images)
                sync_if_cuda(device)
                total_ms = (time.perf_counter() - tick) * 1000.0 / args.batch_size
            total_times.append(total_ms)

    params_m, trainable_m = count_params(model)
    return {
        "config": "",
        "checkpoint": str(checkpoint_path),
        "include_postprocess": True,
        "params_m": params_m,
        "trainable_params_m": trainable_m,
        "latency_model_ms": "",
        "latency_postprocess_ms": "",
        "latency_total_ms": mean_or_zero(total_times),
        "latency_total_p50_ms": percentile(total_times, 0.50),
        "latency_total_p95_ms": percentile(total_times, 0.95),
        "fps_model": "",
        "fps_total": fps_from_latency(mean_or_zero(total_times)),
        **load_info,
    }


def patch_ultralytics_legacy_aattn() -> None:
    """Support YOLOv12 checkpoints saved with legacy AAttn(qk, v) modules."""
    try:
        from ultralytics.nn.modules.block import AAttn
    except Exception:
        return

    if getattr(AAttn, "_drq_fps_legacy_patch", False):
        return

    original_forward = AAttn.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "qkv") or not (hasattr(self, "qk") and hasattr(self, "v")):
            return original_forward(self, x)

        bsz, _, height, width = x.shape
        num_tokens = height * width
        all_head_dim = getattr(self, "all_head_dim", self.head_dim * self.num_heads)
        qk = self.qk(x).flatten(2).transpose(1, 2)
        value = self.v(x).flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(bsz * self.area, num_tokens // self.area, all_head_dim * 2)
            value = value.reshape(bsz * self.area, num_tokens // self.area, all_head_dim)
            bsz, num_tokens, _ = qk.shape

        query, key = (
            qk.view(bsz, num_tokens, self.num_heads, self.head_dim * 2)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim], dim=2)
        )
        value = value.view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

        attn = (query.transpose(-2, -1) @ key) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        out = value @ attn.transpose(-2, -1)
        out = out.permute(0, 3, 1, 2)
        value = value.permute(0, 3, 1, 2)

        if self.area > 1:
            out = out.reshape(bsz // self.area, num_tokens * self.area, all_head_dim)
            value = value.reshape(bsz // self.area, num_tokens * self.area, all_head_dim)
            bsz, num_tokens, _ = out.shape

        out = out.reshape(bsz, height, width, all_head_dim).permute(0, 3, 1, 2).contiguous()
        value = value.reshape(bsz, height, width, all_head_dim).permute(0, 3, 1, 2).contiguous()
        out = out + self.pe(value)
        return self.proj(out)

    AAttn.forward = forward
    AAttn._drq_fps_legacy_patch = True


def benchmark_ultralytics(
    repo_root: Path,
    entry: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prepare_import_path(repo_root, entry.get("package_path"), "ultralytics")
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - depends on optional env.
        raise RuntimeError(
            "ultralytics is not installed in this environment. Install it or run "
            "YOLO/MFO entries in an environment where `from ultralytics import YOLO` works."
        ) from exc
    patch_ultralytics_legacy_aattn()

    checkpoint_path = resolve_path(repo_root, entry.get("checkpoint"))
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    yolo = YOLO(str(checkpoint_path))
    model = yolo.model.to(device).eval()
    model = apply_precision(model, args.precision)
    images = torch.rand(args.batch_size, 3, args.imgsz, args.imgsz, device=device)
    if args.precision == "fp16":
        images = images.half()

    with torch.inference_mode():
        for _ in range(args.warmup):
            with autocast_context(device, args.precision):
                _ = model(images)
        sync_if_cuda(device)

        total_times: list[float] = []
        for _ in range(args.iters):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with autocast_context(device, args.precision):
                    _ = model(images)
                end.record()
                sync_if_cuda(device)
                total_ms = start.elapsed_time(end) / args.batch_size
            else:
                tick = time.perf_counter()
                with autocast_context(device, args.precision):
                    _ = model(images)
                total_ms = (time.perf_counter() - tick) * 1000.0 / args.batch_size
            total_times.append(total_ms)

    params_m, trainable_m = count_params(model)
    return {
        "config": "",
        "checkpoint": str(checkpoint_path),
        "include_postprocess": False,
        "params_m": params_m,
        "trainable_params_m": trainable_m,
        "latency_model_ms": mean_or_zero(total_times),
        "latency_postprocess_ms": "",
        "latency_total_ms": mean_or_zero(total_times),
        "latency_total_p50_ms": percentile(total_times, 0.50),
        "latency_total_p95_ms": percentile(total_times, 0.95),
        "fps_model": fps_from_latency(mean_or_zero(total_times)),
        "fps_total": fps_from_latency(mean_or_zero(total_times)),
        "loaded_keys": "",
        "missing_keys": "",
        "unexpected_keys": "",
        "skipped_keys": "",
        "weight_source": "ultralytics",
    }


def benchmark_yolov5(
    repo_root: Path,
    entry: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.precision != "fp32":
        raise ValueError("yolov5 backend currently supports fp32 only.")

    package_path = resolve_path(repo_root, entry.get("package_path"))
    if package_path is None or not package_path.exists():
        raise FileNotFoundError("A YOLOv5/MFO entry requires package_path pointing to a YOLOv5 repo.")
    checkpoint_path = resolve_path(repo_root, entry.get("checkpoint"))
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    package_text = str(package_path)
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    os.environ.setdefault("RANK", "1")

    for module_name in list(sys.modules):
        if module_name == "models" or module_name.startswith("models.") or module_name == "utils" or module_name.startswith("utils."):
            del sys.modules[module_name]
    importlib.invalidate_caches()

    try:
        from models.common import DetectMultiBackend

        model = DetectMultiBackend(str(checkpoint_path), device=device, fp16=False)
    except ImportError:
        from models.experimental import attempt_load

        try:
            model = attempt_load(str(checkpoint_path), device=device, inplace=True, fuse=True)
        except TypeError:
            model = attempt_load(str(checkpoint_path), map_location=device, inplace=True, fuse=True)
        model = model.to(device)
    model.eval()
    images = torch.rand(args.batch_size, 3, args.imgsz, args.imgsz, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(images)
        sync_if_cuda(device)

        total_times: list[float] = []
        for _ in range(args.iters):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(images)
                end.record()
                sync_if_cuda(device)
                total_ms = start.elapsed_time(end) / args.batch_size
            else:
                tick = time.perf_counter()
                _ = model(images)
                total_ms = (time.perf_counter() - tick) * 1000.0 / args.batch_size
            total_times.append(total_ms)

    params_m, trainable_m = count_params(model)
    return {
        "config": "",
        "checkpoint": str(checkpoint_path),
        "include_postprocess": False,
        "params_m": params_m,
        "trainable_params_m": trainable_m,
        "latency_model_ms": mean_or_zero(total_times),
        "latency_postprocess_ms": "",
        "latency_total_ms": mean_or_zero(total_times),
        "latency_total_p50_ms": percentile(total_times, 0.50),
        "latency_total_p95_ms": percentile(total_times, 0.95),
        "fps_model": fps_from_latency(mean_or_zero(total_times)),
        "fps_total": fps_from_latency(mean_or_zero(total_times)),
        "loaded_keys": "",
        "missing_keys": "",
        "unexpected_keys": "",
        "skipped_keys": "",
        "weight_source": "yolov5",
    }


def device_metadata(device: torch.device) -> dict[str, Any]:
    row: dict[str, Any] = {"device": str(device), "gpu": "", "cuda": torch.version.cuda or ""}
    if device.type == "cuda":
        row["gpu"] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        row["gpu_memory_gb"] = round(props.total_memory / (1024**3), 3)
    else:
        row["gpu_memory_gb"] = ""
    row["torch"] = torch.__version__
    return row


def format_row(
    entry: Mapping[str, Any],
    args: argparse.Namespace,
    device_info: Mapping[str, Any],
    status: str,
    result: Mapping[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    row = {
        "dataset": entry.get("dataset", ""),
        "group": entry.get("group", ""),
        "model": entry.get("model", entry.get("name", "")),
        "backend": entry.get("backend", ""),
        "arch": entry.get("arch", ""),
        "note": entry.get("note", ""),
        "status": status,
        "error": error,
        "precision": args.precision,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        **device_info,
    }
    if result:
        row.update(result)
    return row


def write_outputs(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def cleanup_temp_files(paths: list[Path]) -> None:
    for path in paths:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = resolve_path(repo_root, args.manifest)
    output_path = resolve_path(repo_root, args.output)
    json_output_path = resolve_path(repo_root, args.json_output)
    assert manifest_path is not None and output_path is not None and json_output_path is not None

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    runs = [entry for entry in load_manifest(manifest_path) if entry_selected(entry, args.only)]
    device_info = device_metadata(device)
    temp_files: list[Path] = []
    rows: list[dict[str, Any]] = []

    backend_map = {
        "deim": benchmark_deim,
        "drq_detr": benchmark_deim,
        "rtdetr_official": benchmark_rtdetr_official,
        "torchvision_detection": benchmark_torchvision_detection,
        "ultralytics": benchmark_ultralytics,
        "yolov5": benchmark_yolov5,
    }

    try:
        for idx, entry in enumerate(runs, start=1):
            enabled = bool(entry.get("enabled", True))
            name = f"{entry.get('dataset', '')} / {entry.get('model', entry.get('name', ''))}"
            if not enabled:
                if args.include_disabled:
                    rows.append(
                        format_row(
                            entry,
                            args,
                            device_info,
                            "skipped",
                            error=str(entry.get("note", "disabled in manifest")),
                        )
                    )
                print(f"[{idx}/{len(runs)}] SKIP {name}")
                continue

            backend = str(entry.get("backend", "")).lower()
            print(f"[{idx}/{len(runs)}] RUN  {name} ({backend})", flush=True)
            try:
                if backend not in backend_map:
                    raise ValueError(f"Unsupported backend: {backend}")
                bench_fn = backend_map[backend]
                if backend in {"deim", "drq_detr"}:
                    result = bench_fn(repo_root, entry, device, args, temp_files)
                else:
                    result = bench_fn(repo_root, entry, device, args)
                rows.append(format_row(entry, args, device_info, "ok", result=result))
                print(
                    "          OK   "
                    f"FPS={rows[-1].get('fps_total', '')} "
                    f"Latency={rows[-1].get('latency_total_ms', '')} ms",
                    flush=True,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                rows.append(format_row(entry, args, device_info, "error", error=message))
                print(f"          ERR  {message}", flush=True)
                if args.stop_on_error:
                    traceback.print_exc()
                    raise

        write_outputs(rows, output_path, json_output_path)
        print(f"\nWrote CSV:  {output_path}")
        print(f"Wrote JSON: {json_output_path}")
    finally:
        cleanup_temp_files(temp_files)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
