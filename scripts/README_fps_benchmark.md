# FPS Benchmark

`benchmark_fps.py` provides a repeatable latency and FPS measurement for the
released DRQ-DETR checkpoints.

## Default Protocol

```text
device: cuda:0
input size: 640 x 640
batch size: 1
precision: FP32
warmup: 30 iterations
timed iterations: 100
post-processing: included
timing: synchronized CUDA events
```

## Checkpoint Layout

The default manifest expects:

```text
checkpoints/sard/drq_detr/best_stg2.pth
checkpoints/seadronessee_odv2/drq_detr/best_stg2.pth
checkpoints/visdrone2019/drq_detr/best_stg2.pth
```

Edit a copy of the manifest or pass another manifest when checkpoints are
stored elsewhere. Do not commit private absolute paths.

## Linux or WSL

```bash
python scripts/benchmark_fps.py \
  --manifest scripts/fps_benchmark_manifest.json \
  --output outputs/benchmarks/fps_results.csv \
  --json-output outputs/benchmarks/fps_results.json \
  --device cuda:0 \
  --imgsz 640 \
  --batch-size 1 \
  --warmup 30 \
  --iters 100 \
  --precision fp32
```

Run only matching entries:

```bash
python scripts/benchmark_fps.py --only VisDrone2019
```

Exclude post-processing only for a separately labeled model-only analysis:

```bash
python scripts/benchmark_fps.py --no-postprocess
```

## Windows PowerShell with WSL

```powershell
.\scripts\run_fps_benchmark_wsl.ps1
```

Custom example:

```powershell
.\scripts\run_fps_benchmark_wsl.ps1 `
  -Manifest scripts\fps_benchmark_manifest_ablation.json `
  -Output outputs\benchmarks\fps_ablation.csv `
  -JsonOutput outputs\benchmarks\fps_ablation.json `
  -Device cuda:0 `
  -Precision fp32
```

The wrapper activates `torch_2_3_0_py310` inside WSL. Change that environment
name in the wrapper when using another local environment.

## Output Fields

The CSV and JSON include:

- dataset, group, model, backend, and architecture label;
- GPU model, GPU memory, CUDA version, and PyTorch version;
- precision, image size, batch size, warmup, and iteration count;
- model, post-processing, total, p50, and p95 latency;
- model-only and total FPS;
- total and trainable parameter counts;
- checkpoint weight source and loading coverage;
- status and error message.

For the paper table, use `fps_total` and `latency_total_ms` from runs with
identical hardware and settings.
