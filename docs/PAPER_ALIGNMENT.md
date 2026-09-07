# DRQ-DETR Paper Alignment

This document records the repository-to-paper alignment used for the Drones
manuscript. It is intentionally compact: the README gives the reviewer-facing
guide, while this file keeps the fixed facts that should not drift across
future documentation edits.

## Primary Controlled Results

| Dataset | AP50 | AP |
|---|---:|---:|
| SARD | 92.41 | 62.22 |
| SeaDronesSee-ODv2 | 84.62 | 52.65 |
| VisDrone2019 | 46.83 | 29.14 |

These results belong to the unified controlled evaluation protocol used for
configuration comparison, ablation, and GitHub homepage reporting.

## Final Paper Results

| Dataset | Final evaluation split | AP | AP50 |
|---|---|---:|---:|
| SARD | test | 61.53 | 91.44 |
| SeaDronesSee-ODv2 | validation | 52.71 | 84.58 |
| VisDrone2019 | test-dev | 22.65 | 38.37 |

These are the final paper values reported in the manuscript abstract and main
comparison table. They should not be mixed with the controlled validation
values above.

## Final Model

| Item | Fixed value |
|---|---:|
| Baseline family | DEIM |
| Backbone | HGNetV2 |
| Input size | 640 x 640 |
| Total decoder queries | 300 |
| Detail queries | 64 |
| Standard queries | 236 |
| SDQ first-stage candidates | 1024 |
| SDQ final detail-query quota | 64 |
| Thin P2 width | 64 channels |
| Decoder hidden dimension | 256 |
| Decoder layers | 3 |
| Decoder value levels | 4 |
| Decoder value strides | 4 / 8 / 16 / 32 |
| DSPR semantic weight | 0.20 |
| CGRF P3/P4 sharing | non-shared independent instances |

## Query Flow

```text
DSPR proxy
-> detached feature-energy ranking
-> Top-1024 proxy candidates
-> shared classification scoring
-> Top-64 detail queries

Top-64 detail queries + Top-236 standard multi-scale queries
-> 300 total decoder queries
```

## Decoder Values

```text
V2 / V3 / V4 / V5
stride 4 / 8 / 16 / 32
```

The final architecture file contains `[4, 4, 8, 16, 32]` at the decoder input.
The first stride-4 feature is the DSPR proxy used by SDQ; the remaining four
features are decoder values.

## Efficiency

| Item | Paper value |
|---|---:|
| Hardware | NVIDIA GeForce RTX 4060 Ti |
| Input | 640 x 640 |
| Batch size | 1 |
| Precision | FP32 |
| Warmup | 30 iterations |
| Timed iterations | 100 |
| Scope | model forward only |

Paper efficiency comparison on VisDrone2019 controlled validation:

| Method | AP50 | AP | APs | Params (M) | GFLOPs | Latency (ms) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEIM | 37.35 | 22.07 | 15.15 | 10.232 | 24.86 | 17.30 | 57.81 |
| DRQ-DETR | 46.83 | 29.14 | 21.33 | 12.008 | 54.23 | 35.58 | 28.10 |

## Source Paths

| Component | Repository path |
|---|---|
| DSPR | `engine/extre_module/custom_nn/neck/DSPR.py` |
| CGRF | `engine/extre_module/custom_nn/neck/DSPR.py` |
| SDQ | `engine/deim/dfine_decoder.py` |
| Final model graph | `configs/models/drq_detr_p2_64.yml` |
| Final experiment entries | `configs/experiments/*/drq_detr.yml` |
| FPS benchmark | `scripts/benchmark_fps.py` |
