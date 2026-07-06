# Experiment Configuration Map

This document maps the public config names to the factors evaluated in the
paper. Every dataset uses the common 132-epoch protocol in
`configs/experiments/_fair132_common.yml`.

## Canonical Final Configs

| Dataset | Config | Architecture | DGA |
|---|---|---|---|
| SARD | `configs/experiments/sard/drq_detr.yml` | P2-64 | No |
| SeaDronesSee-ODv2 | `configs/experiments/seadronessee_odv2/drq_detr.yml` | P2-64 | No |
| VisDrone2019 | `configs/experiments/visdrone2019/drq_detr.yml` | P2-64 | No |

All three resolve to `configs/models/drq_detr_p2_64.yml`, which fixes
`sdq_pre_topk=1024`, `sdq_query_topk=64`, Thin-P2 width 64, and DGA disabled.

## Ablation Ladder

The same naming scheme is available in each dataset folder.

| Public config | SDQ/DSPR | CGRF | DGA | P2 value path |
|---|---:|---:|---:|---:|
| `ablation_sdq_only.yml` | Yes | No | No | No |
| `ablation_sdq_cgrf_no_p2_no_dga.yml` | Yes | Yes | No | No |
| `ablation_sdq_cgrf_no_p2_with_dga.yml` | Yes | Yes | Yes | No |
| `ablation_p2_32_no_dga.yml` | Yes | Yes | No | Width 32 |
| `ablation_p2_32_with_dga.yml` | Yes | Yes | Yes | Width 32 |
| `drq_detr.yml` | Yes | Yes | No | Width 64 |
| `drq_detr_with_dga.yml` | Yes | Yes | Yes | Width 64 |

DGA changes the training criterion only. Matched DGA and non-DGA configs use
the same network architecture, parameter count, and inference graph.

## Sensitivity Study

Sensitivity runs use VisDrone2019 and are stored in:

```text
configs/experiments/visdrone2019/sensitivity/
configs/models/sensitivity/visdrone2019/
```

| Config | `sdq_pre_topk` | `sdq_query_topk` | P2 width |
|---|---:|---:|---:|
| `center_p1024_q64_w32.yml` | 1024 | 64 | 32 |
| `pre_topk_512.yml` | 512 | 64 | 32 |
| `pre_topk_1536.yml` | 1536 | 64 | 32 |
| `query_topk_32.yml` | 1024 | 32 | 32 |
| `query_topk_96.yml` | 1024 | 96 | 32 |
| `thin_p2_width_16.yml` | 1024 | 64 | 16 |
| `thin_p2_width_64.yml` | 1024 | 64 | 64 |
| `combo_w64_q64_p1536.yml` | 1536 | 64 | 64 |
| `combo_w64_q96_p1024.yml` | 1024 | 96 | 64 |
| `combo_w64_q96_p1536.yml` | 1536 | 96 | 64 |

These configs keep DGA disabled so that SDQ quotas and Thin-P2 width are
isolated from the optional training regularizer. The final cross-dataset
P2-64 model is the separate canonical `drq_detr.yml` config.

## Legacy Aliases

Historical training products may refer to the following files:

| Legacy file | Canonical replacement |
|---|---|
| `drq_detr_full.yml` | `drq_detr.yml` |
| `ablation_thinp2_no_dga.yml` | `ablation_p2_32_no_dga.yml` |
| `ablation_sdq_cgrf_no_dga.yml` | `ablation_sdq_cgrf_no_p2_no_dga.yml` |
| `ablation_sdq_cgrf_dga_no_p2.yml` | `ablation_sdq_cgrf_no_p2_with_dga.yml` |

Aliases are retained to validate historical checkpoints. New commands and
release metadata should use canonical names.

## Example Commands

```bash
# Final model
python train.py \
  -c configs/experiments/sard/drq_detr.yml \
  --seed 0

# Matched DGA comparison
python train.py \
  -c configs/experiments/sard/drq_detr_with_dga.yml \
  --seed 0

# One sensitivity point
python train.py \
  -c configs/experiments/visdrone2019/sensitivity/query_topk_96.yml \
  --seed 0
```
