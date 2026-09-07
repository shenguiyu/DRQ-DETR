# Experiment Configuration Map

This document maps the public config names to the factors evaluated in the
paper. Every DRQ-DETR experiment uses the common 132-epoch protocol in
`configs/experiments/_fair132_common.yml`, while the DEIM-S baseline uses
`configs/experiments/_baseline132_common.yml`.

## Canonical Final Configs

| Dataset | Config | Architecture |
|---|---|---|
| SARD | `configs/experiments/sard/drq_detr.yml` | Thin-P2-64, P1024-Q64 |
| SeaDronesSee-ODv2 | `configs/experiments/seadronessee_odv2/drq_detr.yml` | Thin-P2-64, P1024-Q64 |
| VisDrone2019 | `configs/experiments/visdrone2019/drq_detr.yml` | Thin-P2-64, P1024-Q64 |

All three resolve to `configs/models/drq_detr_p2_64.yml`, which fixes
`sdq_pre_topk=1024`, `sdq_query_topk=64`, and Thin-P2 width 64.

## Ablation Ladder

The same naming scheme is available in each dataset folder.

| Public config | SDQ/DSPR | CGRF | P2 value path |
|---|---:|---:|---:|
| `baseline_deim_s.yml` | No | No | No |
| `ablation_sdq_only.yml` | Yes | No | No |
| `ablation_sdq_cgrf_no_p2.yml` | Yes | Yes | No |
| `ablation_p2_32.yml` | Yes | Yes | Width 32 |
| `drq_detr.yml` | Yes | Yes | Width 64 |

Alternate P1536-Q96 files are retained for auxiliary checks:

```text
ablation_sdq_only_p1536.yml
ablation_sdq_cgrf_no_p2_p1536.yml
drq_detr_p1536.yml
```

## P2 Access Strategy Controls

Controlled high-resolution access experiments are stored in
`configs/experiments/*/causal_p2/` and use model graphs from
`configs/models/causal_p2/`.

| Public config | Purpose | Model graph |
|---|---|---|
| `direct_p2_seed0.yml` | DEIM-S with directly projected P2 value feature | `configs/models/causal_p2/deim_s_direct_p2.yml` |
| `thin_p2_only_seed0.yml` | DEIM-S with Thin-P2 value path only | `configs/models/causal_p2/deim_s_thin_p2_only.yml` |
| `fpn_pan_p2_seed0.yml` | DEIM-S with conventional full-width P2 FPN/PAN path | `configs/models/causal_p2/deim_s_fpn_pan_p2.yml` |
| `fpn_pan_p2_budget_seed0.yml` | Budget-aware P2 FPN/PAN control | `configs/models/causal_p2/deim_s_fpn_pan_p2_budget.yml` |

Only the controls that were actually run for a dataset are provided in that
dataset directory.

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

The final cross-dataset P2-64 model is the separate canonical `drq_detr.yml`
config.

## Example Commands

```bash
# Final model
python train.py \
  -c configs/experiments/seadronessee_odv2/drq_detr.yml \
  --seed 0

# One P2 access control
python train.py \
  -c configs/experiments/seadronessee_odv2/causal_p2/direct_p2_seed0.yml

# One sensitivity point
python train.py \
  -c configs/experiments/visdrone2019/sensitivity/query_topk_96.yml \
  --seed 0
```
