# DRQ-DETR configuration guide

This directory contains the configuration entry points used for the DRQ-DETR manuscript and its reproducibility checks. The configuration set is trimmed to the manuscript-relevant runs so that reviewers can locate the final model, ablation studies, sensitivity studies, and P2 access controls without legacy training variants.

## Main paper entry points

Use the following dataset-level experiment files for the final DRQ-DETR runs:

- `configs/experiments/sard/drq_detr.yml`
- `configs/experiments/seadronessee_odv2/drq_detr.yml`
- `configs/experiments/visdrone2019/drq_detr.yml`

All three point to the same architecture file:

- `configs/models/drq_detr_p2_64.yml`

The final architecture uses DSPR, CGRF, SDQ, and a 64-channel Thin P2 high-resolution value path. In decoder inputs, the DSPR proxy is used for sparse detail-query selection, while Thin P2/P3/P4/P5 provide the value features.

## SeaDronesSee-ODv2 reproduction map

Launch SeaDronesSee-ODv2 runs from `configs/experiments/seadronessee_odv2/`, not directly from `configs/models/`.

| Result to reproduce | Experiment config | Model graph |
|---|---|---|
| DEIM-S baseline | `baseline_deim_s.yml` | Built from the DEIM base config |
| DSPR + SDQ only | `ablation_sdq_only.yml` | `configs/models/drq_detr_sdq_only.yml` |
| DSPR + CGRF + SDQ, no Thin P2 value path | `ablation_sdq_cgrf_no_p2.yml` | `configs/models/drq_detr_sdq_cgrf.yml` |
| Thin P2-32 ablation | `ablation_p2_32.yml` | `configs/models/drq_detr_thinp2_32.yml` |
| Final DRQ-DETR, Thin P2-64, P1024-Q64 | `drq_detr.yml` | `configs/models/drq_detr_p2_64.yml` |
| Alternate P1536-Q96 setting | `drq_detr_p1536.yml` | `configs/models/drq_detr_p2_64_p1536_q96_arch.yml` |
| DEIM-S + direct P2 | `causal_p2/direct_p2_seed0.yml` | `configs/models/causal_p2/deim_s_direct_p2.yml` |
| DEIM-S + Thin P2 only | `causal_p2/thin_p2_only_seed0.yml` | `configs/models/causal_p2/deim_s_thin_p2_only.yml` |

## Baseline and component ablation entry points

For each dataset directory under `configs/experiments/`, the manuscript-related files have the following meaning:

- `baseline_deim_s.yml`: DEIM-S baseline under the same data protocol.
- `ablation_sdq_only.yml`: DSPR + SDQ only; no CGRF and no Thin P2 value path.
- `ablation_sdq_cgrf_no_p2.yml`: DSPR + CGRF + SDQ; no Thin P2 value path.
- `ablation_p2_32.yml`: DSPR + CGRF + SDQ with a 32-channel Thin P2 value path.
- `drq_detr.yml`: final DRQ-DETR with a 64-channel Thin P2 value path and Q64 detail queries.

Alternate P1536-Q96 ablation files are retained for sensitivity and auxiliary reproduction checks.

## P2 access strategy controls

Controlled high-resolution access experiments are stored in `configs/experiments/*/causal_p2/` and use model files from `configs/models/causal_p2/`:

- `deim_s_direct_p2.yml`: directly projects P2 and feeds it to the decoder.
- `deim_s_thin_p2_only.yml`: keeps a 64-channel Thin P2 value path.
- `deim_s_fpn_pan_p2.yml`: conventional full-width P2 FPN/PAN branch.
- `deim_s_fpn_pan_p2_budget.yml`: reduced-width FPN/PAN-style P2 branch for budget-aware comparison.

## Sensitivity architecture files

VisDrone2019 SDQ and Thin P2 sensitivity architectures are stored in:

- `configs/models/sensitivity/visdrone2019/`

The filename records the tested setting, for example `pre_topk_512_arch.yml`, `query_topk_96_arch.yml`, or `thin_p2_width_64_arch.yml`. These files are formatted in the same layer-by-layer style as the final architecture.

## Resolution notation

In comments and manuscript-facing names, `P2`, `P3`, `P4`, and `P5` denote feature levels corresponding to 1/4, 1/8, 1/16, and 1/32 input spatial resolution, respectively. The numeric stride is retained in inline comments such as `P2/4` only to make the architecture graph easier to audit.
