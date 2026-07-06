# Configuration Guide

DRQ-DETR uses recursive YAML composition through `__include__`.

## Composition Order

A final experiment config resolves in this order:

```text
runtime and base model
        |
shared fair-training protocol
        |
dataset definition
        |
architecture selection
        |
optional experiment override
```

Later files override earlier values. For example,
`drq_detr_with_dga.yml` includes `drq_detr.yml` and then
`_dga_regularizer.yml`, so only the criterion is changed.

## Directories

- `base/`: runtime, optimizer, data augmentation, and DEIM/D-FINE defaults.
- `datasets/`: paths, class counts, category remapping, and evaluator settings.
- `models/`: network graph definitions consumed by `DRQ_DETR`.
- `experiments/`: runnable training and evaluation entry points.

## Canonical Names

- Final model: `drq_detr.yml`
- Matched DGA run: `drq_detr_with_dga.yml`
- P2 width ablation: `ablation_p2_<width>_<dga-state>.yml`
- No-P2 feature ablation:
  `ablation_sdq_cgrf_no_p2_<dga-state>.yml`
- Sensitivity runs: one parameter value per filename, with explicit joint
  settings prefixed by `combo_`

## Validation

```bash
python scripts/check_configs.py
```

The validator checks include chains, architecture references, public model
selection, accidental local paths, final DGA state, SDQ defaults, and FPS
manifest references.
