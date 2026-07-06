# Reviewer Release Checklist

Use this checklist before packaging DRQ-DETR for GitHub or reviewer access.

## Public final model

- Final architecture: `configs/models/drq_detr_p2_64.yml`
- Main configs:
  - `configs/experiments/sard/drq_detr.yml`
  - `configs/experiments/seadronessee_odv2/drq_detr.yml`
  - `configs/experiments/visdrone2019/drq_detr.yml`
- SDQ settings: `sdq_pre_topk=1024`, `sdq_query_topk=64`
- Thin-P2 width: 64 channels
- DGA: disabled in all final configs
- Training protocol: 132 epochs, batch size 12, seed 0, no pretrained backbone

## Optional reproducibility configs

- Ablations are kept under each dataset folder with explicit names.
- Sensitivity runs are kept under `configs/experiments/visdrone2019/sensitivity/`.
- DGA is available only through files that explicitly include `with_dga` or
  `_dga_regularizer.yml`; it is not part of the final public model.

## Files not to upload

Do not upload local or generated artifacts to the source repository:

- `outputs/`, `runs/`, `logs/`, `checkpoints/`, `local_artifacts/`
- datasets under `data/` or `datasets/`
- model weights such as `*.pth`, `*.pt`, `*.onnx`, `*.engine`
- Python caches such as `__pycache__/`

Large checkpoints should be published as release assets or shared through a
separate storage link, with paths matching `docs/CHECKPOINTS.md`.

## Required validation

Run:

```bash
python scripts/check_configs.py
```

Optional model construction check:

```bash
python scripts/check_configs.py --build-model
```

The validation script checks final SDQ settings, DGA state, local machine paths,
missing architecture references, and FPS manifest paths.
