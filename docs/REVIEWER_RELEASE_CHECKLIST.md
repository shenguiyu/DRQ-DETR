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
- Training protocol: 132 epochs, batch size 12, seed 0, no pretrained backbone

## Optional reproducibility configs

- Ablations are kept under each dataset folder with explicit names.
- P2 access controls are kept under each dataset folder in `causal_p2/`.
- Sensitivity runs are kept under `configs/experiments/visdrone2019/sensitivity/`.

## External training result archive

- Baidu Netdisk: `https://pan.baidu.com/s/1ZkrMin5tb2IFsejybHesAQ?pwd=0871`
- Extraction code: `0871`
- Archive label: `训练结果`
- The archive is external to Git and should contain checkpoints plus raw
  training/evaluation logs for reviewer verification.

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

The validation script checks final SDQ settings, local machine paths, missing
architecture references, and FPS manifest paths.
