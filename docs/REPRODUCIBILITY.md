# Reproducibility Protocol

This document defines the protocol used for fair comparison and reviewer
verification.

## Reference Software

| Component | Reference version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.3.0 |
| torchvision | 0.18.0 |
| Input format | RGB tensor |
| Evaluation | COCO bounding-box metrics |

The exact CUDA build may vary with the GPU driver. Record `torch.__version__`,
`torchvision.__version__`, `torch.version.cuda`, and the GPU model for every
reported efficiency run.

## Fixed Training Protocol

The common config is `configs/experiments/_fair132_common.yml`.

- Train from scratch with pretrained backbone initialization disabled.
- Train for 132 epochs.
- Use total batch size 12.
- Use AdamW with base learning rate 0.0004.
- Use backbone learning rate 0.0002.
- Use weight decay 0.0001.
- Use AMP and EMA.
- Use 640 x 640 validation images.
- Use the same augmentation schedule for all compared DRQ-DETR ablations.
- Stop strong augmentation and multi-scale sampling at epoch 120.
- Use `best_stg2.pth` for final evaluation.

The total batch size is a protocol value, not a per-GPU value. If distributed
training is used, preserve the same global batch size and record the world size.
The original sensitivity jobs were scheduled as independent single-process
experiments, one experiment per GPU.

## Randomness

Use an explicit seed:

```bash
python train.py \
  -c configs/experiments/visdrone2019/drq_detr.yml \
  --seed 0
```

CUDA kernels and data-loader scheduling can still introduce small numerical
variation. Report the exact seed and avoid combining results from mismatched
software stacks.

## Accuracy Evaluation

1. Load the matching dataset config.
2. Load `best_stg2.pth`.
3. Confirm EMA is selected.
4. Evaluate at 640 x 640.
5. Record AP, AP50, AP75, APs, APm, and APl.
6. Preserve the raw evaluator artifact and console log.
7. Confirm category counts and remapping before comparing datasets.

## Efficiency Evaluation

The default paper-aligned benchmark uses:

| Item | Setting |
|---|---:|
| Device | One GPU |
| Batch size | 1 |
| Input | Synthetic 640 x 640 tensor |
| Precision | FP32 |
| Warmup | 30 iterations |
| Timed iterations | 100 |
| Timing | CUDA events |
| Synchronization | Before reading elapsed time |
| Scope | Model plus post-processing |

Run:

```bash
python scripts/benchmark_fps.py
```

Do not compare FPS values measured with different precision, batch size, image
size, post-processing scope, GPU, power mode, or software version.

## Release Audit

Run both checks before packaging:

```bash
python scripts/check_configs.py
python scripts/check_configs.py --build-model
```

Then search for machine-specific paths:

```bash
rg -n "^[A-Za-z]:|/home/|/mnt/[a-z]/" configs scripts docs README.md
```

The release should not contain datasets, private credentials, cloud host
addresses, local absolute paths, editor metadata, or untracked training caches.
