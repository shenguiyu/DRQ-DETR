# Checkpoint Guide

## Recommended Reviewer Package

Use the following portable layout:

```text
checkpoints/
|-- sard/
|   |-- drq_detr/best_stg2.pth
|   `-- drq_detr_with_dga/best_stg2.pth
|-- seadronessee_odv2/
|   |-- drq_detr/best_stg2.pth
|   `-- drq_detr_with_dga/best_stg2.pth
`-- visdrone2019/
    |-- drq_detr/best_stg2.pth
    `-- drq_detr_with_dga/best_stg2.pth
```

For every released experiment, also retain:

```text
args.json
log.txt
eval.pth or evaluation summary
best_stg1.pth
best_stg2.pth
```

Large intermediate checkpoints do not need to be tracked by Git. Publish them
through a release asset, institutional repository, or reviewer-only archive.

## Which Checkpoint to Report

Use `best_stg2.pth`.

The training solver first searches for the best validation AP before epoch 120
and writes `best_stg1.pth`. At epoch 120, it restores that state, resets EMA,
disables strong augmentation and multi-scale sampling, and performs final
refinement. The best checkpoint from this second stage is written to
`best_stg2.pth`.

`last.pth` is intended for resuming the first stage. Historical `best.pth`
files may be copies of stage 1 and should not be used in the reviewer package.

## Weight Source

The checkpoint dictionary normally contains:

```text
model
ema
optimizer
lr_scheduler
last_epoch
```

Paper evaluation uses the EMA module when `use_ema: True`. The benchmark script
therefore selects `ema.module` by default. Its JSON output records:

- selected weight source;
- total source keys;
- loaded keys;
- missing keys;
- unexpected keys;
- skipped keys.

Any material mismatch indicates that the checkpoint and architecture config do
not belong to the same experiment.

## Validation Example

```bash
python train.py \
  -c configs/experiments/sard/drq_detr.yml \
  -r checkpoints/sard/drq_detr/best_stg2.pth \
  --test-only
```

Do not pair:

- a P2-32 checkpoint with `drq_detr_p2_64.yml`;
- a different dataset head without deliberate head conversion;
- generated sensitivity metadata with a different architecture YAML;
- `best_stg1.pth` metrics with a table labeled as the final model.
