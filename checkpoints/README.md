# Checkpoint Directory

This directory is a placeholder for locally generated or separately shared
checkpoint files. Checkpoints are intentionally not distributed through ordinary
Git because they are large binary artifacts.

## Expected Layout

```text
checkpoints/
|-- sard/
|   `-- drq_detr/
|       `-- best_stg2.pth
|-- seadronessee_odv2/
|   `-- drq_detr/
|       `-- best_stg2.pth
`-- visdrone2019/
    `-- drq_detr/
        `-- best_stg2.pth
```

## Evaluation Example

After training, evaluate the generated checkpoint with the matching config:

```bash
python train.py \
  -c configs/experiments/visdrone2019/drq_detr.yml \
  -r checkpoints/visdrone2019/drq_detr/best_stg2.pth \
  --test-only
```

Do not pair a checkpoint with a different architecture setting. In particular,
P2 width, SDQ `pre_topk`, SDQ `query_topk`, number of classes, and dataset
category mapping must match the YAML used during training.

## External Reviewer Archive

Author-provided training results and checkpoints are shared outside Git:

```text
Baidu Netdisk: https://pan.baidu.com/s/1ZkrMin5tb2IFsejybHesAQ?pwd=0871
Extraction code: 0871
Archive label: 训练结果
```

Place downloaded checkpoint files under the matching subdirectory shown above
before running evaluation commands.

If official weights are provided in the future, they should be distributed as
GitHub Release assets or through another explicit storage link. No checkpoint
download URL is implied by this placeholder file.
