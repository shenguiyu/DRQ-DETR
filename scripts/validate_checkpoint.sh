#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/experiments/visdrone2019/drq_detr.yml}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to best_stg2.pth or last.pth}
VAL_IMAGES=${VAL_IMAGES:?Set VAL_IMAGES to validation image directory}
VAL_JSON=${VAL_JSON:?Set VAL_JSON to COCO validation annotation JSON}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/validation/visdrone2019}

python train.py \
  -c "$CONFIG" \
  -r "$CHECKPOINT" \
  --test-only \
  -u val_dataloader.dataset.img_folder="$VAL_IMAGES" \
     val_dataloader.dataset.ann_file="$VAL_JSON" \
     output_dir="$OUTPUT_DIR"
