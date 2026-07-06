# Dataset Preparation

DRQ-DETR expects COCO detection JSON files with `images`, `annotations`, and
`categories` arrays. Bounding boxes must use COCO `xywh` format in absolute
pixels, and every annotation must reference an existing image and category.

## Expected Layout

```text
data/
|-- sard/
|   |-- images/train/
|   |-- images/val/
|   `-- annotations/
|       |-- instances_train.json
|       `-- instances_val.json
|-- seadronessee_odv2/
|   |-- images/train/
|   |-- images/val/
|   `-- annotations/
|       |-- instances_train.json
|       `-- instances_val.json
`-- visdrone2019/
    |-- train/images/
    |-- val/images/
    `-- annotations/
        |-- instances_train.json
        `-- instances_val.json
```

## Category Settings

### SARD

- Config: `configs/datasets/sard.yml`
- Model output slots: 1
- Remapping: disabled
- Use the same foreground category id in train and validation JSON files.

### SeaDronesSee-ODv2

- Config: `configs/datasets/seadronessee_odv2.yml`
- Model output slots: 6
- Remapping: disabled
- Category id 0 is retained as an ignored slot.
- Five foreground categories are evaluated: swimmer, boat, jetski,
  life-saving appliance, and buoy.

Do not collapse this config to five slots unless both annotations and all
checkpoints are converted consistently.

### VisDrone2019

- Config: `configs/datasets/visdrone2019.yml`
- Model output slots: 10
- COCO-style category remapping: enabled
- Train and validation files must use the same ten foreground definitions.

## Annotation Audit

Before training, verify:

1. Image ids and annotation ids are unique integers.
2. Every `annotation.image_id` exists in `images`.
3. Every `annotation.category_id` exists in `categories`.
4. Every bounding box has positive width and height.
5. Bounding boxes remain inside image bounds after conversion.
6. `area` is present and consistent with the bounding box.
7. `iscrowd` is present, normally set to 0 for these experiments.
8. Train and validation category ids have identical semantics.
9. Ignored classes are handled consistently across JSON and YAML.

## Alternative Data Locations

Paths can be overridden without editing tracked files:

```bash
python train.py \
  -c configs/experiments/seadronessee_odv2/drq_detr.yml \
  --seed 0 \
  -u train_dataloader.dataset.img_folder=/datasets/SeaDronesSee/train \
     train_dataloader.dataset.ann_file=/datasets/SeaDronesSee/train.json \
     val_dataloader.dataset.img_folder=/datasets/SeaDronesSee/val \
     val_dataloader.dataset.ann_file=/datasets/SeaDronesSee/val.json
```

Dataset licenses and redistribution terms remain governed by their original
providers. Images and annotations are intentionally excluded from this source
repository.
