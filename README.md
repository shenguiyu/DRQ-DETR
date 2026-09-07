# DRQ-DETR

**High-resolution detail routing and query allocation for UAV small-object detection.**

DRQ-DETR is the repository accompanying the Drones manuscript. The detector is
built on the DEIM baseline with an HGNetV2 backbone. It coordinates DSPR,
CGRF, SDQ, and a Thin-P2 high-resolution value path to organize high-resolution
information from detail construction and semantic constraint to query
allocation and decoder access.

Under the unified controlled evaluation protocol, DRQ-DETR achieves AP50/AP
values of **92.41/62.22** on SARD, **84.62/52.65** on
SeaDronesSee-ODv2, and **46.83/29.14** on VisDrone2019. Paper-specific final
evaluation results on SARD test, SeaDronesSee-ODv2 validation, and
VisDrone2019 test-dev are reported separately below.

This repository provides the implementation, configuration files, experiment
maps, and benchmark scripts used to reproduce the reported DRQ-DETR runs. The
source tree does not contain datasets or trained checkpoint files.

<p align="center">
  <img src="assets/figures/fig1_overall_architecture.png" alt="DRQ-DETR overall architecture" width="96%">
</p>

## Highlights

- **DSPR: Detail-Semantic Proxy Router.** DSPR builds a stride-4
  detail-semantic proxy from shallow P2 information, a fixed gradient response,
  and P3 semantic context.
- **CGRF: Cross-Granularity Receptive-Field Fusion.** CGRF injects the DSPR
  proxy into P3 and P4 through independent non-shared gated fusion instances.
- **SDQ: Sparse Detail Query Selection.** SDQ performs a two-stage selection
  from the DSPR proxy: detached feature-energy Top-1024 followed by
  classification-score Top-64.
- **Thin P2 High-Resolution Value Path.** Thin P2 provides a 64-channel
  high-resolution value feature for decoder access while retaining the full
  stride-4 feature grid.
- **Fixed public model.** The final model keeps 300 decoder queries in total:
  64 detail queries and 236 standard multi-scale queries.

## Validation Results — AP50 & AP

These results are obtained under the unified controlled evaluation protocol and
are presented using both AP50 and COCO AP for convenient comparison with UAV
small-object detection literature.

| Dataset | Method | AP50 | AP |
|---|---|---:|---:|
| SARD | DEIM | 89.92 | 58.45 |
| SARD | DRQ-DETR | **92.41** | **62.22** |
| SeaDronesSee-ODv2 | DEIM | 82.31 | 48.84 |
| SeaDronesSee-ODv2 | DRQ-DETR | **84.62** | **52.65** |
| VisDrone2019 | DEIM | 37.35 | 22.07 |
| VisDrone2019 | DRQ-DETR | **46.83** | **29.14** |

| Dataset | Delta AP50 vs. DEIM | Delta AP vs. DEIM |
|---|---:|---:|
| SARD | +2.49 | +3.77 |
| SeaDronesSee-ODv2 | +2.31 | +3.81 |
| VisDrone2019 | +9.48 | +7.07 |

## Final Paper Evaluation Results

This table is the GitHub counterpart of the final paper comparison table. It
uses the final dataset-specific evaluation split reported in the manuscript.

| Dataset | Split | Method | AP | AP50 | AP75 | APs | APm | APl | AR@100 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| SARD | test | YOLO11s | 61.33 | 92.88 | 69.47 | 32.08 | 64.20 | 74.80 | 67.83 |
| SARD | test | RT-DETR | 60.12 | 92.17 | 67.24 | 32.97 | 61.89 | 76.84 | 70.49 |
| SARD | test | DEIM | 57.22 | 88.75 | 64.97 | 31.84 | 59.27 | 72.88 | 72.66 |
| SARD | test | DRQ-DETR | **61.53** | 91.44 | **71.38** | **36.12** | 64.12 | 75.13 | **73.98** |
| SeaDronesSee-ODv2 | validation | YOLO11s | 40.29 | 68.73 | 40.18 | 27.78 | 44.77 | 61.58 | 46.12 |
| SeaDronesSee-ODv2 | validation | RT-DETR | 42.72 | 79.58 | 39.29 | 36.12 | 45.47 | 58.27 | 54.78 |
| SeaDronesSee-ODv2 | validation | DEIM | 48.86 | 82.41 | 49.36 | 45.84 | 50.00 | 64.63 | 63.58 |
| SeaDronesSee-ODv2 | validation | DRQ-DETR | **52.71** | **84.58** | **53.84** | **48.59** | **51.30** | **68.26** | **66.85** |
| VisDrone2019 | test-dev | YOLO11s | 15.11 | 26.50 | 15.22 | 6.70 | 23.84 | 33.29 | 24.06 |
| VisDrone2019 | test-dev | RT-DETR | 15.74 | 28.36 | 15.40 | 8.33 | 23.23 | 35.67 | 31.16 |
| VisDrone2019 | test-dev | DEIM | 17.64 | 30.54 | 17.95 | 9.94 | 25.38 | 32.93 | 33.92 |
| VisDrone2019 | test-dev | DRQ-DETR | **22.65** | **38.37** | **23.14** | **14.06** | **31.77** | **40.98** | **39.19** |

Final paper AP values are **61.53** on SARD test, **52.71** on
SeaDronesSee-ODv2 validation, and **22.65** on VisDrone2019 test-dev. Compared
with DEIM, the corresponding AP gains are **+4.31**, **+3.85**, and
**+5.01** percentage points.

## Result Protocol Note

The repository reports two complementary result groups. The Validation Results
section uses the unified controlled protocol used for ablation and
configuration studies, while the Final Paper Evaluation Results section follows
the final dataset-specific evaluation splits reported in the paper. These
values should not be mixed across protocols.

In particular, VisDrone2019 has a controlled validation result of AP50/AP =
46.83/29.14 and a final paper test-dev result of AP50/AP = 38.37/22.65.
SeaDronesSee-ODv2 controlled values, AP50/AP = 84.62/52.65, and final-paper
validation values, AP50/AP = 84.58/52.71, are both retained because they
correspond to different experimental runs or protocols. SARD controlled values,
AP50/AP = 92.41/62.22, and final-paper test values, AP50/AP = 91.44/61.53, are
also reported separately.

## Architecture

The paper name **DRQ-DETR** refers to the fixed public configuration below.

| Item | Setting |
|---|---:|
| Baseline family | DEIM |
| Backbone | HGNetV2 |
| Input size | 640 x 640 |
| Decoder hidden dimension | 256 |
| Decoder layers | 3 |
| Total decoder queries | 300 |
| SDQ first-stage candidates | 1024 |
| SDQ detail queries | 64 |
| Standard multi-scale queries | 236 |
| Thin P2 width | 64 channels |
| Decoder value levels | 4 |
| Decoder value strides | 4 / 8 / 16 / 32 |
| DSPR semantic weight | 0.20 |
| CGRF instances | independent P3 and P4 modules |

### Query Flow

```text
DSPR proxy
  -> detached feature-energy ranking
  -> Top-1024 proxy candidates
  -> shared classification scoring
  -> Top-64 detail queries

standard multi-scale branch
  -> Top-236 semantic queries

64 detail queries + 236 semantic queries = 300 decoder queries
```

### Value Flow

```text
Thin P2 value feature V2
+ P3 value feature V3
+ P4 value feature V4
+ P5 value feature V5
-> 4 decoder value levels

value strides: 4 / 8 / 16 / 32
```

The final architecture file contains decoder strides `[4, 4, 8, 16, 32]`. The
first stride-4 entry is the DSPR proxy used by SDQ. The remaining entries are
the four decoder value levels: Thin P2, P3, P4, and P5.

## Paper-to-Code Mapping

| Paper component | Implementation |
|---|---|
| Final model graph | `configs/models/drq_detr_p2_64.yml` |
| Final SARD entry | `configs/experiments/sard/drq_detr.yml` |
| Final SeaDronesSee-ODv2 entry | `configs/experiments/seadronessee_odv2/drq_detr.yml` |
| Final VisDrone2019 entry | `configs/experiments/visdrone2019/drq_detr.yml` |
| DSPR | `engine/extre_module/custom_nn/neck/DSPR.py` |
| CGRF | `engine/extre_module/custom_nn/neck/DSPR.py` |
| SDQ two-stage query selection | `engine/deim/dfine_decoder.py` |
| Thin P2 value path | `configs/models/drq_detr_p2_64.yml` |
| Experiment manifest | `scripts/experiments.json` |
| FPS benchmark script | `scripts/benchmark_fps.py` |

Implementation facts checked against the source code:

- DSPR uses P2 projection, channel-mean Sobel gradient response, P3 projection,
  nearest-neighbor P3-to-P2 upsampling, additive detail/semantic gate logits,
  sigmoid gating, residual P2 preservation, and an independent semantic branch
  with `semantic_weight=0.20`.
- The DSPR gate modulates only the local refinement branch, not the whole P2
  residual path.
- CGRF projects the deep feature and resized DSPR proxy, predicts a gate from
  their concatenation, gates the proxy branch, fuses deep and routed proxy
  features, and applies an outer residual.
- P3 and P4 CGRF are two separate YAML layers, so their parameters are not
  shared.
- SDQ masks invalid proxy candidates before Top-K selection and uses detached
  feature energy for the first-stage Top-1024 ranking.

## Installation

The reference training environment recorded for the paper uses Python 3.10,
PyTorch 2.3.0, and torchvision 0.18.0. Install a PyTorch build compatible with
the local CUDA driver before installing the remaining packages.

```bash
conda create -n drq-detr python=3.10 -y
conda activate drq-detr

# CUDA 12.1 example. Select another official PyTorch index if needed.
pip install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Optional visualization dependencies are separated from training dependencies:

```bash
pip install -r requirements-optional.txt
```

## Dataset Preparation

Prepare COCO-format annotations under `data/` using the paths expected by the
dataset YAML files.

```text
data/
|-- sard/
|   |-- images/train
|   |-- images/val
|   `-- annotations/instances_train.json
|-- seadronessee_odv2/
|   |-- images/train
|   |-- images/val
|   `-- annotations/instances_train.json
`-- visdrone2019/
    |-- train/images
    |-- val/images
    `-- annotations/instances_train.json
```

Each dataset YAML also lists the exact validation annotation file expected by
the default public config.

| Dataset | Images / instances used for training and validation | Classes | Size distribution | Final paper split |
|---|---|---:|---|---|
| SARD | 5755 images / 7424 instances; train 4041 / 5229; val 1144 / 1463; test 570 / 732 | 1 | small 14.90%, medium 70.99%, large 14.12%; 1.29 instances/image | test |
| SeaDronesSee-ODv2 | train 8930 / 57,760; validation 1547 / 9630; 10,477 evaluable images / 67,390 instances | 5 evaluated foreground categories | small 36.71%, medium 46.81%, large 16.49%; 6.43 instances/image | validation |
| VisDrone2019 | train 6471 / 343,204; validation 548 / 38,759; test-dev 1610 images | 10 | small 62.36%, medium 32.73%, large 4.91%; 52.97 instances/image | test-dev |

SeaDronesSee-ODv2 keeps category id 0 as an ignored category in the converted
annotation file. The model head therefore uses six category slots while the
reported evaluation covers five foreground categories: swimmer, boat, jetski,
life-saving appliance, and buoy.

For final-paper SARD test evaluation, use the test annotation/image split used
in the manuscript rather than reporting the default validation config as a test
result.

## Training

Paper training protocol:

| Item | Setting |
|---|---:|
| Training GPU | NVIDIA GeForce RTX 4090 |
| Python | 3.10 |
| PyTorch / torchvision | 2.3.0 / 0.18.0 |
| Input baseline | 640 x 640 |
| Epochs | 132 |
| First stage | approximately 480-800 multi-scale for the first 120 epochs |
| Final stage | fixed 640 for the last 12 epochs |
| Total batch size | 12 |
| Optimizer | AdamW |
| AMP | enabled |
| SyncBN | enabled |
| EMA | enabled |
| External backbone pretraining | none |
| Critical runs / ablations | seed=0 |
| Final evaluation | 640 x 640, pycocotools COCOeval, bbox, maxDets=100 |

Final model commands:

```bash
# SARD
python train.py -c configs/experiments/sard/drq_detr.yml --seed 0

# SeaDronesSee-ODv2
python train.py -c configs/experiments/seadronessee_odv2/drq_detr.yml --seed 0

# VisDrone2019
python train.py -c configs/experiments/visdrone2019/drq_detr.yml --seed 0
```

The implementation config records the AdamW learning rates and loss weights in
the corresponding YAML files; these values are not repeated here as separate
paper-level claims.

## Evaluation

After training, evaluate the generated checkpoint with the matching config:

```bash
python train.py \
  -c configs/experiments/seadronessee_odv2/drq_detr.yml \
  -r checkpoints/seadronessee_odv2/drq_detr/best_stg2.pth \
  --test-only
```

`-r/--resume`, `--test-only`, `--device`, `--seed`, and `--output-dir` are
implemented by `train.py`. The helper below lists the public experiment names:

```bash
python scripts/run_experiment.py --list
python scripts/run_experiment.py \
  --dataset visdrone2019 \
  --experiment final_p2_64 \
  --seed 0 \
  --dry-run
```

## Component Ablation

The component ablation follows the P1024-Q96 setting reported in the paper.

| Dataset | Configuration | AP | AP50 | AP75 | APs | AR@100 |
|---|---|---:|---:|---:|---:|---:|
| SARD | DEIM | 58.45 | 89.92 | 65.90 | 28.72 | 73.18 |
| SARD | +DSPR+SDQ | 52.76 | 86.05 | 58.59 | 27.23 | 67.56 |
| SARD | +DSPR+SDQ+CGRF | 60.56 | 91.15 | 69.06 | 30.64 | 73.68 |
| SARD | +DSPR+SDQ+CGRF+Thin P2 | **62.03** | **91.55** | **72.15** | **33.03** | **74.01** |
| SeaDronesSee-ODv2 | DEIM | 48.84 | 82.31 | 49.24 | 46.85 | 63.53 |
| SeaDronesSee-ODv2 | +DSPR+SDQ | 51.50 | 83.69 | 52.80 | 44.79 | 66.35 |
| SeaDronesSee-ODv2 | +DSPR+SDQ+CGRF | 50.99 | 82.55 | 52.19 | 46.42 | 64.76 |
| SeaDronesSee-ODv2 | +Full | **53.65** | **85.83** | **55.48** | **49.85** | **68.65** |
| VisDrone2019 | DEIM | 22.07 | 37.35 | 22.14 | 15.15 | 37.66 |
| VisDrone2019 | +DSPR+SDQ | 23.12 | 38.57 | 23.40 | 15.97 | 38.03 |
| VisDrone2019 | +DSPR+SDQ+CGRF | 23.09 | 38.54 | 23.47 | 15.96 | 37.99 |
| VisDrone2019 | +Full | **29.05** | **46.72** | **30.19** | **21.47** | **44.33** |

## Thin-P2 Width

The final unified configuration uses Thin P2 width 64. This setting is selected
as a cross-dataset configuration; it should not be read as an absolute optimum
for every individual dataset.

| Dataset | Configuration | AP50 | AP | APs | Params (M) | GFLOPs |
|---|---|---:|---:|---:|---:|---:|
| SARD | DEIM | 89.92 | 58.45 | 28.72 | 10.220 | 24.82 |
| SARD | Thin P2-32 | 87.18 | 53.60 | 25.23 | 11.914 | 50.64 |
| SARD | Thin P2-64 | **92.41** | **62.22** | **33.29** | 11.997 | 54.07 |
| SeaDronesSee-ODv2 | DEIM | 82.31 | 48.84 | 46.85 | 10.227 | 24.84 |
| SeaDronesSee-ODv2 | Thin P2-32 | **84.70** | **52.80** | **49.40** | 11.921 | 50.64 |
| SeaDronesSee-ODv2 | Thin P2-64 | 84.62 | 52.65 | 48.42 | 12.003 | 54.16 |
| VisDrone2019 | DEIM | 37.35 | 22.07 | 15.15 | 10.232 | 24.86 |
| VisDrone2019 | Thin P2-32 | 43.90 | 27.10 | 19.80 | 11.926 | 50.64 |
| VisDrone2019 | Thin P2-64 | **46.83** | **29.14** | **21.33** | 12.008 | 54.23 |

## SDQ Sensitivity

The paper denotes these settings as `pre_top` and `query_top`; the
implementation exposes the corresponding Top-K parameters as `pre_topk` and
`query_topk`.

| Dataset | Thin P2 | pre_top / pre_topk | query_top / query_topk | AP50 | AP | AP75 | APs |
|---|---|---:|---:|---:|---:|---:|---:|
| SARD | False | 1024 | 96 | 91.15 | 60.56 | 69.06 | 30.64 |
| SARD | False | 1536 | 96 | 92.90 | 61.47 | 70.55 | 32.75 |
| SARD | True | 1024 | 64 | 92.41 | **62.22** | 70.89 | **33.29** |
| SARD | True | 1024 | 96 | 91.55 | 62.03 | **72.15** | 33.03 |
| SARD | True | 1536 | 96 | 92.50 | 61.77 | 71.28 | 32.85 |
| SeaDronesSee-ODv2 | False | 1024 | 96 | 82.55 | 50.99 | 52.19 | 46.42 |
| SeaDronesSee-ODv2 | False | 1536 | 96 | 83.10 | 50.40 | 51.50 | 45.60 |
| SeaDronesSee-ODv2 | True | 1024 | 64 | 84.62 | 52.65 | 53.97 | 48.42 |
| SeaDronesSee-ODv2 | True | 1024 | 96 | **85.83** | **53.65** | **55.48** | **49.85** |
| SeaDronesSee-ODv2 | True | 1536 | 96 | 83.82 | 52.39 | 53.45 | 46.39 |
| VisDrone2019 | False | 1024 | 96 | 38.54 | 23.09 | 23.47 | 15.96 |
| VisDrone2019 | False | 1536 | 96 | 41.07 | 24.76 | 25.34 | 17.45 |
| VisDrone2019 | True | 1024 | 64 | **46.83** | **29.14** | **30.52** | 21.33 |
| VisDrone2019 | True | 1024 | 96 | 46.72 | 29.05 | 30.19 | **21.47** |
| VisDrone2019 | True | 1536 | 96 | 44.59 | 27.61 | 28.63 | 20.28 |

## P2 Access Strategy

The controlled P2 access study is reported with the formal paper names below.
The validation AP50 progression is 37.35 -> 42.13 -> 43.59 -> 46.83.

| Method | AP50 | AP | AP75 | APs | APm | APl | Params (M) | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEIM | 37.35 | 22.07 | 22.14 | 15.15 | 30.65 | 34.91 | 10.232 | 24.86 |
| + Thin P2 | 42.13 | 25.86 | 26.68 | 18.35 | 34.96 | 39.49 | 11.093 | 42.96 |
| + P2 | 43.59 | 26.87 | 27.62 | 19.90 | 35.42 | 40.59 | 11.691 | 72.89 |
| Full | **46.83** | **29.14** | **30.52** | **21.33** | **39.14** | **43.28** | 12.008 | 54.23 |

Configuration files for these controlled runs are stored under
`configs/experiments/*/causal_p2/`. Their architecture graphs are stored under
`configs/models/causal_p2/`.

## Paper Efficiency

The paper efficiency table uses VisDrone2019 controlled validation accuracy and
forward-only timing.

| Item | Setting |
|---|---:|
| Hardware | NVIDIA GeForce RTX 4060 Ti |
| Input | 640 x 640 |
| Batch size | 1 |
| Precision | FP32 |
| Warmup | 30 iterations |
| Timed iterations | 100 |
| Timing scope | model forward only |
| Excluded | preprocessing, NMS, and other post-processing |

| Method | AP50 | AP | APs | Params (M) | GFLOPs | Latency (ms) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEIM | 37.35 | 22.07 | 15.15 | 10.232 | 24.86 | 17.30 | 57.81 |
| DRQ-DETR | **46.83** | **29.14** | **21.33** | 12.008 | 54.23 | 35.58 | 28.10 |

The benchmark utility can also be used for local runtime checks:

```bash
python scripts/benchmark_fps.py \
  --manifest scripts/fps_benchmark_manifest.json \
  --device cuda:0 \
  --imgsz 640 \
  --batch-size 1 \
  --warmup 30 \
  --iters 100 \
  --precision fp32 \
  --no-postprocess
```

## Additional Repository Runtime Benchmark

The benchmark script is intentionally configurable so that users can measure
runtime under different GPUs, precision modes, checkpoints, and post-processing
settings. Such local benchmark outputs are separate from the paper's Table 8
timing protocol and should be reported with their own hardware and scope.

## Checkpoints

Checkpoint files are not distributed through ordinary Git. See
[`checkpoints/README.md`](checkpoints/README.md) and
[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md) for the expected directory layout
and evaluation commands.

Author-provided training results and checkpoints for reviewer verification are
shared outside Git at:

```text
Baidu Netdisk: https://pan.baidu.com/s/1ZkrMin5tb2IFsejybHesAQ?pwd=0871
Extraction code: 0871
Archive label: 训练结果
```

Do not evaluate a checkpoint with a mismatched architecture file. For example,
a Thin P2-32 or Q96 checkpoint should not be paired with the final P2-64 Q64
configuration.

## Configuration Map

The main experiment-level YAML files are:

```text
configs/experiments/sard/drq_detr.yml
configs/experiments/seadronessee_odv2/drq_detr.yml
configs/experiments/visdrone2019/drq_detr.yml
```

All three point to:

```text
configs/models/drq_detr_p2_64.yml
```

The public configuration guide is available at
[`configs/README.md`](configs/README.md), and the paper-alignment summary is
available at [`docs/PAPER_ALIGNMENT.md`](docs/PAPER_ALIGNMENT.md).

## Repository Layout

```text
DRQ-DETR/
|-- assets/figures/                 README paper figures
|-- checkpoints/                    checkpoint placeholder and instructions
|-- configs/
|   |-- base/                       shared runtime and optimizer config
|   |-- datasets/                   COCO-format dataset definitions
|   |-- experiments/                final, ablation, sensitivity, and P2 controls
|   `-- models/                     network graph definitions
|-- docs/                           reviewer-facing documentation
|-- engine/                         model, loss, data, and solver code
|-- scripts/                        config checks and benchmark utilities
|-- train.py
|-- requirements.txt
`-- LICENSE
```

## Limitations

Similar-category confusion remains challenging in low-resolution aerial images,
especially for visually close categories and strongly occluded instances. Thin
P2 also retains the full stride-4 spatial grid, so the accuracy gain comes with
additional computation. In the paper efficiency setting, GFLOPs increase from
24.86 to 54.23 and forward latency increases from 17.30 ms to 35.58 ms. Future
work will study explicit spatial pruning for high-resolution computation and
dynamic detail-query allocation.

## License

This repository follows the license terms in [`LICENSE`](LICENSE). Third-party
credits and inherited notices are retained in [`NOTICE`](NOTICE).

## Acknowledgements

The codebase is derived from the DEIM / D-FINE / RT-DETR family of open-source
detectors. Please also follow the license requirements of the upstream projects
when redistributing or modifying this repository.

## Citation

If you find DRQ-DETR useful in your research, please cite this repository:

```bibtex
@software{drqdetr2026,
  author  = {{shenguiyu}},
  title   = {DRQ-DETR: Detail-Routed Query Transformer for Drone-View Small Object Detection},
  year    = {2026},
  url     = {https://github.com/shenguiyu/DRQ-DETR}
}
