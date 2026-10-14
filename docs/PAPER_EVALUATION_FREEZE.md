# Frozen paper evaluation protocol — 14 October 2026

The sole official DEIM/DRQ-DETR paper evaluation entry is
`python scripts/paper_evaluate.py infer`. It delegates to `train.py --test-only`,
`DetSolver.val()`, the EMA checkpoint and `deploy()`, then scores the saved
`pred.json` using COCO bbox AP with maxDets 1/10/100. It does not train or modify
model implementation. Do not silently substitute a historical eager export,
add a score cutoff, clip boxes, remap classes, or select a different checkpoint.

## Commands

Run from the repository root with the project's installed dependencies. Replace
the paths with the corresponding downloaded files. The split is a required,
explicit annotation/image choice; no test result is relabeled as validation.

```bash
python scripts/paper_evaluate.py infer --config configs/experiments/sard/drq_detr.yml --checkpoint /weights/sard/drq_detr/best_stg2.pth --annotation /data/sard/annotations/instances_val.json --images /data/sard/images/val --output-dir outputs/frozen/sard_drq_validation
python scripts/paper_evaluate.py infer --config configs/experiments/seadronessee_odv2/drq_detr.yml --checkpoint /weights/seadronessee_odv2/drq_detr/best_stg2.pth --annotation /data/seadronessee_odv2/annotations/instances_val.json --images /data/seadronessee_odv2/images/val --output-dir outputs/frozen/sea_drq_validation
python scripts/paper_evaluate.py infer --config configs/experiments/visdrone2019/drq_detr.yml --checkpoint /weights/visdrone2019/drq_detr/best_stg2.pth --annotation /data/visdrone2019/annotations/instances_val.json --images /data/visdrone2019/val/images --output-dir outputs/frozen/vis_drq_validation
```

For DEIM, substitute `baseline_deim_s.yml` and the matching baseline checkpoint.
For Table 3, SARD uses `instances_test.json`/test images and VisDrone2019 uses
the annotated test-dev JSON/images. SeaDronesSee-ODv2 remains validation.
Validation and final-split metrics are listed separately in
`docs/paper_evidence/official_results.json` together with exact hashes.

To independently check a saved native prediction without loading PyTorch:

```bash
python scripts/paper_evaluate.py score --annotation /data/annotations/instances_val.json --prediction outputs/frozen/sea_drq_validation/pred.json --output outputs/frozen/sea_drq_validation/independent_metrics.json
```

The score command fills omitted `iscrowd=0` and bbox area only in memory;
ignored category names are excluded from foreground evaluation. Annotation
files are never rewritten. No detections are dropped by confidence.

## Configuration and historical experiments

- Official DRQ-DETR: Thin-P2-64, candidate pool 1024, detail queries 64, total
  queries 300. Table 4 is a staged system ablation at Q96, not the official Q64
  configuration and not a separate causal estimate of DSPR and SDQ.
- Historical Thin-P2-32 checkpoints use Q64. The public architecture
  `configs/models/drq_detr_thinp2_32.yml` is corrected to Q64 accordingly.
- Historical ablation/sensitivity rows are retained from full-precision
  training-validation logs, which use an unfused validation graph. They are
  tagged as historical evidence, not claimed to have been rerun with the frozen
  deploy entry. Future reevaluation must use the entry above and preserve outputs.
- Thin-P2-32 archived deployment profiles: SARD 50.2563, SeaDronesSee-ODv2
  50.3468, VisDrone2019 50.4191 GFLOPs at 640 x 640. These are audited archived
  measurements, not new profiling runs. Do not report a common 50.64 GFLOPs.

## FPS scope and remaining evidence gap

The original Table 8 figures are forward-only on RTX 4060 Ti, FP32, batch 1,
640 x 640, 30 warm-ups, 100 measured iterations, on the unfused eval graph.
They must not be compared with the native evaluator's batched
inference-plus-postprocess FPS or described as validated onboard real-time speed.
The original per-iteration timing samples were not preserved: do not manufacture
them or attach new benchmark metadata to old measurements.

`scripts/benchmark_fps.py --no-postprocess` now retains timing samples,
standard deviation and config/checkpoint hashes for future reruns. Such reruns
are separate measurements, not replacements for missing historical samples.

GitHub hosts code and small evidence files without a Baidu account requirement.
Large checkpoint files remain in the existing Baidu mirror. An internationally
accessible large-file mirror is still needed; no uncreated Zenodo/Hugging Face
deposit is claimed.
