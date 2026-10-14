#!/usr/bin/env python3
"""Frozen paper evaluation adapter; never trains, cleans boxes or filters scores.

infer delegates to the repository's native test-only/deploy/EMA entry point.
score independently evaluates the saved pred.json with COCO bbox AP.
Native forward results and historical eager-export results must not be mixed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def score(annotation, prediction, output):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(str(annotation))
    # Some converted annotations omit COCO optional fields. Match the native
    # non-crowd default in memory, without changing the annotation on disk.
    for ann in gt.dataset['annotations']:
        ann.setdefault('iscrowd', 0)
        ann.setdefault('area', ann['bbox'][2] * ann['bbox'][3])
    rows = json.loads(Path(prediction).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty native COCO detection list.')
    ids = set(gt.getImgIds())
    categories = set(gt.getCatIds())
    if any(r['image_id'] not in ids or r['category_id'] not in categories for r in rows):
        raise ValueError('Unknown image/category id: check the frozen annotation and mapping.')
    ev = COCOeval(gt, gt.loadRes(rows), 'bbox')
    ev.params.imgIds = sorted(ids)
    ev.params.catIds = sorted(c['id'] for c in gt.dataset['categories']
                             if str(c.get('name', '')).lower() not in
                             {'ignored regions', 'ignored', 'ignore', 'others', 'other'})
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    result = {
        'protocol': 'native-deploy-EMA-FP32-640; COCO bbox, maxDets=1/10/100',
        'annotation': str(Path(annotation).resolve()), 'annotation_sha256': sha(annotation),
        'prediction': str(Path(prediction).resolve()), 'prediction_sha256': sha(prediction),
        'pycocotools': importlib.metadata.version('pycocotools'),
        'python': platform.python_version(), 'image_count': len(ids),
        'raw': dict(zip(KEYS, map(float, ev.stats))),
        'percent': dict(zip(KEYS, [float(x) * 100 for x in ev.stats])),
        'rounding': 'round full-precision percent to two decimals; never pad console precision',
        'no_extra_score_threshold': True, 'no_box_clipping': True,
        'category_ids': list(map(int, ev.params.catIds)),
        'annotation_defaults_in_memory': {'iscrowd': 0, 'area_if_missing': 'bbox width * height'},
    }
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('score')
    a.add_argument('--annotation', required=True)
    a.add_argument('--prediction', required=True)
    a.add_argument('--output', required=True)
    a = sub.add_parser('infer')
    a.add_argument('--config', required=True)
    a.add_argument('--checkpoint', required=True)
    a.add_argument('--annotation', required=True)
    a.add_argument('--images', required=True)
    a.add_argument('--output-dir', required=True)
    a.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.mode == 'score':
        score(args.annotation, args.prediction, args.output)
        return
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new empty output directory; frozen results are not overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    for path in (args.config, args.checkpoint, args.annotation, args.images):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    cmd = [sys.executable, str(ROOT / 'train.py'), '-c', str(Path(args.config).resolve()),
           '-r', str(Path(args.checkpoint).resolve()), '--test-only', '--seed', '0',
           '--device', args.device, '--no-use-amp', '--output-dir', str(out), '-u',
           'eval_spatial_size=[640,640]',
           'val_dataloader.dataset.img_folder=' + json.dumps(str(Path(args.images).resolve()).replace('\\', '/')),
           'val_dataloader.dataset.ann_file=' + json.dumps(str(Path(args.annotation).resolve()).replace('\\', '/'))]
    evidence = {'command': cmd, 'config_sha256': sha(args.config),
                'checkpoint_sha256': sha(args.checkpoint), 'annotation_sha256': sha(args.annotation),
                'entry_sha256': sha(__file__), 'seed': 0, 'input_size': [640, 640],
                'native_entry': 'train.py --test-only / DetSolver.val() / EMA / deploy()',
                'python': sys.version, 'platform': platform.platform()}
    (out / 'invocation.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    with (out / 'native_eval.log').open('w', encoding='utf-8') as stream:
        run = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if run.returncode:
        raise RuntimeError('Native evaluation failed; see native_eval.log. No metric is accepted.')
    score(args.annotation, out / 'pred.json', out / 'official_metrics.json')


if __name__ == '__main__':
    main()
