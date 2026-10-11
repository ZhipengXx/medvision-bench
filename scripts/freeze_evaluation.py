"""Freeze checkpoint/configuration/split hashes before the first formal test."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.runtime import *

ap = argparse.ArgumentParser()
ap.add_argument('--experiments', required=True)
ap.add_argument('--output', default='splits/evaluation_manifest.json', help='New manifest path for newly trained runs')
args = ap.parse_args()
target = ROOT / args.output
if target.exists():
    raise SystemExit('Manifest exists; preserve the original test freeze')
runs = []
for path in load_config(args.experiments)['experiments']:
    cfg = load_config(ROOT / path)
    rid = cfg['run_id']
    out = ROOT / 'outputs' / rid
    meta = json.loads((out / 'metadata.json').read_text())
    if not meta['formal'] or not meta['complete']:
        raise ValueError(f'Not a completed formal run: {rid}')
    split = 'splits/isic_split.json' if rid.startswith('cls_') else 'splits/la_split.json'
    if sha256(ROOT / split) != meta['split_sha256']:
        raise ValueError('Split changed after training')
    runs.append(dict(run_id=rid, checkpoint=f'outputs/{rid}/best.pt', checkpoint_sha256=sha256(out / 'best.pt'),
                     config=meta['config'], split=split, split_sha256=meta['split_sha256'],
                     data_config_sha256=sha256(ROOT / cfg['data_config']), model_selection=meta['selection']))
if len(runs) != 5:
    raise ValueError('Five formal experiments required')
save_json(target, dict(frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), runs=runs,
          classification_classes=['MEL','NV','BCC','AKIEC','BKL','DF','VASC'],
          classification_rule='argmax, no TTA', segmentation_rule='argmax; Gaussian probabilities; no postprocessing',
          segmentation_metrics='full original grid; case-equal mean; MedPy assd/hd95 connectivity=1; empty failures retained'))
progress('All five formal checkpoints frozen before test')
