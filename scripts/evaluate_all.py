"""Evaluate frozen best models on complete official test splits, saving predictions."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.common.runtime import *
from src.classification.data import ISICDataset, CLASSES
from src.classification.model import build_model
from src.classification.metrics import metrics as cls_metrics
from src.segmentation.model import UNet
from src.segmentation.data import read_volume
from src.segmentation.inference import predict_volume
from src.segmentation.metrics import metrics as seg_metrics


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def safe_numbers(obj):
    if isinstance(obj, dict):
        return {k: safe_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_numbers(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return 'Infinity' if obj > 0 else '-Infinity' if obj < 0 else 'undefined'
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output-root', default='results')
    ap.add_argument('--run-id')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    destination = ROOT / args.output_root
    device = torch.device(args.device)
    cls_summary, seg_summary = [], []
    for run in manifest['runs']:
        rid, cfg = run['run_id'], run['config']
        if args.run_id and rid != args.run_id:
            continue
        seed_all(cfg['seed'])
        for p, h in [(run['checkpoint'],run['checkpoint_sha256']), (run['split'],run['split_sha256']),
                     (cfg['data_config'],run['data_config_sha256'])]:
            if sha256(ROOT / p) != h:
                raise ValueError(f'Frozen input modified: {p}')
        split = json.loads((ROOT / run['split']).read_text())
        data = load_config(ROOT / cfg['data_config'])
        state = torch.load(ROOT / run['checkpoint'], map_location='cpu', weights_only=False)
        is_cls = rid.startswith('cls_')
        model = build_model(False) if is_cls else UNet(cfg['dimensions'], cfg['base_channels'])
        model.load_state_dict(state['model'])
        model.to(device).eval()
        folder = destination / ('classification' if is_cls else 'segmentation') / rid
        folder.mkdir(parents=True, exist_ok=True)
        if is_cls:
            ds = ISICDataset(split['test'], ROOT / data['classification']['root'])
            predictions, y, probs = [], [], []
            with torch.inference_mode():
                for x, target, ids in DataLoader(ds, batch_size=cfg['batch_size'], num_workers=cfg['workers']):
                    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                        p = model(x.to(device)).float().softmax(1).cpu().numpy()
                    for image_id, gt, pr in zip(ids, target.tolist(), p):
                        predictions.append(dict(image_id=image_id, gt=gt, pred=int(pr.argmax()),
                                                **{f'p_{c}':float(pr[i]) for i,c in enumerate(CLASSES)}))
                    y.extend(target.tolist())
                    probs.extend(p)
            m, per_class = cls_metrics(y, probs)
            m.update(run_id=rid, checkpoint_sha256=run['checkpoint_sha256'])
            csv_write(folder / 'predictions.csv', predictions)
            csv_write(folder / 'per_class.csv', per_class)
            save_json(folder / 'metrics.json', m)
            cls_summary.append(dict(run_id=rid, **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1',
                                         'weighted_precision','weighted_recall','weighted_f1','macro_auroc']}))
            progress(f"FORMAL TEST {rid}: accuracy={m['accuracy']:.4f}, macro F1={m['macro_f1']:.4f}, macro AUROC={m['macro_auroc']:.4f}")
        else:
            data_cfg = data['segmentation']
            rows = []
            for row in split['test']:
                im, gt = read_volume(ROOT / data_cfg['root'] / row['path'], data_cfg)
                begin = time.monotonic()
                pred = predict_volume(model, im, cfg, device)
                secs = time.monotonic() - begin
                np.savez_compressed(folder / (row['case_id'] + '.npz'), prediction=pred,
                                    axis_order='ZYX', source_sha256=row['sha256'])
                m = seg_metrics(pred, gt, data_cfg['spacing_zyx'])
                rows.append(dict(case_id=row['case_id'], **m, distance_unit=data_cfg['distance_unit'],
                                 spacing_source=data_cfg['spacing_source'], inference_seconds=secs,
                                 predicted_voxels=int(pred.sum()), gt_voxels=int(gt.sum()),
                                 checkpoint_sha256=run['checkpoint_sha256']))
                print(rid, row['case_id'], m, flush=True)
            csv_write(folder / 'per_case_metrics.csv', rows)
            summary = dict(run_id=rid, n=len(rows), distance_unit=data_cfg['distance_unit'],
                           empty_failure_count=sum(r['empty']=='one' for r in rows))
            for key in ['dice','jaccard','asd','hd95']:
                values = np.array([r[key] for r in rows])
                summary[key+'_mean'] = float(np.mean(values))
                summary[key+'_std'] = float(np.std(values)) if np.isfinite(values).all() else float('inf')
                summary[key+'_finite_n'] = int(np.isfinite(values).sum())
                summary[key+'_finite_mean'] = float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float('inf')
            summary['inference_seconds_mean'] = float(np.mean([r['inference_seconds'] for r in rows]))
            save_json(folder / 'metrics.json', safe_numbers(summary))
            seg_summary.append(summary)
            progress(f"FORMAL TEST {rid}: mean full-volume Dice={summary['dice_mean']:.4f}, HD95={summary['hd95_mean']:.4f} {summary['distance_unit']}")
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if cls_summary:
        csv_write(destination / 'classification/summary.csv', cls_summary)
    if seg_summary:
        csv_write(destination / 'segmentation/summary.csv', seg_summary)
    save_json(destination / 'evaluation_provenance.json', dict(manifest_sha256=sha256(args.manifest),
              finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'), output_root=args.output_root))


if __name__ == '__main__':
    main()
