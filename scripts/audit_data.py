"""Audit official splits before training; refuse ambiguous HDF5 geometry."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from PIL import Image
from src.common.runtime import ROOT, load_config, save_json, sha256, progress
from src.classification.data import CLASSES


def classification(cfg):
    root = ROOT / cfg['root']
    split, counts, hashes, image_ids = {}, [], {}, {}
    for name, expected in [('train', 10015), ('val', 193), ('test', 1512)]:
        path = root / cfg['splits'][name]['labels']
        rows = list(csv.DictReader(path.open()))
        if len(rows) != expected:
            raise ValueError(f'{name}: {len(rows)} rows, expected official {expected}')
        entries = []
        for row in rows:
            image_id = row['image']
            values = np.array([float(row[c]) for c in CLASSES])
            if not np.isin(values, [0, 1]).all() or values.sum() != 1:
                raise ValueError(f'Not a unique one-hot label: {image_id}')
            image_path = root / cfg['splits'][name]['images'] / (image_id + '.jpg')
            with Image.open(image_path) as im:
                im.convert('RGB').load()  # Decode all pixels, not just JPEG headers.
            label = int(values.argmax())
            h = sha256(image_path)
            hashes.setdefault(h, []).append([name, image_id])
            image_ids.setdefault(image_id, []).append(name)
            entries.append(dict(image_id=image_id, label=label, path=str(image_path.relative_to(root)), sha256=h))
        if len(set(x['image_id'] for x in entries)) != len(entries):
            raise ValueError(f'Duplicate IDs within {name}')
        split[name] = entries
        for i, c in enumerate(CLASSES):
            n = sum(x['label'] == i for x in entries)
            counts.append(dict(split=name, class_name=c, count=n))
            if name == 'train' and not n:
                raise ValueError('Missing training class')
    duplicates = [v for v in hashes.values() if len(v) > 1]
    cross = [v for v in duplicates if len(set(x[0] for x in v)) > 1]
    metadata_path = root / 'ISIC2018_Task3_Training_LesionGroupings.csv'
    lesion_audit = {'status': 'No lesion metadata available'}
    if metadata_path.exists():
        lesion_rows = list(csv.DictReader(metadata_path.open()))
        mapping = {r['image']: r['lesion_id'] for r in lesion_rows}
        known = {s: [mapping[r['image_id']] for r in split[s] if r['image_id'] in mapping] for s in split}
        lesion_audit = dict(metadata_rows=len(lesion_rows), unique_train_lesions=len(set(known['train'])),
                            mapped_images={s: len(v) for s, v in known.items()},
                            overlaps_with_known_ids={f'{a}_{b}': sorted(set(known[a]) & set(known[b]))
                                for a, b in [('train','val'),('train','test'),('val','test')]},
                            limitation='Training grouping file has no val/test lesion identities; full cross-split lesion independence cannot be checked')
    save_json(ROOT / 'splits/isic_split.json', split)
    save_json(ROOT / 'splits/class_to_idx.json', {c: i for i, c in enumerate(CLASSES)})
    with (ROOT / 'results/class_counts.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'class_name', 'count'])
        w.writeheader()
        w.writerows(counts)
    return dict(counts={k: len(v) for k, v in split.items()}, duplicate_images=duplicates,
                cross_split_duplicate_images=cross,
                cross_split_duplicate_ids={k: v for k, v in image_ids.items() if len(v) > 1},
                lesion_overlap_check=lesion_audit,
                official_split_preserved=True, missing_or_corrupt_images=0)


def segmentation(cfg):
    root = ROOT / cfg['root']
    paths = {name: sorted(root.glob(cfg['patterns'][name])) for name in ['train', 'test']}
    if not all(paths.values()):
        raise FileNotFoundError('Left atrial train/test HDF5 files missing')
    cases, all_ids = {}, set()
    geometry_records = []
    for name, files in paths.items():
        cases[name] = []
        for path in files:
            case_id = path.stem
            if case_id in all_ids:
                raise ValueError(f'Duplicate case or split overlap: {case_id}')
            all_ids.add(case_id)
            with h5py.File(path, 'r') as f:
                record = dict(case_id=case_id, split=name, path=str(path.relative_to(root)),
                              sha256=sha256(path), attrs={k: np.asarray(v).tolist() for k, v in f.attrs.items()}, datasets={})
                def inspect(key, obj):
                    if isinstance(obj, h5py.Dataset):
                        record['datasets'][key] = dict(shape=list(obj.shape), dtype=str(obj.dtype),
                                                      attrs={k: np.asarray(v).tolist() for k, v in obj.attrs.items()})
                f.visititems(inspect)
                for key in [cfg['image_key'], cfg['label_key']]:
                    if key not in f:
                        raise ValueError(f'{path}: missing key {key}')
                im, label = f[cfg['image_key']][:], f[cfg['label_key']][:]
                if im.shape != label.shape or not np.isfinite(im).all():
                    raise ValueError(f'{path}: inconsistent image/label')
                values = np.unique(label).tolist()
                record['label_values'] = values
                record['foreground_voxels'] = int((label == 1).sum())
                if not np.isin(values, [0, 1]).all():
                    raise ValueError('Label semantics require clarification')
            geometry_records.append(record)
            cases[name].append(dict(case_id=case_id, path=str(path.relative_to(root)), sha256=record['sha256']))
    # Persist raw metadata even when semantics are not confirmed.
    save_json(ROOT / 'results/h5_metadata.json', geometry_records)
    if not cfg.get('axis_order') or not cfg.get('axis_source'):
        raise ValueError('Axis semantics require documentary confirmation; see results/h5_metadata.json')
    order = cfg['axis_order'].upper()
    if sorted(order) != ['X', 'Y', 'Z']:
        raise ValueError('axis_order must be a permutation of XYZ')
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(cases['train']))
    nv = max(1, round(0.2 * len(idx)))
    split = dict(train=[cases['train'][i] for i in idx[nv:]], val=[cases['train'][i] for i in idx[:nv]],
                 test=cases['test'], seed=42, axis_order=order, axis_source=cfg['axis_source'])
    save_json(ROOT / 'splits/la_split.json', split)
    return dict(counts={k: len(split[k]) for k in ['train', 'val', 'test']}, files=geometry_records,
                original_axis_order=order, internal_axis_order='ZYX',
                permutation=[order.index(a) for a in 'ZYX'], axis_source=cfg['axis_source'],
                spacing=cfg.get('spacing_zyx', [1, 1, 1]), distance_unit=cfg.get('distance_unit', 'voxel'),
                spacing_source=cfg.get('spacing_source', 'No trustworthy physical metadata; unit voxel grid'),
                patient_grouping='Each unique case file treated as one case; no repeated patient identifiers supplied')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--task', choices=['classification', 'segmentation', 'all'], default='all')
    args = ap.parse_args()
    cfg = load_config(args.config)
    output = ROOT / 'results/data_audit.json'
    audit = json.loads(output.read_text()) if output.exists() else {}
    errors = []
    for task, fn in [('classification', classification), ('segmentation', segmentation)]:
        if args.task not in ['all', task]:
            continue
        try:
            audit[task] = dict(status='passed', **fn(cfg[task]))
            progress(f'{task} data audit passed')
        except Exception as e:
            audit[task] = dict(status='blocked', error=f'{type(e).__name__}: {e}')
            errors.append(audit[task]['error'])
    save_json(output, audit)
    if errors:
        raise SystemExit('\n'.join(errors))


if __name__ == '__main__':
    main()
