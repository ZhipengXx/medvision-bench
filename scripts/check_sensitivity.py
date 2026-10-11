"""Check supplemental validation selection, saved test metrics and original hashes."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_classification import CLASSES, ROOT, digest, read_predictions, confusion_batches, scores, auc_batches
from scripts.freeze_sensitivity import best_history


def verify_rows(path, expected):
    ids, y, pred, prob = read_predictions(path)
    if ids != [row['image_id'] for row in expected] or not np.array_equal(y, [row['label'] for row in expected]):
        raise ValueError(f'Split image IDs/order/labels differ: {path}')
    cm = confusion_batches(np.ones((1, len(y))), y, pred)
    values = {key: float(value[0]) for key, value in scores(cm).items()}
    values['macro_auroc'] = float(auc_batches(np.ones((1, len(y))), y, prob)[0])
    return values, cm[0].astype(int)


def close(actual, expected, label):
    if not np.isclose(actual, expected, rtol=0, atol=1e-10):
        raise ValueError(f'Metric mismatch {label}: {actual} != {expected}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'splits/sensitivity_manifest.json')
    parser.add_argument('--results', type=Path, default=ROOT / 'results/sensitivity')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['experiment_status'] != 'exploratory_post_test_sensitivity' or len(manifest['runs']) != 3:
        raise ValueError('Expected the three-run exploratory sensitivity manifest')
    for name, expected in manifest['original_artifact_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Original formal artifact changed: {name}')
    provenance = json.loads((args.results / 'evaluation_provenance.json').read_text())
    if provenance['manifest_sha256'] != digest(args.manifest):
        raise ValueError('Supplemental evaluation used a different manifest')
    with (args.results / 'classification/summary.csv').open(newline='') as stream:
        summary = {row['run_id']: row for row in csv.DictReader(stream)}
    if set(summary) != {run['run_id'] for run in manifest['runs']}:
        raise ValueError('Supplemental summary must contain exactly the three frozen runs')
    records = []
    for run in manifest['runs']:
        rid = run['run_id']
        for name, expected in [(run['checkpoint'], run['checkpoint_sha256']),
                               (run['split'], run['split_sha256']),
                               (run['config_file'], run['config_sha256']),
                               (run['config']['data_config'], run['data_config_sha256'])]:
            if digest(ROOT / name) != expected:
                raise ValueError(f'Frozen sensitivity artifact changed: {name}')
        out = ROOT / 'outputs' / rid
        meta = json.loads((out / 'metadata.json').read_text())
        history = json.loads((out / 'history.json').read_text())
        split = json.loads((ROOT / run['split']).read_text())
        selected = best_history(history)
        if not meta['complete'] or meta['best_epoch'] != selected['epoch'] or run['best_epoch'] != selected['epoch']:
            raise ValueError(f'Selection metadata mismatch: {rid}')
        for row in history:
            path = out / 'validation' / f"epoch_{row['epoch']:03d}.csv"
            values, _ = verify_rows(path, split['val'])
            for key in ('accuracy', 'macro_f1', 'macro_recall'):
                close(values[key], row['val_' + key], f'{rid} epoch {row["epoch"]} validation {key}')
        folder = args.results / 'classification' / rid
        values, cm = verify_rows(folder / 'predictions.csv', split['test'])
        _, test_y, _, test_probability = read_predictions(folder / 'predictions.csv')
        saved = json.loads((folder / 'metrics.json').read_text())
        if saved['checkpoint_sha256'] != run['checkpoint_sha256'] or saved['n'] != len(split['test']):
            raise ValueError(f'Test checkpoint/count mismatch: {rid}')
        for key, value in values.items():
            if key in saved:
                close(value, saved[key], f'{rid} test {key}')
            if key in summary[rid]:
                close(value, float(summary[rid][key]), f'{rid} summary {key}')
        if not np.array_equal(cm, saved['confusion_counts']):
            raise ValueError(f'Test confusion matrix mismatch: {rid}')
        normalized = cm / cm.sum(1, keepdims=True)
        if not np.allclose(normalized, saved['confusion_row_normalized'], rtol=0, atol=1e-10):
            raise ValueError(f'Test confusion normalization mismatch: {rid}')
        with (folder / 'per_class.csv').open(newline='') as stream:
            per_class = list(csv.DictReader(stream))
        if [row['class_name'] for row in per_class] != CLASSES:
            raise ValueError(f'Per-class order mismatch: {rid}')
        for k, row in enumerate(per_class):
            tp, support, assigned = cm[k, k], cm[k].sum(), cm[:, k].sum()
            if int(row['support']) != support:
                raise ValueError(f'Per-class test support mismatch: {rid}')
            close(float(row['recall']), tp / support, f'{rid} {CLASSES[k]} recall')
            close(float(row['precision']), tp / assigned if assigned else 0., f'{rid} {CLASSES[k]} precision')
            close(float(row['f1']), 2 * tp / (support + assigned), f'{rid} {CLASSES[k]} F1')
            close(float(row['auroc']), roc_auc_score(test_y == k, test_probability[:, k]),
                  f'{rid} {CLASSES[k]} AUROC')
        records.append(dict(run_id=rid, epochs_checked=len(history), best_epoch=selected['epoch'],
                            best_val_macro_f1=selected['val_macro_f1'], test_images_checked=len(split['test']),
                            test_accuracy=values['accuracy'], test_macro_f1=values['macro_f1']))
    result = dict(passed=True, experiment_status=manifest['experiment_status'],
                  original_artifacts_unchanged=True, manifest_sha256=digest(args.manifest), checks=records)
    destination = args.results / 'validation_checks.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Checked all epoch validation predictions and supplemental test metrics: {destination}')


if __name__ == '__main__':
    main()
