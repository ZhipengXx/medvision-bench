"""Verify the public research artifacts without datasets or checkpoint binaries."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


def load_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def require_close(actual, expected, description):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-10):
        raise ValueError(f'{description}: saved value disagrees with source data')


def verify_classifier(folder):
    predictions = rows(folder / 'predictions.csv')
    saved = load_json(folder / 'metrics.json')
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    ids = []
    for row in predictions:
        truth, predicted = int(row['gt']), int(row['pred'])
        probabilities = [float(row['p_' + name]) for name in CLASSES]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities):
            raise ValueError(f'{folder.name}: invalid class probabilities')
        if not math.isclose(sum(probabilities), 1, rel_tol=0, abs_tol=1e-5):
            raise ValueError(f'{folder.name}: probabilities do not sum to one')
        if max(range(len(CLASSES)), key=probabilities.__getitem__) != predicted:
            raise ValueError(f'{folder.name}: decision does not use multiclass argmax')
        confusion[truth][predicted] += 1
        ids.append(row['image_id'])
    if len(set(ids)) != len(ids) or len(ids) != saved['n']:
        raise ValueError(f'{folder.name}: duplicate image or image-count mismatch')
    if confusion != saved['confusion_counts']:
        raise ValueError(f'{folder.name}: confusion matrix does not match predictions')
    support = [sum(row) for row in confusion]
    assigned = [sum(row[k] for row in confusion) for k in range(len(CLASSES))]
    recall = [confusion[k][k] / support[k] for k in range(len(CLASSES))]
    precision = [confusion[k][k] / assigned[k] if assigned[k] else 0 for k in range(len(CLASSES))]
    f1 = [2 * confusion[k][k] / (support[k] + assigned[k]) for k in range(len(CLASSES))]
    calculated = dict(accuracy=sum(confusion[k][k] for k in range(len(CLASSES))) / len(ids),
                      macro_recall=sum(recall) / len(CLASSES),
                      macro_precision=sum(precision) / len(CLASSES), macro_f1=sum(f1) / len(CLASSES))
    for key, value in calculated.items():
        require_close(value, saved[key], f'{folder.name} {key}')
    per_class = rows(folder / 'per_class.csv')
    if [row['class_name'] for row in per_class] != CLASSES:
        raise ValueError('Unexpected class ordering')
    for k, row in enumerate(per_class):
        if int(row['support']) != support[k]:
            raise ValueError('Per-class support mismatch')
        for key, values in [('recall', recall), ('precision', precision), ('f1', f1)]:
            require_close(row[key], values[k], f'{folder.name} {CLASSES[k]} {key}')
    return dict(run_id=folder.name, images=len(ids), prediction_metrics_match=True), ids


def verify_segmentation(folder):
    records = rows(folder / 'per_case_metrics.csv')
    saved = load_json(folder / 'metrics.json')
    for metric in ['dice', 'jaccard', 'asd', 'hd95']:
        values = [float(row[metric]) for row in records]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError('Invalid segmentation metric')
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        require_close(mean, saved[metric + '_mean'], f'{folder.name} {metric} mean')
        require_close(sd, saved[metric + '_std'], f'{folder.name} {metric} population SD')
    ids = [row['case_id'] for row in records]
    if len(ids) != 20 or len(set(ids)) != len(ids):
        raise ValueError('Expected 20 unique test volumes')
    if any(row['distance_unit'] != 'voxel' for row in records):
        raise ValueError('Unverified physical distance units')
    return dict(run_id=folder.name, volumes=len(ids), aggregate_metrics_match=True), ids


def verify_project():
    required = ['README.md', 'NOTICE', 'LICENSE', 'requirements.lock.txt', 'report/report.pdf', 'report/report.tex',
                'report/methods_expanded.json', 'report/typesetting_checks.json',
                'results/figures/method_overview.pdf', 'results/figures/unet_architecture.svg']
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise ValueError('Required project artifacts missing: ' + ', '.join(missing))
    classifier_checks, reference_ids = [], None
    for prefix, names in [('classification', ['cls_r50_ce', 'cls_r50_wce', 'cls_r50_focal']),
                          ('sensitivity/classification', ['cls_r50_ce_sensitivity',
                           'cls_r50_wce_sensitivity', 'cls_r50_focal_sensitivity'])]:
        for name in names:
            check, ids = verify_classifier(ROOT / 'results' / prefix / name)
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise ValueError('Classifiers do not share the same ordered test images')
            summary = {row['run_id']: row for row in rows(ROOT / 'results' / prefix / 'summary.csv')}
            saved = load_json(ROOT / 'results' / prefix / name / 'metrics.json')
            for key in ['accuracy', 'macro_recall', 'macro_precision', 'macro_f1', 'macro_auroc']:
                require_close(summary[name][key], saved[key], f'{name} summary {key}')
            classifier_checks.append(check)
    segmentation_checks, case_ids = [], None
    for name in ['seg_unet2d', 'seg_unet3d']:
        check, ids = verify_segmentation(ROOT / 'results/segmentation' / name)
        if case_ids is not None and ids != case_ids:
            raise ValueError('Segmentation models do not share ordered test cases')
        case_ids = ids
        summary = {row['run_id']: row for row in rows(ROOT / 'results/segmentation/summary.csv')}
        saved = load_json(ROOT / 'results/segmentation' / name / 'metrics.json')
        for metric in ['dice', 'jaccard', 'asd', 'hd95']:
            for suffix in ['mean', 'std']:
                key = metric + '_' + suffix
                require_close(summary[name][key], saved[key], f'{name} summary {key}')
        segmentation_checks.append(check)
    typesetting = load_json(ROOT / 'report/typesetting_checks.json')
    for key in ['missing_characters', 'undefined_controls', 'undefined_references']:
        if typesetting[key]:
            raise ValueError('Native manuscript typesetting has unresolved errors')
    if typesetting['native_display_equations'] != 11:
        raise ValueError('Expected eleven native display equations')
    return dict(passed=True, classification=classifier_checks, segmentation=segmentation_checks,
                report_pages=typesetting['pages'], native_equations=typesetting['native_display_equations'],
                scope='Saved prediction and metric consistency; no dataset or checkpoint evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON verification record')
    args = parser.parse_args()
    result = verify_project()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Verified 6 classifiers, 2 segmenters and {result["native_equations"]} native equations '
          f'in a {result["report_pages"]}-page manuscript')


if __name__ == '__main__':
    main()
