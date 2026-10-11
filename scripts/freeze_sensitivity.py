"""Freeze three completed exploratory runs without touching the formal manifest."""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_classification import CLASSES, ROOT, digest
from src.classification.selection import selection_score, selection_description

LOSSES = ('ce', 'wce', 'focal')
RELEVANT_SOURCE = ('scripts/train_classification.py', 'src/classification/selection.py',
                   'src/classification/losses.py', 'src/classification/model.py',
                   'src/classification/data.py', 'src/common/runtime.py')


def best_history(history):
    return max(history, key=lambda row: selection_score(
        dict(accuracy=row['val_accuracy'], macro_f1=row['val_macro_f1'],
             macro_recall=row['val_macro_recall']), 'macro_f1'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'splits/sensitivity_manifest.json')
    args = parser.parse_args()
    target = args.output.resolve()
    original_path = ROOT / 'splits/evaluation_manifest.json'
    if target == original_path.resolve() or target.exists():
        raise FileExistsError('Preserve all existing manifests; choose a new destination')
    original = json.loads(original_path.read_text())
    original_runs = {run['run_id']: run for run in original['runs']}
    initial_hash = digest(ROOT / 'outputs/resnet50_initial.pt')
    split_path = ROOT / 'splits/isic_split.json'
    split_hash = digest(split_path)
    runs, common_protocol, common_sources = [], None, None
    for loss in LOSSES:
        config_path = ROOT / f'configs/sensitivity/cls_r50_{loss}.yaml'
        cfg = yaml.safe_load(config_path.read_text())
        rid = f'cls_r50_{loss}_sensitivity'
        if cfg['run_id'] != rid or cfg['loss'] != loss:
            raise ValueError(f'Unexpected run identity: {config_path}')
        if cfg.get('selection_metric') != 'macro_f1' or cfg.get('minimum_epochs') != 20:
            raise ValueError('Sensitivity protocol requires macro F1 selection and minimum_epochs 20')
        baseline_cfg = original_runs[f'cls_r50_{loss}']['config']
        unchanged = {key: value for key, value in cfg.items()
                     if key not in ('run_id', 'selection_metric', 'minimum_epochs', 'save_val_predictions')}
        if unchanged != {key: value for key, value in baseline_cfg.items() if key != 'run_id'}:
            raise ValueError(f'Fitting settings differ from the original controlled protocol: {rid}')
        protocol = {key: value for key, value in cfg.items() if key not in ('run_id', 'loss')}
        if common_protocol is None:
            common_protocol = protocol
        elif protocol != common_protocol:
            raise ValueError('All losses must use the same training/selection protocol')
        out = ROOT / 'outputs' / rid
        meta = json.loads((out / 'metadata.json').read_text())
        history = json.loads((out / 'history.json').read_text())
        if not meta.get('formal') or not meta.get('complete'):
            raise ValueError(f'Not a completed independent run: {rid}')
        if meta['config'] != cfg or meta['seed'] != 42:
            raise ValueError(f'Training configuration changed or seed mismatch: {rid}')
        if meta['initialization']['initial_sha256'] != initial_hash:
            raise ValueError(f'Initialization differs: {rid}')
        if meta['split_sha256'] != split_hash or original_runs[f'cls_r50_{loss}']['split_sha256'] != split_hash:
            raise ValueError(f'Split differs from original protocol: {rid}')
        if meta['selection'] != selection_description('macro_f1'):
            raise ValueError(f'Incorrect selection rule: {rid}')
        selected = best_history(history)
        if len(history) != meta['completed_epochs'] or len(history) < cfg['minimum_epochs']:
            raise ValueError(f'Incomplete history/training floor: {rid}')
        if [row['epoch'] for row in history] != list(range(1, len(history) + 1)):
            raise ValueError(f'Epoch sequence is not complete: {rid}')
        if selected['epoch'] != meta['best_epoch'] or selected['val_macro_f1'] != meta['best_val_macro_f1']:
            raise ValueError(f'Checkpoint is not the macro F1-selected validation maximum: {rid}')
        if digest(out / 'best.pt') != meta['best_checkpoint_sha256']:
            raise ValueError(f'Checkpoint changed after training: {rid}')
        sources = {key: meta['source_hashes'][key] for key in RELEVANT_SOURCE}
        if common_sources is None:
            common_sources = sources
        elif common_sources != sources:
            raise ValueError('Relevant training source differs between losses')
        for name, expected in sources.items():
            if digest(ROOT / name) != expected:
                raise ValueError(f'Training source modified before freeze: {name}')
        runs.append(dict(run_id=rid, checkpoint=f'outputs/{rid}/best.pt',
                         checkpoint_sha256=meta['best_checkpoint_sha256'], config=cfg,
                         config_file=str(config_path.relative_to(ROOT)), config_sha256=digest(config_path),
                         split=str(split_path.relative_to(ROOT)), split_sha256=split_hash,
                         data_config_sha256=digest(ROOT / cfg['data_config']),
                         initialization_sha256=initial_hash, best_epoch=meta['best_epoch'],
                         best_val_macro_f1=meta['best_val_macro_f1'], model_selection=meta['selection']))
    original_artifacts = [original_path]
    for loss in LOSSES:
        original_artifacts += [ROOT / f'results/classification/cls_r50_{loss}/{name}' for name in ('predictions.csv', 'metrics.json')]
    manifest = dict(
        frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        experiment_status='exploratory_post_test_sensitivity',
        interpretation='Original test results were already observed before this protocol was designed. These runs assess sensitivity to checkpoint selection/training duration and cannot support confirmatory improvement claims. Original formal results remain primary.',
        original_manifest_sha256=digest(original_path),
        original_artifact_sha256={str(path.relative_to(ROOT)): digest(path) for path in original_artifacts},
        protocol=common_protocol, relevant_training_source_sha256=common_sources,
        runs=runs, classification_classes=CLASSES,
        classification_rule='argmax, no TTA; all three validation-selected checkpoints frozen before supplemental test evaluation',
        changes_from_original=['Primary checkpoint criterion: validation macro F1 (ties: macro recall, accuracy)',
                               'Minimum 20 epochs before patience stopping; other fitting settings unchanged',
                               'All epoch validation predictions saved for auditing'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f'Frozen three exploratory checkpoints: {target}')


if __name__ == '__main__':
    main()
