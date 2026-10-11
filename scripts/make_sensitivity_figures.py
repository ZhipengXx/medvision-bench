"""Publication figures for separately reported exploratory selection sensitivity.

All three sensitivity runs must be complete before any figure is written. Test
comparison is optional until supplemental evaluation exists. Only saved metrics
are plotted: no fitting, test-driven selection, smoothing or invented intervals.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from src.common.plot_style import COLORS, apply_paper_style, panel_label, save_figure
from src.classification.selection import selection_score

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 183 / 25.4
METHODS = [('cls_r50_ce', 'CE', 'ce'),
           ('cls_r50_wce', 'Weighted CE', 'wce'),
           ('cls_r50_focal', 'Focal', 'focal')]
METRICS = [('accuracy', 'Accuracy'), ('macro_recall', 'Macro recall'), ('macro_f1', 'Macro F1')]


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runs(outputs):
    records = []
    for rid, name, color in METHODS:
        record = dict(run_id=rid, name=name, color=color)
        for label, run_id, criterion in [('original', rid, 'accuracy'),
                                          ('sensitivity', rid + '_sensitivity', 'macro_f1')]:
            folder = outputs / run_id
            meta_path, history_path = folder / 'metadata.json', folder / 'history.json'
            meta, history = read_json(meta_path), read_json(history_path)
            if not meta.get('complete'):
                raise ValueError(f'Run is incomplete; no final figures may be written: {run_id}')
            if not meta.get('formal'):
                raise ValueError(f'Smoke results cannot enter sensitivity figures: {run_id}')
            if [row['epoch'] for row in history] != list(range(1, len(history) + 1)):
                raise ValueError(f'Invalid complete epoch history: {run_id}')
            if len(history) != meta['completed_epochs']:
                raise ValueError(f'History and completion metadata differ: {run_id}')
            def score(row):
                return selection_score(dict(accuracy=row['val_accuracy'], macro_f1=row['val_macro_f1'],
                                             macro_recall=row.get('val_macro_recall', 0.)), criterion)
            best = max(history, key=score)
            if best['epoch'] != meta['best_epoch']:
                raise ValueError(f'Validation selection differs from saved best epoch: {run_id}')
            if label == 'sensitivity':
                cfg = meta['config']
                if cfg.get('selection_metric') != 'macro_f1' or cfg.get('minimum_epochs') != 20 or len(history) < 20:
                    raise ValueError(f'Unexpected exploratory sensitivity protocol: {run_id}')
            values = np.array([row['val_macro_f1'] for row in history])
            if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise ValueError(f'Invalid validation macro F1: {run_id}')
            record[label] = dict(history=history, best_epoch=best['epoch'],
                                 best_val_macro_f1=best['val_macro_f1'],
                                 metadata_sha256=digest(meta_path), history_sha256=digest(history_path))
        records.append(record)
    return records


def read_summary(path, expected):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    mapped = {row['run_id']: row for row in rows}
    if len(rows) != len(expected) or set(mapped) != set(expected):
        raise ValueError(f'Expected exactly the three matching classification runs: {path}')
    for rid, row in mapped.items():
        for key, _ in METRICS:
            value = float(row[key])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid saved {key} for {rid}')
    return mapped


def selection_plot(records, figures):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7), sharex=True, sharey=True, layout='none')
    fig.subplots_adjust(left=.085, right=.985, top=.87, bottom=.24, wspace=.19)
    maximum_epoch = max(len(record[label]['history']) for record in records for label in ['original', 'sensitivity'])
    for ax, record, letter in zip(axes, records, 'abc'):
        color = COLORS[record['color']]
        # Draw the new solid curve first and the original dark dashed curve on
        # top. Coincident trajectories remain visibly striped in both colors.
        for label, linecolor, linestyle, width, zorder in [
                ('sensitivity', color, '-', 1.75, 2),
                ('original', COLORS['muted'], (0, (3.2, 2.4)), 1.1, 3)]:
            rows = record[label]['history']
            ax.plot([row['epoch'] for row in rows], [100 * row['val_macro_f1'] for row in rows],
                    color=linecolor, linestyle=linestyle, linewidth=width, zorder=zorder)
        original, sensitivity = record['original'], record['sensitivity']
        ax.scatter(original['best_epoch'], 100 * original['best_val_macro_f1'],
                   marker='o', s=64, facecolors='white', edgecolors=COLORS['muted'],
                   linewidths=1.3, zorder=5)
        ax.scatter(sensitivity['best_epoch'], 100 * sensitivity['best_val_macro_f1'],
                   marker='D', s=19, facecolors=color, edgecolors='white', linewidths=.5, zorder=6)
        for label, y, textcolor in [('original', .125, COLORS['ink']), ('sensitivity', .025, COLORS['ink'])]:
            ax.text(.035, y, f"{'Original' if label == 'original' else 'Exploratory'} selected: e{record[label]['best_epoch']}",
                    transform=ax.transAxes, fontsize=7, color=textcolor, va='bottom')
        ax.set(xlabel='Epoch', xlim=(.5, maximum_epoch + .5), ylim=(0, 100))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(axis='y')
        panel_label(ax, letter, record['name'])
    axes[0].set_ylabel('Validation macro F1 (%)')
    handles = [
        Line2D([], [], color=COLORS['muted'], linestyle=(0, (3.2, 2.4)), linewidth=1.1,
               marker='o', markerfacecolor='white', markeredgecolor=COLORS['muted'], markersize=6,
               label='Original: accuracy selection'),
        Line2D([], [], color=COLORS['ink'], linewidth=1.75,
               marker='D', markerfacecolor=COLORS['ink'], markeredgecolor='white', markersize=4,
               label='Exploratory: ≥20 epochs, macro F1 selection'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .025),
               ncol=2, fontsize=7, handlelength=2.3, columnspacing=1.1)
    save_figure(fig, figures / 'classification_selection_sensitivity.png', preserve_size=True)
    plt.close(fig)


def metrics_plot(original, sensitivity, figures):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7), sharex=True, sharey=True, layout='none')
    fig.subplots_adjust(left=.12, right=.985, top=.87, bottom=.24, wspace=.18)
    for ax, (key, title), letter in zip(axes, METRICS, 'abc'):
        for y, (rid, name, color_key) in enumerate(METHODS):
            old = 100 * float(original[rid][key])
            new = 100 * float(sensitivity[rid + '_sensitivity'][key])
            color = COLORS[color_key]
            ax.plot([old, new], [y, y], color=color, linewidth=1.5, zorder=2)
            # A larger hollow original point remains visible when values match.
            ax.scatter(old, y, s=48, marker='o', facecolors='white',
                       edgecolors=COLORS['muted'], linewidths=1.15, zorder=3)
            ax.scatter(new, y, s=18, marker='o', facecolors=color,
                       edgecolors='white', linewidths=.45, zorder=4)
            ax.annotate(f'{old:.2f}', (old, y), xytext=(0, 8), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, color=COLORS['ink'])
            ax.annotate(f'{new:.2f}', (new, y), xytext=(0, -9), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color=COLORS['ink'])
        ax.set(xlim=(40, 100), ylim=(2.55, -.55), xticks=[40, 60, 80, 100],
               yticks=range(3), yticklabels=[name for _, name, _ in METHODS], xlabel='Test score (%)')
        ax.grid(axis='x')
        ax.tick_params(axis='both', labelsize=7)
        ax.tick_params(axis='y', length=0)
        panel_label(ax, letter, title)
    handles = [
        Line2D([], [], linestyle='none', marker='o', markerfacecolor='white',
               markeredgecolor=COLORS['muted'], markersize=6, label='Original: accuracy selection'),
        Line2D([], [], linestyle='none', marker='o', markerfacecolor=COLORS['ink'],
               markeredgecolor='white', markersize=4, label='Exploratory: ≥20 epochs, macro F1 selection'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .025),
               ncol=2, fontsize=7, handlelength=1.5, columnspacing=1.1)
    save_figure(fig, figures / 'classification_sensitivity_metrics.png', preserve_size=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--figures', type=Path)
    args = parser.parse_args()
    args.outputs = (ROOT / args.outputs).resolve()
    args.results = (ROOT / args.results).resolve()
    if args.figures is not None:
        args.figures = (ROOT / args.figures).resolve()
    records = load_runs(args.outputs)  # Validate every run before writing figures.
    figures = args.figures or args.results / 'figures'
    new_path = args.results / 'sensitivity/classification/summary.csv'
    original_path = args.results / 'classification/summary.csv'
    original = sensitivity = None
    if new_path.exists():
        original = read_summary(original_path, [rid for rid, _, _ in METHODS])
        sensitivity = read_summary(new_path, [rid + '_sensitivity' for rid, _, _ in METHODS])
    apply_paper_style()
    selection_plot(records, figures)
    generated = ['classification_selection_sensitivity']
    if sensitivity is not None:
        metrics_plot(original, sensitivity, figures)
        generated.append('classification_sensitivity_metrics')
    evidence = dict(experiment_status='exploratory_post_test_sensitivity',
                    methods=records, generated=generated, formats=['png', 'pdf', 'svg'],
                    uncertainty='No intervals displayed; original per-model diagnostics are separate',
                    test_summary_sha256={path.relative_to(ROOT).as_posix(): digest(path) for path in [original_path, new_path]}
                    if sensitivity is not None else {})
    # Histories are already archived separately; record selection and input hashes.
    for record in evidence['methods']:
        for label in ['original', 'sensitivity']:
            del record[label]['history']
    figures.mkdir(parents=True, exist_ok=True)
    (figures / 'classification_sensitivity_provenance.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print('Saved publication PNG/PDF/SVG:', ', '.join(generated))
    if sensitivity is None:
        print('Supplemental test summary absent; selection figure only. Rerun after frozen supplemental evaluation.')


if __name__ == '__main__':
    main()
