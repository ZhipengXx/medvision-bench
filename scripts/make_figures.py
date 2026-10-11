"""Publication figures from formal histories, frozen metrics and predictions.

No model, metric, prediction or checkpoint is changed by this script. Every
figure is exported as 450 dpi PNG plus PDF and SVG. Display cases retain the
original post-test case and slice selection rules.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from sklearn.metrics import auc, roc_curve

from src.common.runtime import ROOT, load_config, save_json
from src.common.plot_style import (
    CLASS_COLORS, COLORS, CONFUSION_CMAP, apply_paper_style,
    panel_label, save_figure,
)
from src.classification.data import CLASSES
from src.segmentation.data import read_volume


WIDTH = 183 / 25.4

RUNS = [('cls_r50_ce', 'CE', 'ce', 'o'),
        ('cls_r50_wce', 'Weighted CE', 'wce', 's'),
        ('cls_r50_focal', 'Focal', 'focal', '^')]
SEG_RUNS = [('seg_unet3d', '3D U-Net', '3d', 'o'),
            ('seg_unet2d', '2D U-Net', '2d', 's')]


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text())


def finish(fig, figures, name):
    save_figure(fig, figures / name, preserve_size=True)
    plt.close(fig)


def metric_axis(ax, ylabel, xlabel='Epoch', percentage=False):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis='y')
    if percentage:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def classification_learning(figures):
    baseline = read_json(ROOT / 'outputs/cls_r50_ce/history.json')
    selected = read_json(ROOT / 'outputs/cls_r50_ce/metadata.json')['best_epoch']
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75))
    for ax, key, label, letter in zip(axes, ['ce', 'accuracy'],
                                    ['Cross-entropy loss', 'Accuracy'], 'ab'):
        for prefix, name, style in [('train', 'Training', '-'),
                                    ('val', 'Validation', '--')]:
            ax.plot([r['epoch'] for r in baseline],
                    [r[f'{prefix}_{key}'] for r in baseline],
                    color=COLORS['train' if prefix == 'train' else 'validation'],
                    linestyle=style, label=name)
        ax.axvline(selected, color=COLORS['muted'], linewidth=0.8,
                   linestyle=':', label=f'Selected epoch ({selected})')
        metric_axis(ax, label, percentage=key == 'accuracy')
        ax.set_xlim(1, baseline[-1]['epoch'])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_ylim((0, 1) if key == 'accuracy' else
                    (0, max(r['train_ce'] for r in baseline) * 1.06))
        panel_label(ax, letter, label)
    axes[0].legend(loc='upper right', fontsize=7)
    finish(fig, figures, 'classification_learning_curves.png')

    # Ordinary validation CE is comparable across all three loss objectives.
    histories = {rid: read_json(ROOT / 'outputs' / rid / 'history.json')
                 for rid, *_ in RUNS}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.8))
    for ax, key, label, letter in zip(
            axes, ['val_ce', 'val_accuracy', 'val_macro_f1'],
            ['Ordinary CE loss', 'Validation accuracy', 'Validation macro F1'], 'abc'):
        for rid, name, color, marker in RUNS:
            history = histories[rid]
            best = read_json(ROOT / 'outputs' / rid / 'metadata.json')['best_epoch']
            ax.plot([r['epoch'] for r in history], [r[key] for r in history],
                    label=name, color=COLORS[color])
            record = next(r for r in history if r['epoch'] == best)
            ax.scatter([best], [record[key]], marker=marker, s=25,
                       color=COLORS[color], edgecolor='white', linewidth=0.65, zorder=4)
        metric_axis(ax, label, percentage=key != 'val_ce')
        ax.set_xlim(1, max(h[-1]['epoch'] for h in histories.values()))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.set_ylim((0, 1) if key != 'val_ce' else
                    (0, max(r[key] for h in histories.values() for r in h) * 1.05))
        panel_label(ax, letter, label)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center',
               ncol=3, fontsize=7)
    finish(fig, figures, 'classification_validation_comparison.png')


def classification_roc(results, figures):
    rows = read_csv(results / 'classification/cls_r50_ce/predictions.csv')
    y = np.asarray([int(r['gt']) for r in rows])
    fig, ax = plt.subplots(figsize=(89 / 25.4, 3.3))
    for k, (name, color) in enumerate(zip(CLASSES, CLASS_COLORS)):
        if (y == k).any() and (y != k).any():
            fpr, tpr, _ = roc_curve(y == k, [float(r['p_' + name]) for r in rows])
            ax.plot(fpr, tpr, color=color, linestyle=['-', '--', '-.'][k % 3],
                    linewidth=1.35, label=f'{name}  {auc(fpr, tpr):.3f}')
    ax.plot([0, 1], [0, 1], color='#AAB4B9', linestyle=':', linewidth=0.9)
    ax.set(xlabel='False positive rate', ylabel='True positive rate',
           xlim=(0, 1), ylim=(0, 1.015))
    ax.set_aspect('equal', adjustable='box')
    ax.legend(title='Class       AUROC', title_fontsize=7,
              loc='lower right', fontsize=7, labelspacing=0.32)
    finish(fig, figures, 'classification_roc_ce.png')


def draw_confusion(ax, matrix, normalized, maximum, show_ylabel=True):
    im = ax.imshow(matrix, cmap=CONFUSION_CMAP, vmin=0, vmax=maximum)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            label = f'{matrix[i, j] * 100:.0f}%' if normalized else str(int(matrix[i, j]))
            ax.text(j, i, label, ha='center', va='center', fontsize=7,
                    fontweight='bold' if i == j else 'normal',
                    color='white' if matrix[i, j] > maximum * 0.58 else COLORS['ink'])
    ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)),
           xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted class', ylabel='True class' if show_ylabel else '')
    ax.tick_params(axis='both', length=0, labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor')
    ax.set_xticks(np.arange(-0.5, len(CLASSES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASSES), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5, alpha=1)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def classification_confusion(results, figures):
    metrics = {rid: read_json(results / 'classification' / rid / 'metrics.json')
               for rid, *_ in RUNS}
    max_count = max(np.asarray(m['confusion_counts']).max() for m in metrics.values())
    for rid, name, *_ in RUNS:
        m = metrics[rid]
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.45))
        for ax, key, norm, limit, letter, title in zip(
                axes, ['confusion_counts', 'confusion_row_normalized'], [False, True],
                [max_count, 1], 'ab', ['Counts', 'Recall by true class']):
            im = draw_confusion(ax, np.asarray(m[key]), norm, limit)
            panel_label(ax, letter, f'{name}: {title}')
            bar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
            bar.ax.tick_params(labelsize=7)
            if norm:
                bar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            else:
                bar.ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        finish(fig, figures, rid + '_confusion.png')

    # Give every 7 x 7 matrix enough width for its cell values at print size.
    # The fourth panel records the denominator of each normalized row.
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 6.3))
    matrix_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    for ax, (rid, name, *_), letter in zip(matrix_axes, RUNS, 'abc'):
        im = draw_confusion(ax, np.asarray(metrics[rid]['confusion_row_normalized']),
                            True, 1)
        panel_label(ax, letter, name)
    counts = np.asarray(metrics['cls_r50_ce']['confusion_counts']).sum(axis=1)
    ax = axes[1, 1]
    y = np.arange(len(CLASSES))
    ax.scatter(counts, y, color=COLORS['ink'], s=16, zorder=3)
    for yy, count in zip(y, counts):
        ax.text(count + 22, yy, f'{int(count):,}', va='center', fontsize=7)
    ax.set(yticks=y, yticklabels=CLASSES, ylim=(6.5, -.5), xlim=(0, 1060),
           xticks=[0, 250, 500, 750, 1000], xlabel='Test images', ylabel='True class')
    panel_label(ax, 'd', 'Shared test support')
    bar = fig.colorbar(im, ax=matrix_axes, fraction=0.018, pad=0.018,
                       shrink=.84, label='Images within true class (%)')
    bar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bar.ax.tick_params(labelsize=7)
    finish(fig, figures, 'classification_confusion_comparison.png')


def classification_recall(results, figures):
    per_class = {rid: {r['class_name']: r for r in read_csv(
        results / 'classification' / rid / 'per_class.csv')} for rid, *_ in RUNS}
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    y = np.arange(len(CLASSES))
    for offset, (rid, name, color, marker) in zip([-.2, 0, .2], RUNS):
        values = [float(per_class[rid][class_name]['recall']) for class_name in CLASSES]
        ax.scatter(values, y + offset, marker=marker, s=20, color=COLORS[color],
                   label=name, zorder=3, edgecolor='white', linewidth=.35)
    labels = [f"{name}  (n = {per_class['cls_r50_ce'][name]['support']})" for name in CLASSES]
    ax.set(yticks=y, yticklabels=labels, xlabel='Test recall (%)', xlim=(0, 1),
           ylim=(len(CLASSES) - .4, -.6))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.58, 1.12))
    finish(fig, figures, 'classification_recall_comparison.png')


def classification_tradeoffs(results, figures):
    metrics = {rid: read_json(results / 'classification' / rid / 'metrics.json')
               for rid, *_ in RUNS}
    for rid, *_ in RUNS:
        per_class = {r['class_name']: r for r in read_csv(
            results / 'classification' / rid / 'per_class.csv')}
        metrics[rid].update({f'recall_{name}': float(per_class[name]['recall'])
                             for name in ['MEL', 'BCC']})
    metric_keys = ['accuracy', 'macro_recall', 'macro_f1', 'recall_MEL', 'recall_BCC']
    labels = ['Accuracy', 'Macro recall', 'Macro F1', 'MEL recall', 'BCC recall']
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.65),
                             gridspec_kw={'width_ratios': [1.1, 1]})
    y = np.arange(len(metric_keys))
    for offset, (rid, name, color, marker) in zip([-.20, 0, .20], RUNS):
        values = [100 * metrics[rid][key] for key in metric_keys]
        axes[0].plot(values, y + offset, linestyle='none', marker=marker,
                     color=COLORS[color], label=name, markersize=4.6)
    axes[0].set(yticks=y, yticklabels=labels, xlabel='Test score (%)',
                xlim=(40, 90), ylim=(len(metric_keys) - .45, -.65))
    axes[0].grid(axis='x')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center',
               ncol=3, fontsize=7)
    panel_label(axes[0], 'a', 'Absolute test performance')
    for offset, (rid, name, color, _) in zip([-.16, .16], RUNS[1:]):
        changes = [100 * (metrics[rid][key] - metrics['cls_r50_ce'][key])
                   for key in metric_keys]
        axes[1].barh(y + offset, changes, height=.27, color=COLORS[color], label=name)
        for yy, delta in zip(y + offset, changes):
            inside = delta < -8
            axes[1].text(delta + (.4 if inside else .15 if delta >= 0 else -.15), yy,
                         f'{delta:+.2f}', va='center',
                         ha='left' if delta >= 0 or inside else 'right', fontsize=7,
                         color='white' if inside else COLORS['ink'])
    axes[1].axvline(0, color=COLORS['muted'], linewidth=.8)
    axes[1].set(yticks=y, yticklabels=labels,
                xlabel='Change from CE (percentage points)',
                xlim=(-21.5, 10.5), ylim=(len(metric_keys) - .45, -.65))
    axes[1].grid(axis='x')
    panel_label(axes[1], 'b', 'Metric tradeoffs relative to CE')
    finish(fig, figures, 'classification_method_tradeoffs.png')


def classification_effect_sizes(results, figures):
    path = results / 'loss_diagnostics.json'
    if not path.exists():
        return
    diagnostics = read_json(path)
    keys = ['accuracy', 'macro_recall', 'macro_f1']
    labels = ['Accuracy', 'Macro recall', 'Macro F1']
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), sharex=True, sharey=True)
    for ax, (rid, name, color, marker), letter in zip(axes, RUNS[1:], 'ab'):
        rows = diagnostics['baseline_comparisons'][rid]['metrics']
        for y, key in enumerate(keys):
            row = rows[key]
            delta = 100 * row['delta']
            low, high = np.asarray(row['ci95']) * 100
            ax.errorbar(delta, y, xerr=[[delta - low], [high - delta]],
                        color=COLORS[color], fmt=marker, capsize=3,
                        elinewidth=1.4, capthick=1.1, markersize=4.5)
            ax.annotate(f'{delta:+.2f} [{low:+.2f}, {high:+.2f}]',
                        (delta, y), xytext=(0, -13), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color=COLORS['muted'])
        ax.axvline(0, color='#777777', linestyle=(0, (2, 2)), linewidth=.7)
        ax.set(yticks=range(len(keys)), yticklabels=labels, xlim=(-8, 14),
               ylim=(len(keys) - .4, -.55), xlabel='Change from CE (percentage points)')
        ax.grid(axis='x')
        panel_label(ax, letter, f'{name} vs CE (95% CI)')
    finish(fig, figures, 'classification_effect_sizes.png')


def classification_distribution(figures):
    split = read_json(ROOT / 'splits/isic_split.json')
    counts = {name: np.bincount([r['label'] for r in split[name]],
                                minlength=len(CLASSES))
              for name in ['train', 'val', 'test']}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.2),
                             gridspec_kw={'width_ratios': [.9, 1.25]})
    y = np.arange(len(CLASSES))
    # A position mark avoids giving log-scaled bars a misleading length origin.
    axes[0].scatter(counts['train'], y, color=COLORS['ce'], s=20, zorder=3)
    axes[0].set(yticks=y, yticklabels=CLASSES, xscale='log', xlabel='Training images (log scale)',
                xlim=(10, counts['train'].max() * 2.8))
    axes[0].invert_yaxis()
    for yy, count in zip(y, counts['train']):
        axes[0].text(count * 1.08, yy, f'{count:,}', va='center', fontsize=7)
    axes[0].grid(axis='x')
    panel_label(axes[0], 'a', 'Training class imbalance')
    for offset, name, label, color in zip(
            [-.23, 0, .23], ['train', 'val', 'test'], ['Train', 'Validation', 'Test'],
            ['#263238', '#788891', '#C2CDD3']):
        axes[1].bar(y + offset, 100 * counts[name] / counts[name].sum(), width=.215,
                    color=color, label=f'{label} (n = {counts[name].sum():,})')
    axes[1].set(xticks=y, xticklabels=CLASSES, ylabel='Images within split (%)', ylim=(0, 82))
    axes[1].tick_params(axis='x', labelsize=7)
    axes[1].grid(axis='y')
    axes[1].legend(loc='upper right', fontsize=7)
    panel_label(axes[1], 'b', 'Class composition across splits')
    finish(fig, figures, 'classification_class_distribution.png')


def segmentation_learning(figures):
    histories = {rid: read_json(ROOT / 'outputs' / rid / 'train_history.json')
                 for rid, *_ in SEG_RUNS}
    validations = {rid: read_json(ROOT / 'outputs' / rid / 'history.json')
                   for rid, *_ in SEG_RUNS}
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), sharex=True, sharey='row')
    ymax = 0
    xmax = max(h[-1]['step'] for h in histories.values()) / 1000
    loss_keys = [('total', 'CE + Dice', COLORS['ce'], '-'),
                 ('ce', 'CE', COLORS['wce'], '--'),
                 ('dice_loss', 'Dice loss', COLORS['focal'], '-.')]
    for col, (rid, name, color, marker) in enumerate(SEG_RUNS):
        hist, vals = histories[rid], validations[rid]
        win = min(100, len(hist))
        steps = np.asarray([r['step'] for r in hist][win - 1:]) / 1000
        stride = max(1, len(steps) // 1200)
        indices = np.unique(np.r_[np.arange(0, len(steps), stride), len(steps) - 1])
        for key, label, line_color, linestyle in loss_keys:
            smooth = np.convolve([r[key] for r in hist], np.ones(win) / win, 'valid')
            ymax = max(ymax, float(smooth.max()))
            axes[0, col].plot(steps[indices], smooth[indices], label=label,
                              color=line_color, linestyle=linestyle, linewidth=1.15)
        axes[0, col].set_ylabel(f'Loss ({win}-update mean)')
        axes[0, col].grid(axis='y')
        panel_label(axes[0, col], 'ab'[col], f'{name}: training objective')
        axes[1, col].plot([r['step'] / 1000 for r in vals],
                          [r['val_mean_dice'] for r in vals], color=COLORS[color],
                          marker=marker, linewidth=1.4, markersize=3.2)
        best = max(vals, key=lambda r: r['val_mean_dice'])
        axes[1, col].scatter([best['step'] / 1000], [best['val_mean_dice']],
                             marker='*', s=75, color=COLORS[color],
                             edgecolor='white', linewidth=.5, zorder=4)
        axes[1, col].set(xlabel=r'Optimizer updates ($\times 10^3$)', ylabel='Full-volume validation Dice',
                         ylim=(0, 1), xlim=(0, xmax))
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axes[1, col].grid(axis='y')
        panel_label(axes[1, col], 'cd'[col], f'{name}: validation')
        axes[1, col].text(.98, .05, f"Best: {best['val_mean_dice']:.3f} at {best['step']:,} updates",
                          transform=axes[1, col].transAxes, ha='right', color=COLORS['muted'],
                          fontsize=7)
    for ax in axes[0]:
        ax.set_ylim(0, ymax * 1.06)
    axes[0, 0].legend(loc='upper right', fontsize=7)
    finish(fig, figures, 'segmentation_learning_curves.png')


def segmentation_paired(results, figures):
    seg = {rid: {r['case_id']: r for r in read_csv(
        results / 'segmentation' / rid / 'per_case_metrics.csv')} for rid, *_ in SEG_RUNS}
    keys = sorted(seg['seg_unet3d'])
    if set(keys) != set(seg['seg_unet2d']):
        raise ValueError('Paired segmentation plot requires the same complete-case test set')
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 5.1), sharey=True)
    y = np.arange(len(keys))
    for ax, key, letter in zip(axes, ['dice', 'hd95'], 'ab'):
        v3 = np.asarray([float(seg['seg_unet3d'][k][key]) for k in keys])
        v2 = np.asarray([float(seg['seg_unet2d'][k][key]) for k in keys])
        for yy, a, b in zip(y, v3, v2):
            ax.plot([a, b], [yy, yy], color='#C7D0D5', linewidth=1.15, zorder=1)
        for values, (_, name, color, marker) in zip([v3, v2], SEG_RUNS):
            ax.scatter(values, y, color=COLORS[color], marker=marker,
                       s=19, label=name, zorder=3, edgecolor='white', linewidth=.4)
        # An equal score should show both methods, rather than the later square
        # hiding the first circle (one test case has identical HD95).
        equal = np.isclose(v3, v2, rtol=0, atol=1e-12)
        if equal.any():
            ax.scatter(v3[equal], y[equal], facecolors='none',
                       edgecolors=COLORS['3d'], s=48, linewidth=1, zorder=4)
        unit = seg['seg_unet3d'][keys[0]]['distance_unit']
        ax.set_xlabel('Volume Dice' if key == 'dice' else f'HD95 ({unit}s)')
        ax.set(yticks=y, ylim=(len(keys) - .3, -1.6))
        ax.set_xlim((0, 1) if key == 'dice' else (0, max(v3.max(), v2.max()) * 1.06))
        ax.grid(axis='x')
        ax.tick_params(axis='y', length=0, labelsize=7)
        panel_label(ax, letter, 'Per-case overlap' if key == 'dice' else 'Per-case boundary distance')
        ax.text(.02, .985, f'Mean: 3D {v3.mean():.3f} / 2D {v2.mean():.3f}',
                transform=ax.transAxes, ha='left', va='top', fontsize=7,
                color=COLORS['muted'])
    axes[0].set_yticklabels([f'{i + 1:02d}  {k}' for i, k in enumerate(keys)])
    axes[0].set_ylabel('Test case', fontsize=7)
    fig.legend(*axes[1].get_legend_handles_labels(), loc='outside lower center',
               ncol=2, fontsize=7)
    finish(fig, figures, 'segmentation_paired_comparison.png')
    return seg, keys


def slice_window(*masks):
    """One union-based window per case; preserve every visible mask error."""
    union = np.logical_or.reduce(masks)
    points = np.argwhere(union)
    if not len(points):
        return 0, union.shape[1], 0, union.shape[0]
    ymin, xmin = points.min(0)
    ymax, xmax = points.max(0) + 1
    pad = max(16, int(max(ymax - ymin, xmax - xmin) * .13))
    return (max(0, xmin - pad), min(union.shape[1], xmax + pad),
            max(0, ymin - pad), min(union.shape[0], ymax + pad))


def overlay(ax, mask, color, alpha=.55):
    if mask.any():
        ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([color]),
                  vmin=0, vmax=1, alpha=alpha, interpolation='nearest')


def segmentation_cases(results, figures, seg, keys):
    ranked = sorted(keys, key=lambda k: float(seg['seg_unet3d'][k]['dice']))
    selected = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    selected.append(max((k for k in keys if k not in selected),
                        key=lambda k: float(seg['seg_unet3d'][k]['hd95'])))
    rules = ['Lowest 3D Dice', 'Median-ranked 3D Dice', 'Highest 3D Dice',
             'Highest remaining 3D HD95']
    data_cfg = load_config(ROOT / 'configs/data.yaml')['segmentation']
    case_paths = {r['case_id']: r for r in read_json(ROOT / 'splits/la_split.json')['test']}
    selection, displays = [], []
    for k, rule in zip(selected, rules):
        im, gt = read_volume(ROOT / data_cfg['root'] / case_paths[k]['path'], data_cfg)
        with np.load(results / 'segmentation/seg_unet3d' / (k + '.npz')) as saved:
            p3 = saved['prediction']
        with np.load(results / 'segmentation/seg_unet2d' / (k + '.npz')) as saved:
            p2 = saved['prediction']
        errors = (p3 != gt).sum((1, 2))
        areas = gt.sum((1, 2))
        valid = np.where(areas >= max(1, areas.max() * .1))[0]
        if not len(valid):
            valid = np.arange(im.shape[0])
        z = int(valid[np.argmax(errors[valid])])
        lo, hi = np.percentile(im, [1, 99])
        xmin, xmax, ymin, ymax = slice_window(gt[z], p3[z], p2[z])
        displays.append(dict(image=im[z].copy(), gt=gt[z].copy(), p3=p3[z].copy(),
                             p2=p2[z].copy(), window=(xmin, xmax, ymin, ymax),
                             contrast=(lo, hi), case_id=k, z=z, rule=rule))
        selection.append(dict(
            case_id=k, z_index=z, selection_rule=rule,
            slice_rule='Largest mismatch slice with >=10% of maximal GT slice area',
            metrics_3d=seg['seg_unet3d'][k], metrics_2d=seg['seg_unet2d'][k]))
    # Fit each row to its crop aspect, with fixed physical gutters for headings
    # and volume metrics. All four panels have identical scale within each case.
    width, left, gap = WIDTH, .06, .065
    panel_width = (width - 2 * left - 3 * gap) / 4
    heights = [panel_width * (d['window'][3] - d['window'][2]) /
               (d['window'][1] - d['window'][0]) for d in displays]
    height = .50 + sum(h + .54 for h in heights) + .35
    fig = plt.figure(figsize=(width, height), layout='none')
    for j, title in enumerate(['MRI', 'Ground truth', '3D U-Net', '2D U-Net']):
        center = left + j * (panel_width + gap) + panel_width / 2
        fig.text(center / width, (height - .19) / height, title, ha='center',
                 fontsize=7, color=COLORS['ink'])
    cursor = height - .50
    for i, (d, image_height) in enumerate(zip(displays, heights)):
        k, z = d['case_id'], d['z']
        xmin, xmax, ymin, ymax = d['window']
        lo, hi = d['contrast']
        image_bottom = cursor - .17 - image_height
        fig.text(left / width, cursor / height, 'abcd'[i],
                 fontsize=8, fontweight='bold', color='#111111')
        fig.text((left + .16) / width, cursor / height,
                 f'{d["rule"]} · {k} · slice {z}', fontsize=7, color='#222222')
        for j, arr in enumerate([None, d['gt'], d['p3'], d['p2']]):
            x = left + j * (panel_width + gap)
            ax = fig.add_axes([x / width, image_bottom / height,
                               panel_width / width, image_height / height])
            ax.imshow(d['image'], cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
            if j == 1:
                overlay(ax, d['gt'].astype(bool), COLORS['gt'], alpha=.5)
            elif j in [2, 3]:
                prediction, target = arr.astype(bool), d['gt'].astype(bool)
                overlay(ax, prediction & target, COLORS['gt'])
                overlay(ax, prediction & ~target, COLORS['fp'])
                overlay(ax, ~prediction & target, COLORS['fn'])
            ax.set(xlim=(xmin - .5, xmax - .5), ylim=(ymax - .5, ymin - .5))
            ax.axis('off')
            if j in [2, 3]:
                m = seg['seg_unet3d' if j == 2 else 'seg_unet2d'][k]
                fig.text((x + panel_width / 2) / width, (image_bottom - .13) / height,
                         f"Dice {float(m['dice']):.3f} · HD95 {float(m['hd95']):.2f}",
                         color=COLORS['ink'], fontsize=7, ha='center', va='top')
        cursor = image_bottom - .37
    fig.legend(handles=[Patch(facecolor=COLORS['gt'], label='Foreground / true positive'),
                        Patch(facecolor=COLORS['fp'], label='False positive'),
                        Patch(facecolor=COLORS['fn'], label='False negative')],
               loc='lower center', bbox_to_anchor=(.5, .19 / height), ncol=3, fontsize=7)
    fig.text(.5, .06 / height, 'Full-volume Dice and HD95 (voxels).',
             ha='center', fontsize=7, color=COLORS['muted'])
    finish(fig, figures, 'segmentation_four_cases.png')
    save_json(results / 'segmentation_cases.json',
              dict(cases=selection, selected_after_test_for_display_only=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results')
    parser.add_argument('--include-case-images', action='store_true',
                        help='Render MRI examples only when authorized local raw data and saved masks are available')
    args = parser.parse_args()
    results = ROOT / args.results
    figures = results / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    classification_learning(figures)
    classification_roc(results, figures)
    classification_confusion(results, figures)
    classification_recall(results, figures)
    classification_tradeoffs(results, figures)
    classification_effect_sizes(results, figures)
    classification_distribution(figures)
    segmentation_learning(figures)
    seg, keys = segmentation_paired(results, figures)
    if args.include_case_images:
        segmentation_cases(results, figures, seg, keys)
    print('Formal publication figures generated:', figures)


if __name__ == '__main__':
    main()
