"""Two evidence-led main figures from frozen predictions and case metrics.

The layout draws on Nature's research figure guide. The saved model outputs,
test case definitions and existing uncertainty calculations are untouched.
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

from src.common.runtime import ROOT
from src.common.plot_style import COLORS, RASTER_DPI, apply_paper_style, save_figure
from src.classification.data import CLASSES

WIDTH = 183 / 25.4
METHODS = [('cls_r50_ce', 'CE', 'ce', 'o'),
           ('cls_r50_wce', 'Weighted CE', 'wce', 's'),
           ('cls_r50_focal', 'Focal', 'focal', '^')]
SEG_METHODS = [('seg_unet3d', '3D U-Net', '3d', 'o'),
               ('seg_unet2d', '2D U-Net', '2d', 's')]
DIFFERENCE_CMAP = LinearSegmentedColormap.from_list(
    'change_blue_white_orange', ['#21648A', '#8FB4CA', '#FAFAF8', '#E4B296', '#B65328'])
GUIDES = [
    'https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/',
    'https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/',
    'https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scientific-visualization/SKILL.md',
]


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def heading(fig, ax, letter, label):
    position = ax.get_position()
    fig.text(position.x0 - .027, position.y1 + .022, letter,
             fontsize=8, fontweight='bold', ha='left', va='bottom')
    fig.text(position.x0, position.y1 + .023, label,
             fontsize=7, ha='left', va='bottom')


def classification_main(results, figures, sources):
    diagnostics = read_json(results / 'loss_diagnostics.json')
    per_class = {rid: {r['class_name']: r for r in read_csv(
        results / 'classification' / rid / 'per_class.csv')} for rid, *_ in METHODS}
    metrics = {rid: read_json(results / 'classification' / rid / 'metrics.json')
               for rid, *_ in METHODS}
    support = np.asarray([int(per_class['cls_r50_ce'][k]['support']) for k in CLASSES])
    if support.sum() != diagnostics['n']:
        raise ValueError('Figure support differs from the saved paired diagnostics')
    fig = plt.figure(figsize=(WIDTH, 5.42), layout='none')
    axa = fig.add_axes([.14, .555, .30, .35])
    axb = fig.add_axes([.65, .555, .305, .35])
    axc = fig.add_axes([.14, .115, .30, .315])
    axd = fig.add_axes([.65, .115, .30, .315])
    y = np.arange(len(CLASSES))
    recall_source = []
    for offset, (rid, name, color, marker) in zip([-.18, 0, .18], METHODS):
        values = np.asarray([float(per_class[rid][k]['recall']) for k in CLASSES]) * 100
        axa.scatter(values, y + offset, s=16, marker=marker,
                    color=COLORS[color], edgecolor='white', linewidth=.35,
                    label=name, zorder=3)
        for k, n, value in zip(CLASSES, support, values):
            recall_source.append(dict(method=name, class_name=k, support=int(n), recall_percent=value))
    axa.set(xlim=(0, 100), ylim=(6.65, -.65), xticks=[0, 25, 50, 75, 100],
            yticks=y, yticklabels=[f'{k}  ({n})' for k, n in zip(CLASSES, support)],
            xlabel='Recall (%)')
    axa.tick_params(axis='y', length=2)
    fig.legend(*axa.get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.54, .982), ncol=3, fontsize=7,
               handlelength=1, columnspacing=1.6)
    heading(fig, axa, 'a', 'Recall by class (test images)')
    contrast_source = []
    contrasts = [('accuracy', 'Accuracy'), ('macro_recall', 'Macro recall'), ('macro_f1', 'Macro F1')]
    for offset, (rid, name, color, marker) in zip([-.13, .13], METHODS[1:]):
        for yi, (key, label) in enumerate(contrasts):
            row = diagnostics['baseline_comparisons'][rid]['metrics'][key]
            estimate = 100 * row['delta']
            lower, upper = np.asarray(row['ci95']) * 100
            axb.errorbar(estimate, yi + offset,
                         xerr=np.array([[estimate - lower], [upper - estimate]]),
                         fmt=marker, markersize=3.6, color=COLORS[color],
                         capsize=2, capthick=.8, elinewidth=1.15,
                         label=name if yi == 0 else None)
            contrast_source.append(dict(method=name, metric=label,
                                        delta_pp=estimate, ci95_lower_pp=lower, ci95_upper_pp=upper))
    axb.axvline(0, linewidth=.65, color='#777777', linestyle=(0, (2, 2)), zorder=0)
    axb.set(xlim=(-7, 13), ylim=(2.5, -.55), xticks=[-5, 0, 5, 10],
            yticks=np.arange(3), yticklabels=[v for _, v in contrasts],
            xlabel='Difference from CE (percentage points)')
    heading(fig, axb, 'b', 'Paired differences (95% CI)')
    deltas = {rid: 100 * (np.asarray(metrics[rid]['confusion_row_normalized']) -
                         np.asarray(metrics['cls_r50_ce']['confusion_row_normalized']))
              for rid, *_ in METHODS[1:]}
    limit = 25
    confusion_source = []
    for ax, (rid, name, *_), letter in zip([axc, axd], METHODS[1:], 'cd'):
        values = deltas[rid]
        im = ax.imshow(values, cmap=DIFFERENCE_CMAP, norm=TwoSlopeNorm(0, -limit, limit),
                       aspect='auto', interpolation='nearest')
        for row in range(len(CLASSES)):
            for col in range(len(CLASSES)):
                value = values[row, col]
                ax.text(col, row, f'{value:+.1f}' if abs(value) >= .05 else '0',
                        ha='center', va='center', fontsize=6,
                        color='white' if abs(value) > 16 else '#222222',
                        fontweight='bold' if row == col else 'normal')
                confusion_source.append(dict(method=name, true_class=CLASSES[row],
                                             predicted_class=CLASSES[col], delta_pp=value))
        ax.set(xticks=np.arange(len(CLASSES)), yticks=np.arange(len(CLASSES)),
               xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted class', ylabel='True class')
        ax.tick_params(axis='both', length=0, labelsize=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # The diagonal remains identifiable without relying on heatmap colour.
        for i in range(len(CLASSES)):
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor='#424242', linewidth=.45))
        heading(fig, ax, letter, f'{name} minus CE')
    cax = fig.add_axes([.16, .043, .77, .010])
    bar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-25, -10, 0, 10, 25])
    bar.outline.set_linewidth(.4)
    bar.ax.tick_params(labelsize=6, length=2, pad=1.5)
    bar.set_label('Difference from CE (percentage points)', fontsize=6.5, labelpad=1)
    save_figure(fig, figures / 'classification_main.png', preserve_size=True)
    plt.close(fig)
    write_csv(sources / 'classification_main_recalls.csv', recall_source)
    write_csv(sources / 'classification_main_paired_contrasts.csv', contrast_source)
    write_csv(sources / 'classification_main_confusion_changes.csv', confusion_source)
    return dict(n_images=int(support.sum()), class_supports=dict(zip(CLASSES, support.tolist())),
                panel_a='Per-class recall from frozen per_class.csv; no new uncertainty added',
                panel_b=diagnostics['uncertainty_protocol'],
                panels_c_d='Difference of saved row-normalized confusion matrices, expressed in percentage points',
                interval_limits=diagnostics['interpretation_limits'])


def segmentation_main(results, figures, sources):
    cases = {rid: {r['case_id']: r for r in read_csv(
        results / 'segmentation' / rid / 'per_case_metrics.csv')} for rid, *_ in SEG_METHODS}
    ids = sorted(cases['seg_unet3d'])
    if len(ids) != 20 or set(ids) != set(cases['seg_unet2d']):
        raise ValueError('Expected all 20 paired segmentation test cases')
    units = {r['distance_unit'] for method in cases.values() for r in method.values()}
    if units != {'voxel'}:
        raise ValueError(f'Unexpected surface-distance units: {units}')
    marked = max(ids, key=lambda case_id: float(cases['seg_unet3d'][case_id]['hd95']))
    marked_index = ids.index(marked)
    # Identical deterministic horizontal offsets for a patient's two estimates
    # keep pairings visible without changing any ordinate.
    offsets = np.random.default_rng(42).permutation(np.linspace(-.125, .125, len(ids)))
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.72), layout='none')
    fig.subplots_adjust(left=.095, right=.975, bottom=.13, top=.91, hspace=.42, wspace=.31)
    source = []
    for ax, key, label, letter in zip(axes.flat, ['dice', 'jaccard', 'asd', 'hd95'],
                                     ['Dice', 'Jaccard', 'ASD (voxels)', 'HD95 (voxels)'], 'abcd'):
        vals = [np.asarray([float(cases[rid][k][key]) for k in ids]) for rid, *_ in SEG_METHODS]
        if not all(np.isfinite(v).all() for v in vals):
            raise ValueError('Nonfinite case metrics must be disclosed before plotting')
        for i, jitter in enumerate(offsets):
            ax.plot([jitter, 1 + jitter], [vals[0][i], vals[1][i]],
                    color='#B6BDC1', alpha=.58, linewidth=.65, zorder=1)
        for xi, (values, (rid, name, color, marker)) in enumerate(zip(vals, SEG_METHODS)):
            ax.scatter(xi + offsets, values, color=COLORS[color], marker=marker,
                       s=15, edgecolors='white', linewidths=.35, zorder=3)
            ax.scatter([xi + offsets[marked_index]], [values[marked_index]], s=35,
                       facecolors='none', edgecolors='#222222', marker=marker,
                       linewidths=.8, zorder=4)
            mean, sd = values.mean(), values.std(ddof=0)
            ax.plot([xi - .21, xi + .21], [mean, mean], color='#222222',
                    linewidth=1.15, solid_capstyle='butt', zorder=5)
            # Cohort spread, consistent with the saved aggregate metrics.
            # Draw the SD outside the patient swarm to keep every point visible.
            ax.errorbar(xi + .25, mean, yerr=sd, fmt='none', color='#222222',
                        capsize=2, capthick=.7, elinewidth=.7, zorder=5)
            for k, value in zip(ids, values):
                source.append(dict(case_id=k, method=name, metric=key,
                                   value=value, distance_unit='voxel', marked_hd95_case=k == marked))
        ax.set(xlim=(-.4, 1.4), xticks=[0, 1], xticklabels=['3D U-Net', '2D U-Net'], ylabel=label)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if key in ['dice', 'jaccard']:
            ax.set_ylim(.5 if key == 'dice' else .4, 1)
            ax.set_yticks([.5, .6, .7, .8, .9, 1] if key == 'dice' else [.4, .55, .7, .85, 1])
        else:
            maximum = max(v.max() for v in vals)
            # The descriptive mean ± SD whisker can extend slightly below zero.
            # Preserve the complete arithmetic interval rather than clipping it.
            ax.set_ylim(-maximum * .035, maximum * 1.10)
        ax.tick_params(axis='x', length=0)
        heading(fig, ax, letter, label.split(' (')[0])
        if key == 'hd95':
            ax.annotate(marked if len(marked) <= 10 else f'{marked[:5]}…', (offsets[marked_index], vals[0][marked_index]),
                        xytext=(21, 4), textcoords='offset points', fontsize=6,
                        ha='left', va='bottom', color='#222222',
                        arrowprops=dict(arrowstyle='-', color='#555555', linewidth=.6))
    handles = [Line2D([], [], color='#B6BDC1', linewidth=.8, label='Matched test volumes (n = 20)'),
               Line2D([], [], color='#222222', linewidth=1.15, label='Mean ± cohort SD'),
               Line2D([], [], color='#222222', linestyle='none', marker='o', markerfacecolor='none',
                      markersize=4, label=f'Largest 3D HD95: {marked if len(marked) <= 10 else marked[:5] + chr(8230)}')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .034),
               ncol=3, fontsize=6.5, handlelength=1.6, columnspacing=1.5)
    save_figure(fig, figures / 'segmentation_main.png', preserve_size=True)
    plt.close(fig)
    write_csv(sources / 'segmentation_main_paired_metrics.csv', source)
    return dict(n_paired_volumes=len(ids), case_ids=ids, distance_unit='voxel',
                dot_definition='One frozen full-volume metric per test case and method',
                linkage='Paired case IDs; identical deterministic horizontal offsets in all panels',
                summary='Black horizontal bars are arithmetic means; offset capped lines show population SD (ddof=0) across the 20 cases, not confidence intervals',
                highlighted_case=marked, highlight_rule='Largest 3D U-Net HD95 across all 20 test cases',
                omitted_cases=[])


def classification_precision_recall(results, figures, sources):
    """Class-wise operating points; these are not threshold-swept PR curves."""
    rows = {rid: {r['class_name']: r for r in read_csv(
        results / 'classification' / rid / 'per_class.csv')} for rid, *_ in METHODS}
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 4.1), layout='none')
    fig.subplots_adjust(left=.072, right=.985, bottom=.16, top=.92,
                        wspace=.38, hspace=.53)
    source = []
    for ax, class_name, letter in zip(axes.flat, CLASSES, 'abcdefg'):
        n = int(rows['cls_r50_ce'][class_name]['support'])
        if any(int(rows[rid][class_name]['support']) != n for rid, *_ in METHODS):
            raise ValueError('The classification operating points must share test support')
        ax.plot([0, 100], [0, 100], color='#BFC7CB', linewidth=.65,
                linestyle=(0, (2, 2)), zorder=0)
        for rid, name, color, marker in METHODS:
            recall = 100 * float(rows[rid][class_name]['recall'])
            precision = 100 * float(rows[rid][class_name]['precision'])
            ax.scatter(recall, precision, marker=marker, s=27, color=COLORS[color],
                       edgecolor='white', linewidth=.4, zorder=3)
            source.append(dict(method=name, class_name=class_name, support=n,
                               recall_percent=recall, precision_percent=precision))
        ax.set(xlim=(-1.5, 102.5), ylim=(-1.5, 102.5), xticks=[0, 50, 100], yticks=[0, 50, 100])
        ax.set_aspect('equal', adjustable='box')
        heading(fig, ax, letter, f'{class_name} (n = {n})')
    key = axes.flat[-1]
    key.axis('off')
    handles = [Line2D([], [], linestyle='none', marker=marker, markersize=4,
                      color=COLORS[color], label=name) for _, name, color, marker in METHODS]
    key.legend(handles=handles, loc='upper left', bbox_to_anchor=(.02, .96),
               fontsize=7, labelspacing=.7)
    key.text(.08, .38, 'One operating point\nper class and method.\nDashed line: precision\nequals recall.',
             transform=key.transAxes, fontsize=7, va='top', linespacing=1.35)
    fig.supxlabel('Recall (%)', x=.53, y=.05, fontsize=7)
    fig.supylabel('Precision (%)', x=.013, y=.56, fontsize=7)
    save_figure(fig, figures / 'classification_precision_recall_by_class.png', preserve_size=True)
    plt.close(fig)
    write_csv(sources / 'classification_precision_recall_by_class.csv', source)
    return dict(n_images=sum(int(rows['cls_r50_ce'][k]['support']) for k in CLASSES),
                input='Frozen per-class test metrics at argmax operating points',
                uncertainty='No intervals added; class supports displayed',
                interpretation='Precision and recall are different quantities; dashed equality line is a visual reference, not chance performance')


def classification_validation_test_alignment(results, figures, sources):
    """Compare existing selected checkpoints; never reselect on test evidence."""
    records, inputs = [], []
    for rid, name, color, marker in METHODS:
        for suffix, label, folder in [('', 'Original', results / 'classification'),
                                      ('_sensitivity', 'Exploratory', results / 'sensitivity/classification')]:
            run = rid + suffix
            metadata_path = ROOT / 'outputs' / run / 'metadata.json'
            history_path = ROOT / 'outputs' / run / 'history.json'
            metrics_path = folder / run / 'metrics.json'
            metadata, history = read_json(metadata_path), read_json(history_path)
            if not metadata.get('complete') or not metadata.get('formal'):
                raise ValueError(f'Validation/test alignment requires a complete formal run: {run}')
            selected = next(r for r in history if r['epoch'] == metadata['best_epoch'])
            metrics = read_json(metrics_path)
            records.append(dict(run_id=run, method=name, color=color, marker=marker,
                                protocol=label, epoch=metadata['best_epoch'],
                                validation=selected, test=metrics))
            inputs.extend([metadata_path, history_path, metrics_path])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.55), layout='none')
    fig.subplots_adjust(left=.26, right=.985, bottom=.20, top=.90, wspace=.28)
    source = []
    for ax, key, title, letter in zip(axes, ['accuracy', 'macro_f1'],
                                    ['Accuracy', 'Macro F1'], 'ab'):
        for y, record in enumerate(records):
            validation = 100 * float(record['validation']['val_' + key])
            test = 100 * float(record['test'][key])
            color = COLORS[record['color']]
            ax.plot([test, validation], [y, y], color='#BCC4C8', linewidth=.85, zorder=1)
            ax.scatter(validation, y, marker=record['marker'], s=30, facecolors='white',
                       edgecolors=color, linewidths=.9, zorder=3)
            ax.scatter(test, y, marker=record['marker'], s=14, color=color,
                       edgecolor='white', linewidth=.3, zorder=4)
            source.append(dict(run_id=record['run_id'], method=record['method'],
                               protocol=record['protocol'], selected_epoch=record['epoch'], metric=key,
                               validation_percent=validation, test_percent=test,
                               test_minus_validation_pp=test - validation))
        ax.set(xlim=(45, 100), xticks=[50, 60, 70, 80, 90, 100], ylim=(5.6, -.6),
               yticks=np.arange(6), xlabel='Score (%)')
        ax.axhline(1.5, color='#E2E6E8', linewidth=.5)
        ax.axhline(3.5, color='#E2E6E8', linewidth=.5)
        ax.tick_params(axis='y', length=0)
        if ax is axes[0]:
            ax.set_yticklabels([f"{r['method']} · {r['protocol']} · e{r['epoch']}" for r in records], fontsize=6.7)
        else:
            ax.set_yticklabels([])
        heading(fig, ax, letter, title)
    handles = [Line2D([], [], linestyle='none', marker='o', markerfacecolor='white',
                      markeredgecolor='#222222', markersize=4.5, label='Validation at selected epoch'),
               Line2D([], [], linestyle='none', marker='o', color='#222222', markersize=3.2,
                      label='Frozen test evaluation')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.59, .025),
               ncol=2, fontsize=7, handlelength=1, columnspacing=1.4)
    save_figure(fig, figures / 'classification_validation_test_alignment.png', preserve_size=True)
    plt.close(fig)
    write_csv(sources / 'classification_validation_test_alignment.csv', source)
    return dict(n_selected_runs=len(records), selection='Existing metadata best_epoch; no new checkpoint selection',
                units='percentage points', uncertainty='No new intervals or significance tests',
                interpretation='Validation and test refer to different dataset partitions; differences include sampling and class-composition effects',
                input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})


def paired_segmentation_cases(results):
    cases = {rid: {r['case_id']: r for r in read_csv(
        results / 'segmentation' / rid / 'per_case_metrics.csv')} for rid, *_ in SEG_METHODS}
    ids = sorted(cases['seg_unet3d'])
    if len(ids) != 20 or set(ids) != set(cases['seg_unet2d']):
        raise ValueError('The derivative analyses require all 20 matched test volumes')
    return cases, ids


def segmentation_metric_summary(results, figures, sources):
    cases, ids = paired_segmentation_cases(results)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.05), layout='none')
    fig.subplots_adjust(left=.15, right=.975, bottom=.19, top=.92, hspace=.61, wspace=.37)
    source = []
    jitters = np.random.default_rng(42).permutation(np.linspace(-.11, .11, len(ids)))
    for ax, key, label, letter in zip(axes.flat, ['dice', 'jaccard', 'asd', 'hd95'],
                                     ['Dice', 'Jaccard', 'ASD (voxels)', 'HD95 (voxels)'], 'abcd'):
        for y, (rid, name, color, marker) in enumerate(SEG_METHODS):
            values = np.asarray([float(cases[rid][k][key]) for k in ids])
            mean, sd = values.mean(), values.std(ddof=0)
            ax.scatter(values, y + jitters, marker=marker, s=11, color=COLORS[color],
                       edgecolor='white', linewidth=.3, zorder=2)
            ax.errorbar(mean, y - .25, xerr=sd, fmt=marker, markersize=3.2,
                        color='#222222', capsize=2.5, elinewidth=.8, capthick=.7, zorder=4)
            source.append(dict(method=name, metric=key, n_cases=len(ids), mean=mean,
                               population_sd=sd, ddof=0, distance_unit='voxel'))
        ax.set(yticks=[0, 1], yticklabels=['3D U-Net', '2D U-Net'], ylim=(1.35, -.55), xlabel=label)
        ax.tick_params(axis='y', length=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        maximum = max(float(cases[rid][k][key]) for rid, *_ in SEG_METHODS for k in ids)
        ax.set_xlim((.4, 1) if key in ['dice', 'jaccard'] else
                    (-maximum * .035, maximum * 1.06))
        heading(fig, ax, letter, label.split(' (')[0])
    handles = [Line2D([], [], linestyle='none', marker='o', color=COLORS['3d'], markersize=3,
                      label='Individual volume (n = 20 per method)'),
               Line2D([], [], color='#222222', linewidth=.8, marker='o', markersize=3,
                      label='Mean ± population SD (ddof = 0)')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .04),
               ncol=2, fontsize=7, handlelength=1.5)
    save_figure(fig, figures / 'segmentation_metric_summary.png', preserve_size=True)
    plt.close(fig)
    write_csv(sources / 'segmentation_metric_summary.csv', source)
    return dict(n_test_volumes=len(ids), excluded_cases=[], ddof=0,
                uncertainty='Population SD describes cohort spread, not estimator uncertainty or confidence intervals',
                raw_points='All saved case metrics are displayed without winsorization, filtering, or smoothing')


def segmentation_volume_bias(results, figures, sources):
    cases, ids = paired_segmentation_cases(results)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.9), layout='none')
    fig.subplots_adjust(left=.105, right=.975, bottom=.17, top=.92, hspace=.50, wspace=.31)
    source, max_count, max_bias, cdf_lines = [], 0, 0, []
    for rid, name, color, marker in SEG_METHODS:
        gt = np.asarray([int(cases[rid][k]['gt_voxels']) for k in ids])
        pred = np.asarray([int(cases[rid][k]['predicted_voxels']) for k in ids])
        reference = np.asarray([int(cases['seg_unet3d'][k]['gt_voxels']) for k in ids])
        if np.any(gt <= 0) or not np.array_equal(reference, gt):
            raise ValueError('Volume bias requires positive, identical ground-truth volumes')
        bias = 100 * (pred - gt) / gt
        dice = np.asarray([float(cases[rid][k]['dice']) for k in ids])
        hd95 = np.asarray([float(cases[rid][k]['hd95']) for k in ids])
        for ax, x, y in [(axes[0, 0], gt / 100000, pred / 100000),
                         (axes[0, 1], bias, dice), (axes[1, 0], np.abs(bias), hd95)]:
            ax.scatter(x, y, color=COLORS[color], marker=marker, s=19,
                       edgecolor='white', linewidth=.45, alpha=.88, zorder=3)
        ordered = np.sort(np.abs(bias))
        line, = axes[1, 1].step(np.r_[0, ordered], np.r_[0, np.arange(1, len(ids) + 1) / len(ids)],
                                where='post', color=COLORS[color], linewidth=1.2,
                                linestyle='-' if rid == 'seg_unet3d' else '--', label=name)
        cdf_lines.append(line)
        max_count = max(max_count, gt.max() / 100000, pred.max() / 100000)
        max_bias = max(max_bias, np.abs(bias).max())
        for i, k in enumerate(ids):
            source.append(dict(case_id=k, method=name, predicted_voxels=int(pred[i]),
                               gt_voxels=int(gt[i]), signed_volume_bias_percent=bias[i],
                               absolute_volume_bias_percent=abs(bias[i]), dice=dice[i],
                               hd95=hd95[i], distance_unit='voxel'))
    limits = (0, max_count * 1.07)
    for line in cdf_lines:
        x, y = line.get_data()
        line.set_data(np.r_[x, max_bias * 1.08], np.r_[y, 1])
    axes[0, 0].plot(limits, limits, color='#AAB5BB', linestyle=(0, (2, 2)), linewidth=.7)
    axes[0, 0].set(xlim=limits, ylim=limits, xlabel=r'GT foreground ($10^5$ voxels)',
                   ylabel=r'Predicted foreground ($10^5$ voxels)')
    axes[0, 1].axvline(0, color='#AAB5BB', linestyle=(0, (2, 2)), linewidth=.7)
    axes[0, 1].set(xlim=(-max_bias * 1.08, max_bias * 1.08), ylim=(.5, 1),
                   xlabel='Signed volume bias (%)', ylabel='Dice')
    axes[1, 0].set(xlim=(0, max_bias * 1.08), ylim=(0, 75),
                   xlabel='Absolute volume bias (%)', ylabel='HD95 (voxels)')
    axes[1, 1].set(xlim=(0, max_bias * 1.08), ylim=(0, 1.03),
                   xlabel='Absolute volume bias (%)', ylabel='Cumulative case fraction',
                   yticks=[0, .25, .5, .75, 1])
    for ax, letter, label in zip(axes.flat, 'abcd', ['Foreground volume', 'Bias and overlap',
                                                  'Bias and boundary error', 'Absolute bias distribution']):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        heading(fig, ax, letter, label)
    handles = [Line2D([], [], color=COLORS[color], marker=marker, markersize=4,
                      linewidth=1.1, linestyle='-' if rid == 'seg_unet3d' else '--',
                      label=name) for rid, name, color, marker in SEG_METHODS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .026), ncol=2,
               fontsize=7, handlelength=2, columnspacing=2.1)
    save_figure(fig, figures / 'segmentation_volume_bias.png', preserve_size=True)
    plt.close(fig)
    write_csv(sources / 'segmentation_volume_bias.csv', source)
    return dict(n_paired_volumes=len(ids), case_ids=ids, excluded_cases=[],
                bias_definition='100 * (predicted_voxels - gt_voxels) / gt_voxels',
                volume_units='foreground voxels, not physical volume',
                empirical_distribution='Right-continuous empirical CDF of absolute percent bias; n = 20 per method',
                interpretation='Descriptive test-set diagnostic; no regression, association test, or new model selection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    args.results = (ROOT / args.results).resolve()
    figures = args.results / 'figures'
    sources = args.results / 'figure_source_data'
    figures.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    cls = classification_main(args.results, figures, sources)
    seg = segmentation_main(args.results, figures, sources)
    cls_pr = classification_precision_recall(args.results, figures, sources)
    cls_alignment = classification_validation_test_alignment(args.results, figures, sources)
    seg_summary = segmentation_metric_summary(args.results, figures, sources)
    seg_volume = segmentation_volume_bias(args.results, figures, sources)
    inputs = [args.results / 'loss_diagnostics.json']
    inputs += [args.results / 'classification' / rid / name
               for rid, *_ in METHODS for name in ['metrics.json', 'per_class.csv']]
    inputs += [args.results / 'segmentation' / rid / 'per_case_metrics.csv' for rid, *_ in SEG_METHODS]
    provenance = dict(style_references=GUIDES, figure_width_mm=183,
                      font='Liberation Sans (Arial-compatible)', formats=['png', 'pdf', 'svg'],
                      raster_dpi=RASTER_DPI, vector_text='Editable SVG; embedded TrueType PDF',
                      formal_results_modified=False, classification_main=cls, segmentation_main=seg,
                      classification_precision_recall_by_class=cls_pr,
                      classification_validation_test_alignment=cls_alignment,
                      segmentation_metric_summary=seg_summary, segmentation_volume_bias=seg_volume,
                      input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in inputs})
    (figures / 'nature_figure_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Rendered two main and four diagnostic figures; source data:', sources)


if __name__ == '__main__':
    main()
