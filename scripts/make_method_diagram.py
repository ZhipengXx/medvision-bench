"""Editable study and U-Net schematics, drawn from the implemented protocols.

Run with ``python scripts/make_method_diagram.py``.  No results are
changed and no remote image-generation service is used.  The schematics use
fixed physical dimensions, editable text, and the project's publication style.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Ellipse, PathPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.plot_style import COLORS, RASTER_DPI, apply_paper_style, save_figure

OUT = ROOT / 'results' / 'figures'
INK = '#222222'
EDGE = '#A3ADB3'
LIGHT = '#F6F8F9'
BLUE_LIGHT = '#EDF5FA'
ORANGE_LIGHT = '#FFF2E8'
MM = 25.4


def canvas(width, height):
    fig = plt.figure(figsize=(width / MM, height / MM), layout=None)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(0, height), aspect='equal')
    ax.axis('off')
    return fig, ax


def label(ax, x, y, s, *, size=7, weight='normal', ha='left', va='center',
          color=INK, linespacing=1.18):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
                   color=color, linespacing=linespacing)


def box(ax, x, y, width, height, text, *, face='white', edge=EDGE,
        color=INK, size=7, weight='normal', accent=None):
    patch = FancyBboxPatch((x, y), width, height,
                          boxstyle='round,pad=0,rounding_size=1.1',
                          facecolor=face, edgecolor=edge, linewidth=0.65,
                          zorder=3)
    ax.add_patch(patch)
    if accent:
        ax.add_patch(Rectangle((x, y + 1.1), 0.8, height - 2.2,
                               color=accent, linewidth=0, zorder=4))
    label(ax, x + width / 2, y + height / 2, text, size=size,
          weight=weight, ha='center', color=color)
    return patch


def arrow(ax, points, *, color='#56636B', width=0.7, dashed=False,
          head=4.3, zorder=2):
    """Orthogonal polyline with an arrow only on its final segment."""
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                linewidth=width, linestyle='--' if dashed else '-',
                solid_capstyle='round', zorder=zorder)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1],
                                 arrowstyle='-|>', mutation_scale=head,
                                 color=color, linewidth=width,
                                 linestyle='--' if dashed else '-',
                                 shrinkA=0, shrinkB=0, zorder=zorder))


def line(ax, points, *, color='#56636B', width=0.7):
    ax.plot(*zip(*points), color=color, linewidth=width,
            solid_capstyle='round', zorder=2)


def panel_heading(ax, letter, title, y):
    label(ax, 4, y, letter, size=8, weight='bold')
    label(ax, 10, y, title, size=7, weight='bold')


def modality_icon(ax, modality, x, y, width, height):
    """Draw explicitly illustrative vector icons; no dataset image is read."""
    if modality == 'dermoscopy':
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0,rounding_size=1.6',
                                   facecolor='#EADAC8', edgecolor='#CDB9A5', linewidth=0.6))
        vertices = [(x+8, y+6), (x+5, y+11), (x+7, y+17), (x+13, y+18),
                    (x+19, y+15), (x+20, y+9), (x+15, y+5), (x+8, y+6)]
        codes = [PlotPath.MOVETO] + [PlotPath.CURVE3] * 6 + [PlotPath.CLOSEPOLY]
        ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor='#846047',
                               edgecolor='#68432D', linewidth=0.7))
        ax.add_patch(Ellipse((x+13, y+11), 8, 6, angle=-20, facecolor='#61422F', edgecolor='none', alpha=0.65))
    else:
        for offset in [2, 1, 0]:
            ox, oy = x + 2*offset, y + 1.5*offset
            ax.add_patch(FancyBboxPatch((ox, oy), width-4, height-3,
                                       boxstyle='round,pad=0,rounding_size=1',
                                       facecolor='#E5EBEF' if offset else '#34424B',
                                       edgecolor='#9AA7AF', linewidth=0.6))
        ax.add_patch(Ellipse((x+11, y+10), 15, 13, facecolor='#7D8B95', edgecolor='none'))
        ax.add_patch(Ellipse((x+11, y+10), 9, 7, facecolor='#B6C1C9', edgecolor='none'))
        ax.add_patch(Ellipse((x+13, y+9), 4, 4.5, facecolor=COLORS['3d'], edgecolor='white', linewidth=0.5))


def export(fig, name):
    """Refuse output when a label falls outside the fixed physical canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for ax in fig.axes:
        for text in ax.texts:
            bounds = text.get_window_extent(renderer)
            if (bounds.x0 < -0.1 or bounds.y0 < -0.1 or
                    bounds.x1 > fig.bbox.width + 0.1 or
                    bounds.y1 > fig.bbox.height + 0.1):
                clipped.append(text.get_text())
    if clipped:
        raise ValueError(f'Labels clipped in {name}: {clipped}')
    save_figure(fig, OUT / f'{name}.png', preserve_size=True)


def read_sources():
    cls = json.loads((ROOT / 'splits/isic_split.json').read_text())
    seg = json.loads((ROOT / 'splits/la_split.json').read_text())
    if [len(cls[k]) for k in ('train', 'val', 'test')] != [10015, 193, 1512]:
        raise ValueError('Update diagram labels for the changed ISIC split.')
    if [len(seg[k]) for k in ('train', 'val', 'test')] != [11, 3, 20]:
        raise ValueError('Update diagram labels for the changed MRI split.')
    configs = {name: json.loads((ROOT / f'outputs/{name}/config.json').read_text())
               for name in ('cls_r50_ce', 'cls_r50_wce', 'cls_r50_focal',
                            'seg_unet2d', 'seg_unet3d')}
    for name in ('cls_r50_ce', 'cls_r50_wce', 'cls_r50_focal'):
        if configs[name]['batch_size'] != 16:
            raise ValueError('Diagram assumes classification batch size 16.')
    for name, dim, patch, batch in [('seg_unet2d', 2, [112, 112], 16),
                                   ('seg_unet3d', 3, [80, 112, 112], 1)]:
        cfg = configs[name]
        if (cfg['dimensions'], cfg['patch'], cfg['batch_size'],
                cfg['base_channels'], cfg['overlap']) != (dim, patch, batch, 16, 0.5):
            raise ValueError('Update diagram for the changed segmentation protocol.')
    return {
        'modality_icons': {'source': 'Matplotlib vector shapes generated by this script',
                           'role': 'Illustrative modality icons; not dataset images or anatomical evidence'},
        'configs': {k: f'outputs/{k}/config.json' for k in configs},
    }


def make_overview():
    fig, ax = canvas(183, 158)
    panel_heading(ax, 'a', 'Loss reweighting in seven-class dermoscopy', 152)

    # Cohort illustrations identify the modality; role labels are separate.
    modality_icon(ax, 'dermoscopy', 5, 119, 25, 21)
    label(ax, 17.5, 142.6, 'Illustrative input', ha='center', size=6.0, color='#5B666D')
    label(ax, 17.5, 115, 'ISIC 2018', ha='center', weight='bold')
    label(ax, 17.5, 110.5, 'Training, n = 10,015', ha='center', size=6.7)
    label(ax, 39, 148.5, 'IMAGE PREPARATION', size=6.1, color='#5B666D')
    label(ax, 74, 148.5, 'COMMON BACKBONE', size=6.1, color='#5B666D')
    label(ax, 109, 148.5, 'LOSS ALTERNATIVES', size=6.1, color='#5B666D')
    box(ax, 39, 126, 27, 18, '224 × 224 pixels\nMild crop + flips\nColour jitter\nImageNet scaling', face=LIGHT, size=6.8)
    box(ax, 74, 126, 27, 18, 'ResNet-50\nImageNet-1K V2\n7-class linear head\nSame initial weights', face=LIGHT, size=6.8)
    arrow(ax, [(30.5, 135), (39, 135)])
    arrow(ax, [(66, 135), (74, 135)])
    for key, text, y in [('ce', 'Cross-entropy', 139),
                         ('wce', 'Weighted CE (1/freq)', 129),
                         ('focal', 'Focal (γ = 2)', 119)]:
        box(ax, 109, y, 28, 8, text, accent=COLORS[key], size=6.8)
        arrow(ax, [(101, 135), (105, 135), (105, y + 4), (109, y + 4)],
              color=COLORS[key], width=0.8)
        line(ax, [(137, y + 4), (141, y + 4), (141, 135)],
             color=COLORS[key], width=0.8)
    label(ax, 123, 115.2, 'Three independent runs', ha='center', size=6.2)
    box(ax, 146, 126, 33, 20, 'Validation, n = 193\nAccuracy-selected\nMacro F1 tie-break\nFreeze each checkpoint', face=LIGHT, size=6.8)
    arrow(ax, [(141, 135), (146, 135)])
    box(ax, 4, 91, 27, 12, 'Test, n = 1,512\nOfficial held-out split', size=6.8)
    box(ax, 39, 89, 27, 21, 'Deterministic resize\n224 × 224 pixels\nImageNet scaling', face=LIGHT, size=6.8)
    box(ax, 74, 89, 63, 21, 'Frozen CE / weighted CE / focal models\nSoftmax → class argmax\nOne view per image', face=LIGHT, size=6.8)
    box(ax, 146, 89, 33, 23, 'Accuracy / macro F1\nMacro recall\nClass-specific errors\nPaired uncertainty\nGrad-CAM examples', face=LIGHT, size=6.8)
    arrow(ax, [(31, 97), (39, 97)])
    arrow(ax, [(66, 99.5), (74, 99.5)])
    arrow(ax, [(137, 99.5), (146, 99.5)])
    arrow(ax, [(162.5, 126), (162.5, 117.5), (100, 117.5), (100, 110)],
          color='#56636B')

    ax.plot([4, 179], [84.5, 84.5], color='#D8DFE3', linewidth=0.6)
    panel_heading(ax, 'b', 'Spatial context in left-atrial MRI segmentation', 79.5)
    modality_icon(ax, 'mri', 5, 45, 25, 22)
    label(ax, 17.5, 69.7, 'Illustrative input', ha='center', size=6.0, color='#5B666D')
    label(ax, 17.5, 40.5, 'LGE MRI', ha='center', weight='bold')
    label(ax, 17.5, 36.4, 'Training, n = 11', ha='center', size=6.8)
    box(ax, 39, 52, 27, 21, 'XYZ → ZYX\nWhole-volume z-score\nCase-uniform sampling\n50% foreground centres', face=LIGHT, size=6.4)
    box(ax, 74, 59, 31, 15, '2D U-Net\n112 × 112 patches\nBatch size 16',
        face=ORANGE_LIGHT, accent=COLORS['2d'], size=6.8)
    box(ax, 74, 35, 31, 15, '3D U-Net\n80 × 112 × 112 patches\nBatch size 1',
        face=BLUE_LIGHT, accent=COLORS['3d'], size=6.8)
    box(ax, 113, 46, 24, 20, 'CE + soft Dice\nMatched width\nAdamW + cosine\n≤12,000 updates', face=LIGHT, size=6.6)
    box(ax, 146, 50, 33, 23, 'Validation, n = 3\nComplete volumes\nMean foreground Dice\nFreeze each checkpoint', face=LIGHT, size=6.8)
    arrow(ax, [(30.5, 60), (39, 60)])
    arrow(ax, [(66, 62.5), (70, 62.5), (70, 66.5), (74, 66.5)], color=COLORS['2d'])
    arrow(ax, [(66, 62.5), (70, 62.5), (70, 42.5), (74, 42.5)], color=COLORS['3d'])
    arrow(ax, [(105, 66.5), (109, 66.5), (109, 56), (113, 56)], color=COLORS['2d'])
    arrow(ax, [(105, 42.5), (109, 42.5), (109, 56), (113, 56)], color=COLORS['3d'])
    arrow(ax, [(137, 56), (146, 56)])
    label(ax, 179, 76, '14 source cases: 11 train + 3 validation',
          ha='right', size=6.2)
    box(ax, 4, 16, 27, 13, 'Test, n = 20\nFull MRI volumes', size=6.8)
    box(ax, 39, 14, 27, 18, 'Whole-volume z-score\nComplete voxel grid\nNo label-based ROI', face=LIGHT, size=6.5)
    box(ax, 74, 14, 63, 18, '2D slice-wise / 3D volumetric inference\nGaussian probability fusion, 50% overlap\nClass argmax on the original grid', face=LIGHT, size=6.8)
    box(ax, 146, 12, 33, 23, 'Case-equal metrics\nDice / Jaccard\nASD / HD95 (voxels)\nPaired case differences', face=LIGHT, size=6.8)
    arrow(ax, [(31, 22.5), (39, 22.5)])
    arrow(ax, [(66, 23), (74, 23)])
    arrow(ax, [(137, 23), (146, 23)])
    arrow(ax, [(162.5, 50), (162.5, 39), (124, 39), (124, 32)])
    label(ax, 4, 5.7, 'Primary protocol shown; post-test sensitivity analyses are reported separately.',
          size=6.5, color='#56636B')
    export(fig, 'method_overview')
    plt.close(fig)


def make_unet():
    fig, ax = canvas(183, 122)
    panel_heading(ax, 'a', 'The same encoder–decoder topology in two or three dimensions', 116)
    xs = [16, 34, 52, 70, 88, 106, 124, 142, 160]
    ys = [89, 78, 67, 56, 45, 56, 67, 78, 89]
    channels = [16, 32, 64, 128, 256, 128, 64, 32, 16]
    w, h = 14, 12
    for idx, (x, y, ch) in enumerate(zip(xs, ys, channels)):
        box(ax, x - w / 2, y - h / 2, w, h, f'{ch}\nConv × 2',
            face=BLUE_LIGHT if idx < 5 else ORANGE_LIGHT, size=6.8)
    box(ax, 1.5, 84, 6, 10, '1\nin', face=LIGHT, size=6.2)
    box(ax, 169, 84, 12, 10, '2\nlogits', face=LIGHT, size=6.8)
    arrow(ax, [(7.5, 89), (9, 89)], width=0.7, head=3.8)
    arrow(ax, [(167, 89), (169, 89)], width=0.7, head=3.8)
    for idx in range(8):
        start, end = (xs[idx] + w / 2, ys[idx]), (xs[idx + 1] - w / 2, ys[idx + 1])
        arrow(ax, [start, end], color=COLORS['3d'] if idx < 4 else COLORS['2d'], head=4)
    # Skip connections are drawn above the nodes at their matching resolution.
    # Each route ends at the decoder node and denotes concatenation, not addition.
    for idx in range(4):
        source, target = idx, 8 - idx
        skip_y = ys[idx] + h / 2 + 3
        arrow(ax, [(xs[source], ys[source] + h / 2),
                   (xs[source], skip_y), (xs[target], skip_y),
                   (xs[target], ys[target] + h / 2)],
              color='#929DA5', width=0.6, head=3.7)
    label(ax, 90, 101.8, 'Skip features concatenated at the matching scale',
          ha='center', size=6.8, color='#5B666D')
    label(ax, 37, 47, 'Max-pooling, stride 2', ha='center', size=6.6,
          color=COLORS['3d'])
    label(ax, 144, 47, 'Transposed convolution, stride 2', ha='center', size=6.6,
          color=COLORS['2d'])
    label(ax, 4, 37.7, 'Block: [convolution → affine instance normalization → ReLU] × 2. Channels are shown inside each block.',
          size=6.8)
    label(ax, 4, 33.2, 'Kernels are 3 × 3 in 2D and 3 × 3 × 3 in 3D. A final 1 × 1 (× 1) convolution produces two logits.',
          size=6.8)
    panel_heading(ax, 'b', 'Spatial scales for the training patches', 25.3)
    columns = [5, 40, 73, 106, 139, 171]
    headers = ['Input', '16 channels', '32 channels', '64 channels', '128 channels', '256 channels']
    ax.plot([4, 179], [21.7, 21.7], color='#68737B', linewidth=0.7)
    for x, text in zip(columns, headers):
        label(ax, x, 18.2, text, ha='left' if x == 5 else 'center', size=6.6, weight='bold')
    ax.plot([4, 179], [15.6, 15.6], color='#B7C0C6', linewidth=0.55)
    sizes = [('2D', ['112 × 112', '56 × 56', '28 × 28', '14 × 14', '7 × 7']),
             ('3D', ['80 × 112 × 112', '40 × 56 × 56', '20 × 28 × 28', '10 × 14 × 14', '5 × 7 × 7'])]
    for y, (text, values) in zip([12.3, 7.8], sizes):
        label(ax, 5, y, text, size=6.8, weight='bold', color=COLORS[text.lower()])
        for x, value in zip(columns[1:], values):
            label(ax, x, y, value, ha='center', size=6.6)
    ax.plot([4, 179], [4.6, 4.6], color='#68737B', linewidth=0.7)
    export(fig, 'unet_architecture')
    plt.close(fig)


def main():
    apply_paper_style()
    # Fixed-position schematics deliberately bypass automatic layout resizing.
    plt.rcParams['figure.constrained_layout.use'] = False
    source = read_sources()
    make_overview()
    make_unet()
    provenance = {
        'purpose': 'Research workflow schematics using scientific figure preparation guidance',
        'rendering': {'backend': 'Matplotlib', 'font': 'Liberation Sans (Arial-compatible)',
                      'text_size_pt': '6.1–7, panel letters 8', 'png_dpi': RASTER_DPI,
                      'pdf_fonttype': 42, 'svg_fonttype': 'none',
                      'fixed_canvas_mm': {'method_overview': [183, 158], 'unet_architecture': [183, 122]}},
        'source': source,
        'implementation_sources': [
            'src/classification/data.py', 'src/classification/model.py',
            'src/classification/losses.py', 'scripts/train_classification.py',
            'src/segmentation/data.py', 'src/segmentation/model.py',
            'src/segmentation/losses.py', 'src/segmentation/inference.py',
            'scripts/train_segmentation.py', 'scripts/evaluate_all.py',
            'splits/isic_split.json', 'splits/la_split.json', 'configs/data.yaml',
            'splits/evaluation_manifest.json'],
        'design_sources': [
            {'name': 'scientific-visualization skill v1.4',
             'url': 'https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scientific-visualization/SKILL.md',
             'used': 'fixed physical dimensions, exact vector relationships, editable fonts, provenance and manual review'},
            {'name': 'scientific-schematics skill v1.10',
             'url': 'https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scientific-schematics/SKILL.md',
             'used': 'layout principles only; its external raster-generation pipeline was not used'},
            {'name': 'Nature final guide to authors',
             'url': 'https://www.nature.com/documents/Final_guide_to_authors.pdf',
             'used': '183 mm width, restrained sans-serif labels, upright panel letters, editable vector artwork'}],
        'software_design_reference': {'title': 'Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents',
                                      'url': 'https://arxiv.org/abs/2609.00065'},
        'data_changes': 'none',
        'schematic_semantics': {
            'classification': 'Three independent trainings have a common initialization, not a jointly trained ensemble; held-out images enter only frozen inference.',
            'segmentation': 'Two independent dimension-matched U-Nets use a common objective and selection protocol; selected frozen models process entire test grids.',
            'unet': 'Five encoder scales, four max pools, four transposed convolutions, concatenation skips, two-logit output.'},
        'captions': {
            'method_overview': 'Study design and evaluation. a, Three independently trained ResNet-50 classifiers share an ImageNet-1K V2 initialization and image preparation, differing only in their training objective. Each checkpoint is selected on the official validation set by accuracy, with macro F1 as a tie-breaker, then evaluated on 1,512 held-out dermoscopy images. b, Left-atrial MRI training uses 11 of the 14 supplied source volumes, with three retained for validation. Case-uniform, partly foreground-centred sampling supplies either 2D patches or 3D patches to U-Nets with matched channel widths. Checkpoints are selected by mean foreground Dice on complete validation volumes. Frozen models reconstruct the full grids of 20 test volumes using Gaussian probability fusion. ASD and HD95 are expressed in voxel units because physical image spacing is unavailable. The modality icons are illustrative vector drawings rather than dataset images or anatomical evidence. Supplementary sensitivity runs performed after inspection of the primary test results are outside this primary design.',
            'unet_architecture': 'U-Net architecture and patch scales. a, Both segmentation models have five encoder scales, four max-pooling operations, and four decoder scales with transposed-convolution upsampling. At each decoder scale, the upsampled features are concatenated with encoder features at the same resolution, then processed by a double-convolution block. Each convolution is followed by affine instance normalization and ReLU. The architecture uses 2D or 3D operations throughout, with base width 16 and a two-logit output. b, Spatial sizes after each encoder block for the actual 2D and 3D training patches. The decoder restores the original patch resolution. General inputs are padded to multiples of 16 before the forward pass and the final logits are cropped to the original size.'},
        'alt_text': {
            'method_overview': 'Two independent left-to-right pipelines show training and held-out evaluation separately. Dermoscopy uses shared initial ResNet-50 weights and three loss alternatives; MRI compares 2D and 3D U-Nets. Validation chooses each checkpoint; test inputs feed frozen inference and metric calculation only.',
            'unet_architecture': 'A U-shaped encoder-decoder has channel widths 16, 32, 64, 128 and 256 on the encoder, then 128, 64, 32 and 16 on the decoder. Four matching-scale skips are concatenated. A table lists all spatial dimensions of the 2D and 3D patches.'}}
    (OUT / 'method_diagram_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Wrote method_overview and unet_architecture in PNG, PDF and SVG.')


if __name__ == '__main__':
    main()
