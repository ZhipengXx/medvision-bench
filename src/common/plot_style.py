"""Shared publication style and vector exports for saved experimental results."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RASTER_DPI = 450

COLORS = {
    'ce': '#0072B2', 'wce': '#D55E00', 'focal': '#009E73',
    'train': '#0072B2', 'validation': '#D55E00',
    '3d': '#0072B2', '2d': '#D55E00',
    'ink': '#263238', 'muted': '#6A747B', 'grid': '#E6EBED',
    'gt': '#009E73', 'fp': '#D55E00', 'fn': '#56B4E9',
}
CLASS_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00',
                '#CC79A7', '#56B4E9', '#6A747B']
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    'paper_blues', ['#FAFCFD', '#DCEEF6', '#81BED6', '#3485AB', '#124B6B'])


def apply_paper_style():
    plt.rcParams.update({
        # Liberation Sans is metrically compatible with Arial and is available
        # in the reproducible Linux environment. All lettering remains text.
        'font.family': ['Liberation Sans', 'DejaVu Sans'], 'font.size': 7,
        'mathtext.fontset': 'dejavusans', 'text.usetex': False,
        'axes.labelsize': 7, 'axes.titlesize': 7, 'axes.titleweight': 'normal',
        'axes.labelcolor': '#222222', 'axes.edgecolor': '#444444',
        'axes.linewidth': 0.6, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.axisbelow': True,
        'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
        'xtick.color': '#222222', 'ytick.color': '#222222',
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': 0.65, 'ytick.major.width': 0.65,
        'grid.color': COLORS['grid'], 'grid.linewidth': 0.45,
        'grid.alpha': 0.0,
        'lines.linewidth': 1.2, 'lines.markersize': 3.5,
        'legend.fontsize': 7, 'legend.frameon': False,
        'legend.handlelength': 2.1, 'legend.columnspacing': 1.4,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.dpi': RASTER_DPI,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.04,
        'figure.constrained_layout.w_pad': 0.04,
    })


def panel_label(ax, letter, text=None):
    ax.set_title(text or '', loc='left', pad=9, color='#222222', fontsize=7)
    ax.text(-.12, 1.04, letter, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8, fontweight='bold', fontstyle='normal', color='#111111')


def save_figure(fig, path, *, preserve_size=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix('')
    # Resolve the layout once, then keep identical geometry in all formats.
    # Matplotlib's print_figure restores its previous layout engine after saving;
    # disabling the rc default also prevents a restored None from re-enabling it.
    with plt.rc_context({'figure.constrained_layout.use': False,
                         'figure.autolayout': False}):
        fig.canvas.draw()
        fig.set_layout_engine('none')
        for suffix in ('.png', '.pdf', '.svg'):
            fig.savefig(stem.with_suffix(suffix), dpi=RASTER_DPI,
                        bbox_inches=None if preserve_size else 'tight',
                        pad_inches=0 if preserve_size else 0.045, facecolor='white')
