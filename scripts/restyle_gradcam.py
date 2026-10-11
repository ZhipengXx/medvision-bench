"""Render cached real Grad-CAM arrays without loading a model or recomputing CAMs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

from src.common.plot_style import apply_paper_style, save_figure


ROOT = Path(__file__).resolve().parents[1]


def render_gradcam(results, output_dir=None):
    results = Path(results)
    output_dir = Path(output_dir) if output_dir else results / "figures"
    manifest = json.loads((results / "gradcam_cases.json").read_text())
    records = manifest["cases"]
    if len(records) != 4 or sum(bool(r["correct"]) for r in records) != 2:
        raise ValueError("The figure requires the original two correct and two error cases")
    paths = {r["image_id"]: results / r["path"]
             for r in manifest.get("provenance", {}).get("assets", [])}
    images = []
    for record in records:
        asset_path = paths.get(record["image_id"], results / "gradcam_assets" / f"{record['image_id']}.npz")
        if not asset_path.is_file():
            raise FileNotFoundError(f"Missing real CAM array {asset_path}; first run scripts/gradcam.py")
        with np.load(asset_path, allow_pickle=False) as asset:
            rgb, heat = asset["rgb"].copy(), asset["heat"].copy()
        if rgb.shape != (224, 224, 3) or heat.shape != (224, 224):
            raise ValueError(f"Cached CAM/input geometry mismatch: {asset_path}")
        if not np.isfinite(heat).all() or heat.min() < 0 or heat.max() > 1 + 1e-6:
            raise ValueError(f"Invalid normalized CAM values: {asset_path}")
        images.append((rgb, heat))

    apply_paper_style()
    # Four cases form two rows of paired three-image blocks. Explicit physical
    # coordinates keep square inputs and an identical 183-mm vector footprint.
    width_mm, height_mm = 183, 94
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), layout="none")

    def position(x, y, width, height):
        return [x / width_mm, y / height_mm, width / width_mm, height / height_mm]

    def label(x, y, text, **kwargs):
        fig.text(x / width_mm, y / height_mm, text, color="#222222", **kwargs)

    cmap = "magma"
    for i, (record, (rgb, heat)) in enumerate(zip(records, images)):
        row, column = divmod(i, 2)
        left_mm = 3 + 92 * column
        header_mm = 90 - 42 * row
        image_bottom_mm = 54 - 42 * row
        status = "Correct" if record["correct"] else "Error"
        label(left_mm, header_mm, chr(97 + i), va="center", fontsize=8, fontweight="bold")
        label(left_mm + 5, header_mm,
              f"{status}   GT {record['gt']} → Pred {record['pred']}   p={record['confidence']:.3f}",
              va="center", fontsize=7)
        for j, title in enumerate(["Image", "Activation", "Overlay"]):
            image_left_mm = left_mm + 28.5 * j
            label(image_left_mm + 13.75, header_mm - 6, title,
                  ha="center", va="center", fontsize=7)
            ax = fig.add_axes(position(image_left_mm, image_bottom_mm, 27.5, 27.5))
            if j == 1:
                ax.imshow(heat, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
            else:
                ax.imshow(rgb, interpolation="bilinear")
                if j == 2:
                    ax.imshow(heat, cmap=cmap, vmin=0, vmax=1, alpha=0.42, interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.45)
                spine.set_edgecolor("#D9DEE1")

    colorbar_ax = fig.add_axes(position(140.5, 5, 39, 1.8))
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                           cax=colorbar_ax, orientation="horizontal", ticks=[0, 0.5, 1])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=7, length=2, pad=1.5, colors="#222222")
    label(136, 5.9, "Normalized activation", ha="right", va="center", fontsize=7)
    path = output_dir / "classification_gradcam_ce.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Matplotlib restores a None layout engine after each tight export. Disable
    # automatic layout in that context too, so vector backends keep this grid.
    with matplotlib.rc_context({"figure.constrained_layout.use": False, "figure.autolayout": False}):
        save_figure(fig, path, preserve_size=True)
    plt.close(fig)
    return [path.with_suffix(suffix) for suffix in [".png", ".pdf", ".svg"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print("Saved " + ", ".join(str(p) for p in render_gradcam(args.results, args.output_dir)))


if __name__ == "__main__":
    main()
