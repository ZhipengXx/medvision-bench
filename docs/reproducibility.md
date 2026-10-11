# Reproducibility

There are two useful entry points: regenerate analyses from retained results,
or train and evaluate new models using authorized local inputs. Both commands
run from the repository root in a Python 3.12 environment.

## Inspect and rebuild retained evidence

```bash
python scripts/verify_project.py
python -m pytest tests -q
python scripts/diagnose_classification.py
python scripts/make_figures.py --results results
python scripts/make_nature_figures.py --results results
python scripts/make_sensitivity_figures.py
python scripts/make_method_diagram.py
python scripts/restyle_gradcam.py --results results
python scripts/make_readme_assets.py
```

These steps use saved numerical outcomes, histories and the four licensed
dermoscopy activation caches. They need neither a training GPU nor model
weights. The method schematic uses original vector icons to illustrate input
modalities. MRI case imagery is not part of the public release; its local
renderer remains available through `make_figures.py --include-case-images`
when authorized source volumes and prediction arrays are present.

The release checker independently reconstructs classification confusion,
accuracy and classwise metrics from 1,512 predictions for each of six models,
and mean/population SD from every segmentation case. It checks evidence
consistency; it does not rerun model inference.

## Environment

[`requirements.txt`](../requirements.txt) pins portable package versions.
[`requirements.lock.txt`](../requirements.lock.txt) records the original CUDA
12.6 build, including packages present during the measured experiments:

```bash
python -m pip install -r requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126
```

The measured training used a GeForce RTX 3090, seed 42 and deterministic cuDNN
settings. Elapsed times reflect shared-device workloads. Generic software
versions and resource measurements are retained; machine identifiers,
process listings and device topology are omitted.

## New training and evaluation

Prepare and audit local data as described in [the data guide](data.md).
Historical run directories are already populated with metadata. Create five
local configurations with fresh run IDs and a local experiment list:

```bash
python - <<'PY'
from pathlib import Path
import yaml

directory = Path('configs/local')
directory.mkdir(parents=True, exist_ok=True)
names = ['cls_r50_ce', 'cls_r50_wce', 'cls_r50_focal', 'seg_unet2d', 'seg_unet3d']
paths = []
for name in names:
    config = yaml.safe_load(Path(f'configs/{name}.yaml').read_text())
    config['run_id'] = name + '_reproduction'
    path = directory / (name + '.yaml')
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    paths.append(str(path))
(directory / 'experiments.yaml').write_text(yaml.safe_dump({'experiments': paths}))
PY

CUDA_VISIBLE_DEVICES=0 python scripts/train_classification.py --config configs/local/cls_r50_ce.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/train_classification.py --config configs/local/cls_r50_wce.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/train_classification.py --config configs/local/cls_r50_focal.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/train_segmentation.py --config configs/local/seg_unet2d.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/train_segmentation.py --config configs/local/seg_unet3d.yaml

python scripts/freeze_evaluation.py \
  --experiments configs/local/experiments.yaml \
  --output splits/local_evaluation_manifest.json
python scripts/evaluate_all.py \
  --manifest splits/local_evaluation_manifest.json \
  --output-root results/reproduction
```

Classification uses one shared ImageNet-1K V2 initialization and a seed-42
linear head. Inverse-frequency CE weights come from training counts only;
focal uses gamma 2, alpha 1. Validation accuracy selects the primary
classification checkpoint, with macro F1 breaking ties. Segmentation selects
mean foreground Dice on complete validation volumes. Gaussian probability
fusion reconstructs the full supplied grid, with no postprocessing.

Runs refuse implicit overwrites. Add `--smoke` for separate debug output or
resume explicitly with `--resume outputs/<run_id>/last.pt`. Checkpoints
include optimizer, scheduler, mixed-precision and RNG state. Model binaries
stay local and are ignored by Git.

The public historical manifests record original checkpoint/configuration
hashes. Identifier aliasing and metadata neutralization changed the release
copies of certain inputs; `published_input_hashes` records those copies.
Historical manifests are documentary evidence rather than the manifest for a
new training run. The freeze command creates an independent, executable
manifest after the new runs finish.

## Statistical scope

Classification comparisons use 10,000 paired, true-class-stratified bootstrap
replicates with a fixed seed. Confidence intervals condition on the fitted
models and observed class composition; they exclude training-seed variation
and unavailable patient/lesion clusters. Segmentation summaries assign equal
weight to each of 20 complete volumes and use population SD (`ddof=0`).
Plots distinguish those SD intervals from confidence intervals.

The three supplemental classification runs changed checkpoint selection
after the primary test results were inspected. They are exploratory and are
stored separately. They do not replace the primary comparison or establish
confirmatory improvement.

## Native manuscript build

Install XeLaTeX with `fontspec`, `unicode-math`, `amsmath`, `booktabs`,
`graphicx`, `caption`, `float`, `enumitem`, `hyperref`, `titlesec` and fonts
TeX Gyre Termes, Liberation Sans and Latin Modern Math. On Ubuntu, the
following package set supplies these components:

```bash
sudo apt-get install texlive-xetex texlive-latex-extra texlive-fonts-recommended texlive-plain-generic lmodern fonts-texgyre fonts-liberation fonts-lmodern
python scripts/build_report.py --results results --output report/report.pdf
python scripts/verify_project.py
```

The builder uses native inline/display mathematics, compiles three passes
with shell escape disabled, and records missing glyphs, unresolved references
and layout checks in `report/typesetting_checks.json`. The prebuilt PDF,
editable LaTeX and Markdown source are included.

## Package a clean snapshot

```bash
python scripts/package_project.py
```

The archive contains source, documentation, manuscript and release evidence,
with a SHA-256 manifest and ZIP integrity check. Raw datasets, model binaries,
environments and temporary build files are excluded. Dataset-derived
dermoscopy examples retain the separate terms stated in [NOTICE](../NOTICE).
