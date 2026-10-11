# Scientific sources and attribution

The implementation is original project code. The following papers and primary
documentation describe the datasets, model families, interpretability method
and metric conventions. References do not imply endorsement or independent
validation of this project's measured results.

| Component | Primary source |
|:--|:--|
| ISIC 2018 challenge | Codella et al., [ISIC 2018 overview](https://arxiv.org/abs/1902.03368) |
| HAM10000 | Tschandl, Rosendahl & Kittler, [Scientific Data, 2018](https://doi.org/10.1038/sdata.2018.161) |
| ResNet | He et al., [Deep residual learning](https://arxiv.org/abs/1512.03385) |
| Focal loss | Lin et al., [Focal loss for dense object detection](https://arxiv.org/abs/1708.02002) |
| U-Net | Ronneberger, Fischer & Brox, [U-Net](https://arxiv.org/abs/1505.04597) |
| 3D U-Net | Çiçek et al., [3D U-Net](https://arxiv.org/abs/1606.06650) |
| Grad-CAM | Selvaraju et al., [Gradient-based localization](https://arxiv.org/abs/1610.02391) |
| Left atrial benchmark | Xiong et al., [Medical Image Analysis, 2021](https://doi.org/10.1016/j.media.2020.101832) |
| Symmetric ASD | [MedPy assd documentation](https://loli.github.io/medpy/generated/medpy.metric.binary.assd.html) |
| HD95 | [MedPy hd95 documentation](https://loli.github.io/medpy/generated/medpy.metric.binary.hd95.html) |
| Array-order inference | [UA-MT preprocessing source](https://github.com/yulequan/UA-MT/blob/88ed29ad794f877122e542a7fa9505a76fa83515/code/dataloaders/la_heart_processing.py) and [pynrrd reading API](https://pynrrd.readthedocs.io/en/stable/reference/reading.html) |

The UA-MT preprocessing signature supports the operational XYZ convention;
it does not recover absent physical geometry or prove the complete conversion
history of the measured HDF5 subset. No third-party implementation or
checkpoint is vendored. ImageNet-1K V2 weights are obtained through
[torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).

## Data terms

The [official ISIC data page](https://challenge.isic-archive.com/data/)
provides downloads, dataset attribution and CC BY-NC terms. [NOTICE](NOTICE)
identifies the retained dermoscopy examples and their adaptations. MRI source
volumes and imagery are excluded from this release; [the data guide](docs/data.md)
explains the subset and provenance limitations. The
[Cardiac Atlas source page](https://www.cardiacatlas.org/atriaseg2018-challenge/atria-seg-data/)
describes the broader public challenge.

## Visualization and software

The quantitative plots follow the
[Nature research figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
for panel typography and vector exports. Layout guidance was also consulted
from [scientific-visualization](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scientific-visualization/SKILL.md)
and [scientific-schematics](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scientific-schematics/SKILL.md).
Diagrams and README artwork are drawn as original code-defined vectors;
quantitative figures come from retained numerical evidence.

Dependency versions are recorded in `requirements.lock.txt`. The report
contains full bibliographic references and detailed Methods. No downloaded
reference-paper PDFs, external software binaries, raw datasets or model
weights are included.
