# Data preparation

The repository retains measured predictions and numerical results. Training
requires locally obtained data; model weights and complete datasets are not
bundled. The four public dermoscopy examples and their activation caches have
separate data terms in [NOTICE](../NOTICE).

## Skin lesion classification

Obtain the **2018 Task 3** training, validation and test inputs and ground-truth
CSV files from the [official ISIC dataset page](https://challenge.isic-archive.com/data/).
That page specifies CC BY-NC 4.0 and the required dataset attribution. The
seven-class order is `MEL, NV, BCC, AKIEC, BKL, DF, VASC`; labels must be
one-hot. The recorded partitions contain 10,015, 193 and 1,512 images.

```text
data/isic/
├── ISIC2018_Task3_Training_Input/*.jpg
├── ISIC2018_Task3_Training_GroundTruth/
│   └── ISIC2018_Task3_Training_GroundTruth.csv
├── ISIC2018_Task3_Validation_Input/*.jpg
├── ISIC2018_Task3_Validation_GroundTruth/
│   └── ISIC2018_Task3_Validation_GroundTruth.csv
├── ISIC2018_Task3_Test_Input/*.jpg
├── ISIC2018_Task3_Test_GroundTruth/
│   └── ISIC2018_Task3_Test_GroundTruth.csv
└── ISIC2018_Task3_Training_LesionGroupings.csv  # optional grouping audit
```

The optional downloader fetches the official archives:

```bash
python scripts/download_data.py
python scripts/audit_data.py --config configs/data.yaml --task classification
```

The audit checks image decoding, labels, byte-identical duplicates and known
lesion groups. The recorded training set contains two duplicate image pairs;
no byte-identical cross-partition duplicates were found. The grouping file
does not identify validation/test lesions, so complete lesion independence
cannot be established from this audit.

## Left atrial segmentation

The measured comparison uses **34 anonymized, preprocessed volumes**:
14 source fitting volumes, partitioned into 11 training and 3 validation
cases, plus 20 held-out cases. Exact correspondence between this subset and
the complete public challenge release is unresolved. The official
[Cardiac Atlas challenge data page](https://www.cardiacatlas.org/atriaseg2018-challenge/atria-seg-data/)
is a starting point for obtaining authorized data; its raw files are not a
drop-in replacement for the measured HDF5 subset.

Prepare authorized volumes as HDF5 files containing `image` and `label` arrays
with matching three-dimensional shapes. Images must be finite; masks must use
0 for background and 1 for foreground. Put source fitting cases under `train`
and held-out cases under `test`:

```text
data/la/
├── train/*.h5  # source training pool; case-level validation is held out
└── test/*.h5   # held-out cases
```

Edit the keys, root, patterns and axis convention in
[`configs/data.yaml`](../configs/data.yaml), then run:

```bash
python scripts/audit_data.py --config configs/data.yaml --task segmentation
```

The historical conversion is **source-inferred XYZ**, transposed to internal
ZYX. HDF5 spatial metadata was absent: unit voxel spacing is used, with no
validated anatomical orientation. ASD and HD95 are therefore voxel distances.
Do not interpret them as millimetres or reuse this convention for new data
without checking its geometry. The report describes the inference source and
remaining uncertainty.

## Historical identifiers and new experiments

Public ISIC identifiers remain intact because they identify publicly released
benchmark images. MRI records use release-local `LA-TRxx`, `LA-VAxx` and
`LA-TExx` aliases. Their private source-name mapping is not included.
The aliases preserve correspondence between historical metric tables; they do
not provide access to source volumes.

Auditing a new dataset writes new split records and data-audit results. Run
new training with distinct run IDs and create a new evaluation manifest;
historical scores and manifests refer to the recorded experiment inputs.
Matching historical summaries is possible from the retained predictions and
metrics. Repeating MRI inference requires the exact authorized inputs and
corresponding weights, which this release does not distribute.
