"""Independently recompute formal metrics from saved probabilities/3D masks."""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.common.runtime import ROOT, load_config, save_json
from src.classification.data import CLASSES
from src.classification.metrics import metrics as cls_metrics
from src.segmentation.data import read_volume
from src.segmentation.metrics import metrics as seg_metrics

checks = []
for folder in sorted((ROOT/'results/classification').glob('cls_*')):
    rows=list(csv.DictReader((folder/'predictions.csv').open()))
    m,_=cls_metrics([int(x['gt']) for x in rows],[[float(x['p_'+c]) for c in CLASSES] for x in rows])
    original=json.loads((folder/'metrics.json').read_text())
    for k in ['accuracy','macro_f1','macro_recall','weighted_f1','macro_auroc']:
        assert np.isclose(m[k],original[k],atol=1e-7), (folder,k)
    checks.append(dict(run_id=folder.name, predictions=len(rows), passed=True))
split=json.loads((ROOT/'splits/la_split.json').read_text())
cfg=load_config(ROOT/'configs/data.yaml')['segmentation']
for folder in sorted((ROOT/'results/segmentation').glob('seg_*')):
    rows={r['case_id']:r for r in csv.DictReader((folder/'per_case_metrics.csv').open())}
    for case in split['test']:
        _,gt=read_volume(ROOT/cfg['root']/case['path'],cfg)
        pred=np.load(folder/(case['case_id']+'.npz'))['prediction']
        m=seg_metrics(pred,gt,cfg['spacing_zyx'])
        for k in ['dice','jaccard','asd','hd95']:
            assert np.isclose(m[k],float(rows[case['case_id']][k])), (folder,case,k)
    checks.append(dict(run_id=folder.name, predictions=len(rows), passed=True))
save_json(ROOT/'results/reaggregation_checks.json',checks)
print(checks)
