"""Compare clean checkpoint reevaluation against original predictions and metrics."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.common.runtime import ROOT,save_json

checks=[]
for rid in ['cls_r50_ce','cls_r50_wce','cls_r50_focal','seg_unet3d','seg_unet2d']:
    kind='classification' if rid.startswith('cls') else 'segmentation'
    original=ROOT/'results'/kind/rid;clean=ROOT/'results/reproduction'/kind/rid
    a=json.loads((original/'metrics.json').read_text());b=json.loads((clean/'metrics.json').read_text())
    keys=['accuracy','macro_f1','macro_recall','macro_auroc'] if kind=='classification' else [m+'_mean' for m in ['dice','jaccard','asd','hd95']]
    if not all(np.isclose(a[k],b[k],rtol=0,atol=1e-12) for k in keys):raise ValueError(f'{rid}: metrics differ')
    if kind=='classification':
        if (original/'predictions.csv').read_bytes()!=(clean/'predictions.csv').read_bytes():raise ValueError(f'{rid}: probabilities differ')
    else:
        for p in original.glob('*.npz'):
            if not np.array_equal(np.load(p)['prediction'],np.load(clean/p.name)['prediction']):raise ValueError(f'{rid}: mask differs')
    checks.append(dict(run_id=rid,metrics_match=True,predictions_match=True))
save_json(ROOT/'results/reproduction_checks.json',checks)
print('All five clean-directory predictions and core metrics match')
