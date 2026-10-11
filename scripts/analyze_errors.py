"""Descriptive full-mask diagnostics only; never modify predictions/postprocessing."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import ndimage
from src.common.runtime import ROOT,load_config,save_json
from src.segmentation.data import read_volume

cfg=load_config(ROOT/'configs/data.yaml')['segmentation']
split=json.loads((ROOT/'splits/la_split.json').read_text())
paths={r['case_id']:r for r in split['test']}
selection=json.loads((ROOT/'results/segmentation_cases.json').read_text())['cases']
records=[]
for case in selection:
    _,gt=read_volume(ROOT/cfg['root']/paths[case['case_id']]['path'],cfg)
    record=dict(case_id=case['case_id'],display_z=case['z_index'],models={})
    for rid in ['seg_unet3d','seg_unet2d']:
        pred=np.load(ROOT/'results/segmentation'/rid/(case['case_id']+'.npz'))['prediction']
        fp=(pred==1)&(gt==0);fn=(pred==0)&(gt==1)
        labelled,n=ndimage.label(pred,ndimage.generate_binary_structure(3,1))
        sizes=np.bincount(labelled.ravel())[1:]
        record['models'][rid]=dict(false_positive_voxels=int(fp.sum()),false_negative_voxels=int(fn.sum()),
            relative_voxel_volume_bias=float((int(pred.sum())-int(gt.sum()))/max(1,int(gt.sum()))),
            components=n,largest_component_fraction=float((sizes.max() if len(sizes) else 0)/max(1,pred.sum())),
            display_slice_fp=int(fp[case['z_index']].sum()),display_slice_fn=int(fn[case['z_index']].sum()),
            max_fp_z=int(fp.sum((1,2)).argmax()))
    records.append(record)
save_json(ROOT/'results/case_error_diagnostics.json',records)
print('Full-mask diagnostics saved without modifying any prediction')
