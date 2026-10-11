import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
from src.common.runtime import ROOT, load_config, save_json, sha256
from src.segmentation.data import read_volume
from src.segmentation.nifti import export_array
from src.segmentation.metrics import metrics

ap=argparse.ArgumentParser()
ap.add_argument('--selection',required=True)
args=ap.parse_args()
selection=json.loads(Path(args.selection).read_text())
cfg=load_config(ROOT/'configs/data.yaml')['segmentation']
split=json.loads((ROOT/'splits/la_split.json').read_text())
cases={r['case_id']:r for r in split['test']}
for case in selection['cases']:
    rid=case['case_id']
    row=cases[rid]
    im,gt=read_volume(ROOT/cfg['root']/row['path'],cfg,normalize=False)
    out=ROOT/'results/nifti'/rid
    out.mkdir(parents=True,exist_ok=True)
    arrays={'mri':im,'gt':gt}
    for model in ['seg_unet3d','seg_unet2d']:
        arrays[model]=np.load(ROOT/'results/segmentation'/model/(rid+'.npz'))['prediction']
    geometry={name:export_array(arr,out/(name+'.nii.gz'),cfg['spacing_zyx'],cfg.get('affine_xyz')) for name,arr in arrays.items()}
    geometry['source_sha256']=row['sha256']
    geometry['distance_unit']=cfg['distance_unit']
    geometry['spacing_source']=cfg['spacing_source']
    geometry['output_sha256']={p.name:sha256(p) for p in out.glob('*.nii.gz')}
    gt_back=np.asanyarray(nib.load(out/'gt.nii.gz').dataobj).transpose(2,1,0)
    geometry['roundtrip_metrics']={}
    for model in ['seg_unet3d','seg_unet2d']:
        pred_back=np.asanyarray(nib.load(out/(model+'.nii.gz')).dataobj).transpose(2,1,0)
        before=metrics(arrays[model],gt,cfg['spacing_zyx'])
        after=metrics(pred_back,gt_back,cfg['spacing_zyx'])
        for key in ['dice','jaccard','asd','hd95']:
            if not np.isclose(before[key],after[key]):raise ValueError('Round-trip metric changed')
        geometry['roundtrip_metrics'][model]={k:('Infinity' if isinstance(v,float) and not np.isfinite(v) else v) for k,v in after.items()}
    save_json(out/'geometry.json',geometry)
print('NIfTI round trips passed')
