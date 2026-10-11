"""Debug fixed training examples; outputs are excluded from formal experiments."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.common.runtime import ROOT,load_config,seed_all,save_json
from src.segmentation.data import read_volume,pad_volume
from src.segmentation.model import UNet
from src.segmentation.losses import ce_dice

ap=argparse.ArgumentParser();ap.add_argument('--dimensions',type=int,choices=[2,3],required=True);args=ap.parse_args()
seed_all(42);torch.set_num_threads(4)
cfg=load_config(ROOT/f'configs/seg_unet{args.dimensions}d.yaml')
data=load_config(ROOT/'configs/data.yaml')['segmentation']
row=json.loads((ROOT/'splits/la_split.json').read_text())['train'][0]
im,gt=read_volume(ROOT/data['root']/row['path'],data)
center=np.argwhere(gt).mean(0).astype(int)
if args.dimensions==2:
    im,gt=im[center[0]],gt[center[0]];center=center[1:]
im,gt=pad_volume(im,gt,cfg['patch'])
start=[int(np.clip(c-n//2,0,s-n)) for c,n,s in zip(center,cfg['patch'],im.shape)]
sl=tuple(slice(a,a+n) for a,n in zip(start,cfg['patch']))
x=torch.from_numpy(im[sl][None,None]).cuda();y=torch.from_numpy(gt[sl][None].astype(np.int64)).cuda()
model=UNet(args.dimensions,cfg['base_channels']).cuda()
opt=torch.optim.AdamW(model.parameters(),lr=.001)
scaler=torch.amp.GradScaler('cuda')
losses=[]
for i in range(100):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):
        logits=model(x);loss,_,_=ce_dice(logits,y)
    scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(loss.item())
with torch.no_grad():
    pred=model(x).argmax(1);dice=float(2*((pred==1)&(y==1)).sum()/((pred==1).sum()+(y==1).sum()))
record=dict(debug_only=True,case_id=row['case_id'],dimensions=args.dimensions,steps=100,
            initial_loss=losses[0],final_loss=losses[-1],fixed_patch_dice=dice,losses=losses,
            passed=losses[-1]<losses[0]*.5 and dice>.9)
save_json(ROOT/f'results/overfit_seg{args.dimensions}d.json',record)
print({k:v for k,v in record.items() if k!='losses'},flush=True)
if not record['passed']:raise SystemExit('Overfit check failed')
