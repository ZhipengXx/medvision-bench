"""Fixed balanced training-image debugging; never used as formal performance."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from src.common.runtime import ROOT,load_config,seed_all,save_json
from src.classification.data import ISICDataset
from src.classification.model import build_model

seed_all(42);torch.set_num_threads(4)
split=json.loads((ROOT/'splits/isic_split.json').read_text())
rows=[next(r for r in split['train'] if r['label']==k) for k in range(7)]
data=load_config(ROOT/'configs/data.yaml')['classification']
ds=ISICDataset(rows,ROOT/data['root'],False)
x=torch.stack([ds[i][0] for i in range(7)]).cuda();y=torch.tensor([r['label'] for r in rows]).cuda()
model=build_model(True).cuda().train()
opt=torch.optim.AdamW(model.parameters(),lr=1e-4);scaler=torch.amp.GradScaler('cuda')
losses=[]
for i in range(100):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):
        logits=model(x);loss=F.cross_entropy(logits,y)
    scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(loss.item())
with torch.no_grad():
    model.eval();accuracy=float((model(x).argmax(1)==y).float().mean())
record=dict(debug_only=True,image_ids=[r['image_id'] for r in rows],steps=100,initial_loss=losses[0],
            final_loss=losses[-1],accuracy=accuracy,passed=accuracy==1. and losses[-1]<losses[0]*.2)
save_json(ROOT/'results/overfit_classification.json',record);print(record)
if not record['passed']:raise SystemExit('Classification overfit check failed')
