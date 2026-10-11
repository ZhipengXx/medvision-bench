"""Synthetic checks verify numerical behavior independently of benchmark scores."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from src.classification.losses import FocalLoss, inverse_frequency
from src.classification.metrics import metrics as cm
from src.segmentation.metrics import metrics as sm
from src.segmentation.data import read_volume
from src.segmentation.inference import infer_array, predict_volume, starts
from src.segmentation.model import UNet
from src.segmentation.losses import ce_dice
from src.segmentation.nifti import export_array


def test_inverse_weights_and_focal():
    counts=[3,6,9,12,15,18,21]
    weights=inverse_frequency(counts)
    assert torch.isclose(weights[0]/weights[1],torch.tensor(2.))
    x=torch.randn(20,7,requires_grad=True)
    y=torch.randint(7,(20,))
    assert torch.allclose(FocalLoss(0)(x,y),F.cross_entropy(x,y))
    extreme=torch.tensor([[10000.,-10000.,0.,0.,0.,0.,0.]],requires_grad=True)
    loss=FocalLoss()(extreme,torch.tensor([1]));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(extreme.grad).all()


def test_all_seven_classes_enter_macro():
    p=np.tile([0.,1.,0.,0.,0.,0.,0.],(7,1))
    m,rows=cm(np.arange(7),p)
    assert rows[0]['f1']==0 and rows[0]['support']==1
    assert m['macro_f1']==pytest.approx((2/8)/7)
    assert len(rows)==7 and len(m['confusion_counts'])==7


def test_full_volume_metrics_and_spacing():
    a=np.zeros((16,16,16),bool);a[3:7,3:7,3:7]=True
    perfect=sm(a,a)
    assert perfect['dice']==perfect['jaccard']==1 and perfect['hd95']==0
    b=np.zeros_like(a);b[10:14,10:14,10:14]=True
    assert sm(a,b)['dice']==sm(a,b)['jaccard']==0
    shifted=np.roll(a,1,axis=0)
    assert sm(a,shifted,(2,2,2))['asd']==pytest.approx(2*sm(a,shifted)['asd'])
    assert sm(a,shifted,(2,2,2))['hd95']==pytest.approx(2*sm(a,shifted)['hd95'])
    assert sm(np.zeros_like(a),np.zeros_like(a))['dice']==1
    assert np.isinf(sm(a,np.zeros_like(a))['hd95'])


def test_explicit_axes_and_nifti(tmp_path):
    xyz=np.arange(17*19*21,dtype=np.float32).reshape(17,19,21)
    mask=(xyz%5==0).astype(np.uint8)
    p=tmp_path/'case.h5'
    with h5py.File(p,'w') as f:f['image']=xyz;f['label']=mask
    cfg=dict(axis_order='XYZ',axis_source='Synthetic test definition',image_key='image',label_key='label')
    im,gt=read_volume(p,cfg,False)
    normalized,_=read_volume(p,cfg,True)
    assert normalized.dtype==np.float32
    assert normalized.mean()==pytest.approx(0,abs=1e-6)
    assert np.array_equal(im,xyz.transpose(2,1,0))
    assert np.array_equal(gt[7],mask[:,:,7].T)
    assert export_array(gt,tmp_path/'gt.nii.gz')['roundtrip_exact']
    with pytest.raises(ValueError):read_volume(p,{**cfg,'axis_source':None})


class IdentityMask(torch.nn.Module):
    def forward(self,x):return torch.cat([-10*x,10*x],dim=1)


def test_sliding_coverage_padding_and_slice_reconstruction():
    rng=np.random.default_rng(1)
    im=rng.choice([-1.,1.],(23,25,27)).astype(np.float32)
    model=IdentityMask()
    pred=infer_array(model,im,(32,16,16),torch.device('cpu'),batch_size=2)
    assert pred.shape==im.shape and np.array_equal(pred,im>0)
    cfg=dict(dimensions=2,patch=[16,16],overlap=0.5,inference_batch=4)
    pred=predict_volume(model,im,cfg,torch.device('cpu'))
    assert np.array_equal(pred,im>0)


@pytest.mark.parametrize('dim',[2,3])
def test_unet_backward_and_odd_sizes(dim):
    torch.set_num_threads(2)
    model=UNet(dim,2)
    shape=(33,35) if dim==2 else (32,33,35)
    x=torch.randn(1,1,*shape)
    y=torch.randint(2,(1,*shape))
    logits=model(x)
    assert logits.shape==(1,2,*shape)
    loss,_,_=ce_dice(logits,y);loss.backward()
    assert torch.isfinite(loss) and model.out.weight.grad.abs().sum()>0
