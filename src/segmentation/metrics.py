import numpy as np
from medpy.metric.binary import assd, hd95


def metrics(pred, gt, spacing=(1, 1, 1)):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.shape != gt.shape:
        raise ValueError('Original-grid shape mismatch')
    a, b = int(pred.sum()), int(gt.sum())
    if a == 0 and b == 0:
        return dict(dice=1., jaccard=1., asd=0., hd95=0., empty='both')
    if a == 0 or b == 0:
        # JSON uses "Infinity" string; CSV uses inf. Failed cases stay in means.
        return dict(dice=0., jaccard=0., asd=float('inf'), hd95=float('inf'), empty='one')
    intersection = int((pred & gt).sum())
    return dict(dice=2 * intersection / (a + b), jaccard=intersection / (a + b - intersection),
                asd=float(assd(pred, gt, voxelspacing=spacing, connectivity=1)),
                hd95=float(hd95(pred, gt, voxelspacing=spacing, connectivity=1)), empty='none')
