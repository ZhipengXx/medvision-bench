"""Explicit HDF5 adapter; no guessing axes from dimension sizes."""
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def read_volume(path, cfg, normalize=True):
    order = cfg['axis_order'].upper()
    if sorted(order) != ['X', 'Y', 'Z'] or not cfg.get('axis_source'):
        raise ValueError('Confirmed XYZ axis_order and documentary axis_source required')
    perm = tuple(order.index(a) for a in 'ZYX')
    with h5py.File(path, 'r') as f:
        image = np.asarray(f[cfg['image_key']], dtype=np.float32)
        label = np.asarray(f[cfg['label_key']])
    if image.shape != label.shape or image.ndim != 3:
        raise ValueError(f'Image/label shape mismatch: {path}')
    if not np.isfinite(image).all() or not np.isin(label, [0, 1]).all():
        raise ValueError(f'Invalid intensity or binary labels: {path}')
    if normalize:
        image = (image - image.mean(dtype=np.float64)) / max(float(image.std(dtype=np.float64)), 1e-6)
    # float64 statistics improve stability; explicitly restore float32 for AMP.
    return np.ascontiguousarray(image.transpose(perm), dtype=np.float32), np.ascontiguousarray(label.transpose(perm), dtype=np.uint8)


def pad_volume(x, y, shape):
    pad = [(0, max(0, n - s)) for n, s in zip(shape, x.shape)]
    return np.pad(x, pad), np.pad(y, pad)


class PatchDataset(Dataset):
    def __init__(self, rows, root, cfg, dimensions, patch, length):
        self.rows, self.root, self.cfg = rows, Path(root), cfg
        self.dimensions, self.patch, self.length = dimensions, tuple(patch), length
        self.cache = {}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Uniform cases first; 50% foreground-centered sampling only on train GT.
        row = self.rows[np.random.randint(len(self.rows))]
        if row['case_id'] not in self.cache:
            self.cache[row['case_id']] = read_volume(self.root / row['path'], self.cfg)
        image, label = self.cache[row['case_id']]
        image, label = pad_volume(image, label, self.patch if self.dimensions == 3 else (1, *self.patch))
        fg = np.random.rand() < 0.5 and label.any()
        center = np.argwhere(label)[np.random.randint(label.sum())] if fg else None
        if self.dimensions == 2:
            z = int(center[0]) if center is not None else np.random.randint(image.shape[0])
            image, label = image[z], label[z]
            center = center[1:] if center is not None else None
        starts = [int(np.clip(c - p // 2, 0, s - p)) if center is not None else np.random.randint(s - p + 1)
                  for c, p, s in zip(center if center is not None else [0] * self.dimensions, self.patch, image.shape)]
        sl = tuple(slice(a, a + n) for a, n in zip(starts, self.patch))
        x, y = image[sl].copy(), label[sl].copy()
        for axis in [-1, -2]:
            if np.random.rand() < 0.5:
                x, y = np.flip(x, axis), np.flip(y, axis)
        k = np.random.randint(4)
        x, y = np.rot90(x, k, (-2, -1)), np.rot90(y, k, (-2, -1))
        x = x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)
        return torch.from_numpy(np.ascontiguousarray(x[None], dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
