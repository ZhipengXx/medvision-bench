"""Gaussian probability fusion, full coverage, no ground-truth-dependent ROI."""
import itertools
import numpy as np
import torch


def starts(size, patch, overlap=0.5):
    if size <= patch:
        return [0]
    stride = max(1, int(patch * (1 - overlap)))
    return sorted(set(list(range(0, size - patch + 1, stride)) + [size - patch]))


def gaussian(shape):
    w = np.ones(shape, dtype=np.float32)
    for axis, n in enumerate(shape):
        g = np.exp(-0.5 * ((np.arange(n) - (n - 1) / 2) / (n * 0.125)) ** 2)
        view = [1] * len(shape)
        view[axis] = n
        w *= g.reshape(view)
    return np.maximum(w, 1e-5)


@torch.inference_mode()
def infer_array(model, image, patch, device, overlap=0.5, batch_size=2):
    original = image.shape
    image = np.pad(image, [(0, max(0, n - s)) for n, s in zip(patch, original)])
    probs = np.zeros((2, *image.shape), np.float32)
    weight = np.zeros(image.shape, np.float32)
    gauss = gaussian(patch)
    locations = list(itertools.product(*(starts(s, p, overlap) for s, p in zip(image.shape, patch))))
    for off in range(0, len(locations), batch_size):
        locations_batch = locations[off:off + batch_size]
        slices = [tuple(slice(a, a + n) for a, n in zip(loc, patch)) for loc in locations_batch]
        x = torch.from_numpy(np.stack([image[sl] for sl in slices])[:, None]).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            pred = model(x).float().softmax(1).cpu().numpy()
        for sl, prob in zip(slices, pred):
            probs[(slice(None),) + sl] += prob * gauss
            weight[sl] += gauss
    if np.any(weight <= 0):
        raise RuntimeError('Uncovered voxels')
    probs /= weight[None]
    crop = tuple(slice(0, s) for s in original)
    return probs[(slice(None),) + crop].argmax(0).astype(np.uint8)


def predict_volume(model, image, cfg, device):
    model.eval()
    if cfg['dimensions'] == 3:
        return infer_array(model, image, cfg['patch'], device, cfg['overlap'], cfg['inference_batch'])
    # The adapter always puts confirmed Z first; reconstruct exactly in that order.
    return np.stack([infer_array(model, sl, cfg['patch'], device, cfg['overlap'], cfg['inference_batch'])
                     for sl in image])
