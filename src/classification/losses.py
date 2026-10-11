"""Three controlled objectives, all consuming unnormalized logits."""
import torch
from torch import nn
from torch.nn import functional as F


def inverse_frequency(counts):
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if torch.any(counts <= 0):
        raise ValueError('Every class needs training samples')
    # w_c = N/(C*n_c). Weighted mean CE divides by sum of target weights.
    return counts.sum() / (len(counts) * counts)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        # FL = -(1-p_y)^gamma log(p_y), alpha=1, arithmetic batch mean.
        return (-((1 - logp.exp()) ** self.gamma) * logp).mean()


def make_loss(name, counts, device):
    if name == 'ce':
        return nn.CrossEntropyLoss()
    if name == 'wce':
        return nn.CrossEntropyLoss(weight=inverse_frequency(counts).to(device))
    if name == 'focal':
        return FocalLoss()
    raise ValueError(name)
