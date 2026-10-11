from torch.nn import functional as F


def ce_dice(logits, target, eps=1e-6):
    ce = F.cross_entropy(logits, target)
    p, g = logits.softmax(1)[:, 1].float(), target.float()
    axes = tuple(range(1, p.ndim))
    # Foreground soft Dice per sample, then batch mean; CE and Dice weight = 1.
    dice = 1 - ((2 * (p * g).sum(axes) + eps) / (p.sum(axes) + g.sum(axes) + eps)).mean()
    return ce + dice, ce, dice
