"""Standard padded U-Net: four pools, double convs, transposed convs and skips."""
import torch
from torch import nn
from torch.nn import functional as F


class UNet(nn.Module):
    def __init__(self, dimensions=3, base_channels=16):
        super().__init__()
        self.dimensions = dimensions
        Conv = nn.Conv3d if dimensions == 3 else nn.Conv2d
        Norm = nn.InstanceNorm3d if dimensions == 3 else nn.InstanceNorm2d
        Pool = nn.MaxPool3d if dimensions == 3 else nn.MaxPool2d
        Up = nn.ConvTranspose3d if dimensions == 3 else nn.ConvTranspose2d
        def block(a, b):
            return nn.Sequential(Conv(a, b, 3, padding=1, bias=False), Norm(b, affine=True), nn.ReLU(),
                                 Conv(b, b, 3, padding=1, bias=False), Norm(b, affine=True), nn.ReLU())
        channels = [base_channels * 2 ** i for i in range(5)]
        self.encoders = nn.ModuleList([block(a, b) for a, b in zip([1] + channels[:-1], channels)])
        self.pool = Pool(2)
        self.ups = nn.ModuleList([Up(channels[i], channels[i - 1], 2, stride=2) for i in range(4, 0, -1)])
        self.decoders = nn.ModuleList([block(2 * channels[i - 1], channels[i - 1]) for i in range(4, 0, -1)])
        self.out = Conv(channels[0], 2, 1)

    def forward(self, x):
        original = x.shape[2:]
        pad = []
        for n in reversed(original):
            pad += [0, (-n) % 16]
        x = F.pad(x, pad)
        skips = []
        for i, enc in enumerate(self.encoders):
            if i:
                x = self.pool(x)
            x = enc(x)
            skips.append(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.out(x)[(slice(None), slice(None)) + tuple(slice(0, n) for n in original)]
