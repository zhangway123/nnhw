# -*- coding: utf-8 -*-
"""轻量 U-Net + VQ 瓶颈 + 跳连解码器 + 1×1 分类头（随机初始化）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vq import VectorQuantizerEMA


def _groups(requested, channels):
    g = min(int(requested), int(channels))
    while g > 1 and channels % g != 0:
        g //= 2
    return max(1, g)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, groups=8):
        super().__init__()
        g = _groups(groups, cout)
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, cin, cout, groups=8):
        super().__init__()
        g = _groups(groups, cout)
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, cin, cskip, cout, groups=8):
        super().__init__()
        self.conv = ConvBlock(cin + cskip, cout, groups)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.conv(torch.cat([x, skip], 1))


class VQUNet(nn.Module):
    def __init__(self, num_classes=5, enc_channels=(32, 64, 128, 256),
                 codebook_size=128, code_dim=32, beta=0.25, ema_decay=0.99,
                 downsample=16, groups=8, restart_every=100):
        super().__init__()
        c1, c2, c3, c4 = enc_channels
        self.downsample = int(downsample)

        self.stem = ConvBlock(3, c1, groups)
        self.down1 = Down(c1, c2, groups)
        self.down2 = Down(c2, c3, groups)
        self.down3 = Down(c3, c4, groups)
        self.use_down4 = self.downsample == 16
        self.down4 = Down(c4, c4, groups) if self.use_down4 else None

        self.quant_in = nn.Conv2d(c4, code_dim, 1)
        self.quant = VectorQuantizerEMA(codebook_size, code_dim, beta, ema_decay,
                                        restart_every=restart_every)
        self.quant_out = nn.Conv2d(code_dim, c4, 1)

        if self.use_down4:
            self.ups = nn.ModuleList([
                Up(c4, c4, c4, groups),
                Up(c4, c3, c3, groups),
                Up(c3, c2, c2, groups),
                Up(c2, c1, c1, groups),
            ])
        else:
            self.ups = nn.ModuleList([
                Up(c4, c3, c3, groups),
                Up(c3, c2, c2, groups),
                Up(c2, c1, c1, groups),
            ])

        self.head = nn.Conv2d(c1, num_classes, 1)
        self.recon_head = nn.Conv2d(c1, 3, 1)

    def forward(self, x):
        s1 = self.stem(x)              # H
        s2 = self.down1(s1)            # H/2
        s3 = self.down2(s2)            # H/4
        s4 = self.down3(s3)            # H/8

        if self.use_down4:
            z = self.down4(s4)         # H/16
            skips = [s4, s3, s2, s1]
        else:
            z = s4                     # H/8
            skips = [s3, s2, s1]

        ze = self.quant_in(z)
        zq, indices, vq_loss = self.quant(ze)
        d = self.quant_out(zq)

        for up, skip in zip(self.ups, skips):
            d = up(d, skip)

        logits = self.head(d)
        recon = torch.sigmoid(self.recon_head(d))
        return {
            "logits": logits,
            "recon": recon,
            "vq_loss": vq_loss,
            "indices": indices,
        }
