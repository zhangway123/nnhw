# -*- coding: utf-8 -*-
"""VQ 码本：EMA 在线更新 + 死码重启，仅作瓶颈正则，不提供类别原型。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes=128, dim=32, beta=0.25, decay=0.99,
                 eps=1e-5, restart_every=100):
        super().__init__()
        self.num_codes = int(num_codes)
        self.dim = int(dim)
        self.beta = float(beta)
        self.decay = float(decay)
        self.eps = float(eps)
        self.restart_every = int(restart_every)

        emb = torch.randn(self.num_codes, self.dim) * 0.1
        self.register_buffer("emb", emb)
        self.register_buffer("cluster_size", torch.zeros(self.num_codes))
        self.register_buffer("ema_sum", emb.clone())
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))
        self.last_used = 0

    def forward(self, z):
        """z: [B, D, H, W] → (z_q, indices [B,H,W], commit_loss)"""
        B, D, H, W = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, D)

        dist = (
            zf.pow(2).sum(1, keepdim=True)
            - 2.0 * zf @ self.emb.t()
            + self.emb.pow(2).sum(1).unsqueeze(0)
        )
        idx = dist.argmin(1)
        zq = self.emb[idx].to(zf.dtype)

        commit = F.mse_loss(zf, zq.detach())
        zq_st = zf + (zq - zf).detach()          # 直通估计
        zq_map = zq_st.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        if self.training:
            self._ema_update(zf.detach(), idx)
            self.last_used = int(torch.unique(idx).numel())

        return zq_map, idx.view(B, H, W), self.beta * commit

    @torch.no_grad()
    def _ema_update(self, zf, idx):
        onehot = F.one_hot(idx, self.num_codes).to(zf.dtype)
        n = onehot.sum(0)

        self.cluster_size.mul_(self.decay).add_(n, alpha=1.0 - self.decay)
        self.ema_sum.mul_(self.decay).add_(onehot.t() @ zf, alpha=1.0 - self.decay)

        # 计数下限取 1：EMA 计数从 0 开始时直接相除会让码向量爆炸（曾导致 vq loss ~1e6）
        denom = self.cluster_size.clamp(min=1.0).unsqueeze(1)
        self.emb.copy_(self.ema_sum / denom)

        self.steps += 1
        if self.restart_every > 0 and int(self.steps.item()) % self.restart_every == 0:
            dead = self.cluster_size < 0.5
            k = int(dead.sum().item())
            if k > 0 and zf.shape[0] > 0:
                pick = torch.randint(0, zf.shape[0], (k,), device=zf.device)
                src = zf[pick].to(self.emb.dtype)     # autocast 下 zf 可能是 fp16
                self.emb[dead] = src
                self.ema_sum[dead] = src
                self.cluster_size[dead] = 1.0
