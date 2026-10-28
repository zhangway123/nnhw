# -*- coding: utf-8 -*-
"""
HW1 无监督流程（nonsupervise.md）—— 第二部分：模型训练（VQ-VAE）
    输入：data_preprocess.py 的产物 dataset.json + split.json（outputs/ 下）
    模型：Encoder(下采样x16) + Codebook(K=1024, EMA 更新) + Decoder(上采样x16)
    损失：重建 MSE + 承诺 beta=0.25；优化器 Adam；AMP(fp16)；early stop；模型保存在本地

环境：Python3 + PyTorch（RTX 5060 Ti Blackwell 需装 cu128 轮子，CUDA 12.8+）
运行：python train_vqvae.py --out_dir outputs
"""
import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_preprocess import VQDataset


# ============================ 超参数（待确认项已在 md 中标注） ============================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="outputs")   # 读取 dataset.json / split.json
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--data_dir", type=str, default="train_eval_photos",
                    help="图像文件夹（数据更多时可改为 train_eval_photos）")
    p.add_argument("--num_codes", type=int, default=1024)      # 码本 K
    p.add_argument("--beta", type=float, default=0.25)         # commitment loss 权重
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)          # 待确认
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ============================ VQ-VAE 模型 ============================
class Encoder(nn.Module):
    """512 -> 32，下采样 x16；4x Conv k3 s2 p1，前三层 BN+ReLU，最后一层仅 Conv+BN（保留负值）"""
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        blocks, ch = [], in_ch
        for i in range(4):
            blocks.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, 3, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True) if i < 3 else nn.Identity(),
            ))
            ch = out_ch
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)                                     # z_e [B,64,32,32]


class Codebook(nn.Module):
    """K=1024 个 64 维码向量；EMA 更新避免码本崩溃"""
    def __init__(self, num_codes=1024, dim=64, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_codes, self.dim = num_codes, dim
        self.decay, self.eps = decay, eps
        self.register_buffer("emb", torch.randn(num_codes, dim))
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("emb_avg", self.emb.clone())

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)          # [B*H*W, dim]
        dist = (flat.pow(2).sum(1, keepdim=True)
                + self.emb.pow(2).sum(1)
                - 2 * flat @ self.emb.t())                     # [N, K]
        idx = dist.argmin(1)                                   # 最近邻 code id

        if self.training:                                      # 仅训练时做 EMA 更新
            one_hot = F.one_hot(idx, self.num_codes).float()   # [N, K]
            embed_sum = flat.t() @ one_hot                     # [dim, K]
            self.cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
            self.emb_avg.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)
            n = self.cluster_size.sum()
            size = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
            self.emb.copy_(self.emb_avg / size.unsqueeze(1))

        z_q = self.emb[idx].reshape(B, H, W, C).permute(0, 3, 1, 2)
        return z_q, idx.reshape(B, H, W)                       # z_q, code-index 图


class Decoder(nn.Module):
    """32 -> 512，上采样 x16；前三层 ConvT k4 s2 p1 + BN + ReLU，最后一层无激活，输出 Sigmoid"""
    def __init__(self, in_ch=64, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),        # 无激活，直接输出
            nn.Sigmoid(),                                      # 归一化到 (0,1)
        )

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):
    """512*512*3 -> 编码器 -> 32*32*64 -> 码本量化 -> 32*32 code-index -> 解码器 -> 512*512*3"""
    def __init__(self, latent_dim=64, num_codes=1024, beta=0.25):
        super().__init__()
        self.encoder = Encoder(3, latent_dim)
        self.codebook = Codebook(num_codes, latent_dim)
        self.decoder = Decoder(latent_dim, 3)
        self.beta = beta

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, idx = self.codebook(z_e)
        z_q_st = z_e + (z_q - z_e).detach()                    # 直通估计器（量化不可导）
        x_hat = self.decoder(z_q_st)
        loss_rec = F.mse_loss(x_hat, x)                        # 重建损失
        loss_commit = self.beta * F.mse_loss(z_e, z_q.detach())  # 承诺损失
        return x_hat, idx, loss_rec, loss_commit


# ============================ 训练循环 ============================
@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    total, n = 0.0, 0
    for x in dl:
        x = x.to(device)
        _, _, l_rec, l_com = model(x)
        total += (l_rec + l_com).item() * len(x)
        n += len(x)
    return total / n


def train(args, train_dl, eval_dl, device):
    model = VQVAE(args.latent_dim, args.num_codes, args.beta).to(device)
    # 码本 emb 由 EMA 更新，不进优化器
    opt = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    best_loss, patience = float("inf"), 0
    ckpt_path = os.path.join(args.out_dir, "vqvae.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x in train_dl:
            x = x.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, _, l_rec, l_com = model(x)
                loss = l_rec + l_com
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
        train_loss = total / len(train_dl.dataset)
        eval_loss = evaluate(model, eval_dl, device)
        print(f"[epoch {epoch:3d}] train_loss={train_loss:.4f}  eval_loss={eval_loss:.4f}")

        if eval_loss < best_loss - 1e-4:
            best_loss, patience = eval_loss, 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)  # 模型保存在本地
            print(f"  -> save best ckpt to {ckpt_path}")
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print("early stop.")
                break


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    torch.manual_seed(args.seed)

    # 读取 data_preprocess.py 的产物
    json_path = os.path.join(args.out_dir, "dataset.json")
    split_path = os.path.join(args.out_dir, "split.json")
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    train_idx, eval_idx = split["train"], split["eval"]
    print(f"训练 {len(train_idx)}，评估 {len(eval_idx)}（来自 {split_path}）")

    train_dl = DataLoader(VQDataset(records, train_idx, args.image_size),
                          batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_dl = DataLoader(VQDataset(records, eval_idx, args.image_size),
                         batch_size=args.batch_size, shuffle=False, num_workers=0)
    train(args, train_dl, eval_dl, device)


if __name__ == "__main__":
    main()
