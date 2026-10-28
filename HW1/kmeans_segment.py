# -*- coding: utf-8 -*-
"""
HW1 无监督流程（nonsupervise.md）—— 第三部分：KMeans 聚类 + 评估
    加载 train_vqvae.py 训练好的 vqvae.pt（编码器与码本冻结）
    1) 图像级聚类：code id 直方图(1024维) → 归一化 → KMeans → 每张图像一个类别
    2) 像素级分割：码本向量 KMeans 聚成 C 簇 → cluster-id 图 → 最近邻上采样 → 512x512 掩码
    3) 评估：重建 MSE/PSNR/SSIM、码本利用率、轮廓/DB 指数、可视化（mIoU 等暂不计算，待有参考后补）

运行：python kmeans_segment.py --ckpt outputs/vqvae.pt --C 5
依赖：scikit-learn、matplotlib
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_preprocess import VQDataset, load_image, build_dataset_json
from train_vqvae import VQVAE


# ============================ 参数 ============================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="train_eval_photos")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--ckpt", type=str, default="outputs/vqvae.pt")
    p.add_argument("--C", type=int, default=5,
                   help="聚类数（0 = 用轮廓系数自动选择，2~10 扫一遍）")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ============================ 重建评估指标 ============================
def _gauss_kernel(size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k = g.outer(g).unsqueeze(0).unsqueeze(0)               # 1,1,size,size
    return k.repeat(channels, 1, 1, 1)


@torch.no_grad()
def ssim(x, y):
    """结构相似度，x/y 均为 [B,3,H,W] 且取值在 [0,1]"""
    k = _gauss_kernel(channels=x.size(1), device=x.device)
    pad = k.shape[-1] // 2
    mu_x = F.conv2d(x, k, padding=pad, groups=x.size(1))
    mu_y = F.conv2d(y, k, padding=pad, groups=y.size(1))
    sx2 = F.conv2d(x * x, k, padding=pad, groups=x.size(1)) - mu_x * mu_x
    sy2 = F.conv2d(y * y, k, padding=pad, groups=y.size(1)) - mu_y * mu_y
    sxy = F.conv2d(x * y, k, padding=pad, groups=x.size(1)) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sx2 + sy2 + c2)
    return (num / den).mean().item()


@torch.no_grad()
def encode_all(model, dl, device):
    """输出每张图像的 code-index 图 [N,32,32] 与重建图像 [N,3,512,512]"""
    model.eval()
    idx_maps, recon = [], []
    for x in dl:
        x = x.to(device)
        x_hat, idx, _, _ = model(x)
        idx_maps.append(idx.cpu())
        recon.append(x_hat.cpu())
    return torch.cat(idx_maps), torch.cat(recon)


# ============================ 主流程 ============================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # 加载数据与划分（与第一部分一致）
    records, _ = build_dataset_json(args.data_dir, args.out_dir)
    with open(os.path.join(args.out_dir, "split.json"), "r", encoding="utf-8") as f:
        split = json.load(f)
    train_idx, eval_idx = split["train"], split["eval"]

    # 加载训练好的 VQ-VAE（冻结）
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = VQVAE(**{k: v for k, v in ckpt["args"].items() if k in ("latent_dim", "num_codes", "beta")})
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"loaded ckpt: {args.ckpt}")

    # 编码：code-index 图 + 重建图像
    eval_dl = DataLoader(VQDataset(records, eval_idx, args.image_size), batch_size=16, shuffle=False)
    idx_all, recon = encode_all(model, eval_dl, device)      # 评估集用于重建指标
    train_dl = DataLoader(VQDataset(records, train_idx, args.image_size), batch_size=16, shuffle=False)
    idx_tr, _ = encode_all(model, train_dl, device)          # 训练集用于聚类

    N = idx_tr.shape[0]

    # ---- 评估 1：重建质量 MSE / PSNR / SSIM ----
    originals = torch.stack([torch.from_numpy(load_image(records[i]["path"], args.image_size)).permute(2, 0, 1)
                             for i in eval_idx])
    mse = ((recon - originals) ** 2).mean().item()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
    s = ssim(recon, originals)
    print(f"[recon] MSE={mse:.5f}  PSNR={psnr:.2f} dB  SSIM={s:.4f}")

    # ---- 评估 2：码本利用率 ----
    used = len(torch.unique(idx_all))
    util = used / model.codebook.num_codes
    print(f"[codebook] utilization={util:.1%} ({used}/{model.codebook.num_codes})")

    # ---- 图像级聚类：code id 直方图 -> KMeans ----
    idx_flat = idx_tr.reshape(N, -1)                          # [N, 1024]
    hist = F.one_hot(idx_flat, model.codebook.num_codes).float().sum(1)   # [N, K]
    hist = hist / hist.sum(1, keepdim=True)
    feat = hist.sqrt()                                        # Hellinger 变换（抑制高频/背景码）
    feat = normalize(feat.numpy())                            # L2 归一化

    if args.C <= 0:                                           # 肘部/轮廓自动选 C
        best_c, best_s = 2, -1
        for c in range(2, 11):
            km = KMeans(n_clusters=c, n_init=10, random_state=args.seed).fit(feat)
            sc = silhouette_score(feat, km.labels_)
            print(f"  C={c} silhouette={sc:.4f}")
            if sc > best_s:
                best_s, best_c = sc, c
        C = best_c
        print(f"auto select C = {C}")
    else:
        C = args.C

    km_img = KMeans(n_clusters=C, n_init=10, random_state=args.seed).fit(feat)
    img_labels = km_img.labels_                                # 每张图像一个类别
    sil = silhouette_score(feat, img_labels)
    db = davies_bouldin_score(feat, img_labels)
    print(f"[image-level] C={C}  silhouette={sil:.4f}  Davies-Bouldin={db:.4f}")
    print("image labels:", img_labels.tolist())

    # ---- 像素级分割：码本向量 KMeans -> cluster-id 图 -> 上采样 ----
    code_vecs = model.codebook.emb.detach().cpu().numpy()      # [K, 64]
    km_pix = KMeans(n_clusters=C, n_init=10, random_state=args.seed).fit(code_vecs)
    cluster_of_code = km_pix.labels_                           # 每个 code id 属于一个簇

    def to_mask(idx_map):                                      # [B,32,32] -> [B,1,512,512]
        m = torch.from_numpy(cluster_of_code[idx_map.numpy()]) # 查表
        return F.interpolate(m.unsqueeze(1).float(), scale_factor=16, mode="nearest")

    masks_all = to_mask(idx_all)                               # 评估集分割掩码
    masks_tr = to_mask(idx_tr)                                 # 训练集分割掩码

    # ---- 可视化：原图 / 重建 / 分割掩码叠加 ----
    os.makedirs(os.path.join(args.out_dir, "vis"), exist_ok=True)
    cmap = plt.get_cmap("tab10")
    for i, gi in enumerate(eval_idx[:8]):
        orig = originals[i].permute(1, 2, 0).numpy()
        rec = recon[i].permute(1, 2, 0).numpy()
        mask = masks_all[i, 0].numpy()
        col = cmap((mask / max(C - 1, 1)).astype(int))[:, :, :3]   # 簇 -> 颜色
        overlay = 0.6 * orig + 0.4 * col
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, im, t in zip(axes, [orig, rec, overlay], ["original", "recon", "mask overlay"]):
            ax.imshow(np.clip(im, 0, 1)); ax.set_title(t); ax.axis("off")
        fig.savefig(os.path.join(args.out_dir, "vis", f"sample_{i}.png"), bbox_inches="tight")
        plt.close(fig)
    print("vis saved to", os.path.join(args.out_dir, "vis"))

    # ---- 保存结果（mIoU 等分割指标暂不计算：无参考标签，待人工标注/伪标签后再补） ----
    results = {
        "C": C, "recon_mse": mse, "recon_psnr": psnr, "recon_ssim": s,
        "codebook_utilization": util,
        "silhouette": float(sil), "davies_bouldin": float(db),
        "img_labels": img_labels.tolist(),
        "note": "mIoU/PA/mPA/Dice/Precision/Recall 等分割指标待参考标签（人工标注或伪标签）后计算",
    }
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("results saved to", os.path.join(args.out_dir, "results.json"))


if __name__ == "__main__":
    main()
