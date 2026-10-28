# -*- coding: utf-8 -*-
"""SLIC 超像素与超像素级特征（颜色 / LBP 纹理 / 边缘 / 位置）。"""
import numpy as np
from skimage.color import rgb2lab, rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.segmentation import slic


def make_superpixels(rgb, n_segments, compactness=15.0, sigma=1.0):
    """在原图上做 SLIC，返回 (seg, lab)。

    seg: int32 [H, W]，标签从 0 连续编号
    lab: float32 [H, W, 3]
    """
    lab = rgb2lab(rgb).astype(np.float32)
    seg = slic(
        lab,
        n_segments=int(max(2, n_segments)),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
        channel_axis=-1,
    ).astype(np.int32)
    # 连续重编号，保证 0..S-1
    _, inv = np.unique(seg, return_inverse=True)
    return inv.reshape(seg.shape).astype(np.int32), lab


def _bincount(idx, weights, size):
    return np.bincount(idx, weights=weights, minlength=size)[:size]


def _boundary_mask(seg):
    b = np.zeros(seg.shape, dtype=bool)
    b[:, 1:] |= seg[:, 1:] != seg[:, :-1]
    b[:, :-1] |= seg[:, 1:] != seg[:, :-1]
    b[1:, :] |= seg[1:, :] != seg[:-1, :]
    b[:-1, :] |= seg[1:, :] != seg[:-1, :]
    return b


def superpixel_features(rgb, lab, seg):
    """计算原始特征块（未归一化），返回 (blocks, mean_lab)。

    blocks: dict[name] -> float32 [S, d]
        color: Lab 均值(3) + 标准差(3)
        lbp:   uniform LBP 10 bin 直方图
        edge:  超像素内 Sobel 均值 + 边界像素 Sobel 均值
        pos:   归一化质心 (y/H, x/W)
    mean_lab: float32 [S, 3]，用于图割平滑项
    """
    H, W = seg.shape
    S = int(seg.max()) + 1
    idx = seg.reshape(-1)
    cnt = np.bincount(idx, minlength=S).astype(np.float64)
    cnt_safe = np.maximum(cnt, 1.0)

    # ---- 颜色 ----
    labf = lab.reshape(-1, 3).astype(np.float64)
    mean = np.stack([_bincount(idx, labf[:, c], S) for c in range(3)], 1) / cnt_safe[:, None]
    sq = np.stack([_bincount(idx, labf[:, c] ** 2, S) for c in range(3)], 1) / cnt_safe[:, None]
    std = np.sqrt(np.maximum(sq - mean ** 2, 0.0))

    # ---- LBP 纹理（uniform，P=8，R=1，10 bin）----
    gray = rgb2gray(rgb)
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_u8, P=8, R=1, method="uniform").astype(np.uint8)
    lbp_flat = lbp.reshape(-1)
    lbp_hist = np.stack(
        [_bincount(idx, (lbp_flat == b).astype(np.float64), S) for b in range(10)], 1
    ) / cnt_safe[:, None]

    # ---- 边缘 ----
    sob = sobel(gray).astype(np.float64)
    sob_flat = sob.reshape(-1)
    edge_mean = _bincount(idx, sob_flat, S) / cnt_safe

    bmask = _boundary_mask(seg)
    b_idx = seg[bmask]
    b_val = sob[bmask]
    b_sum = np.bincount(b_idx, weights=b_val, minlength=S)[:S]
    b_cnt = np.bincount(b_idx, minlength=S)[:S]
    edge_boundary = np.where(b_cnt > 0, b_sum / np.maximum(b_cnt, 1), edge_mean)

    # ---- 位置（归一化质心）----
    ys = np.repeat(np.arange(H, dtype=np.float64) / max(H - 1, 1), W)
    xs = np.tile(np.arange(W, dtype=np.float64) / max(W - 1, 1), H)
    cy = _bincount(idx, ys, S) / cnt_safe
    cx = _bincount(idx, xs, S) / cnt_safe

    blocks = {
        "color": np.concatenate([mean, std], 1).astype(np.float32),
        "lbp": lbp_hist.astype(np.float32),
        "edge": np.stack([edge_mean, edge_boundary], 1).astype(np.float32),
        "pos": np.stack([cy, cx], 1).astype(np.float32),
    }
    return blocks, mean.astype(np.float32), edge_boundary.astype(np.float32)


def superpixel_triangle_overlap(seg, tri_mask):
    """返回每个超像素被三角形覆盖的比例 rho [S]。"""
    S = int(seg.max()) + 1
    idx = seg.reshape(-1)
    hit = np.bincount(idx, weights=tri_mask.reshape(-1).astype(np.float64), minlength=S)[:S]
    tot = np.bincount(idx, minlength=S)[:S]
    return (hit / np.maximum(tot, 1)).astype(np.float32)


def fit_feature_stats(blocks_list):
    """用所有图像的原始特征块统计全局均值/标准差（跨图统一特征空间）。"""
    stats = {}
    for name in blocks_list[0].keys():
        allv = np.concatenate([b[name] for b in blocks_list], 0).astype(np.float64)
        stats[name] = (allv.mean(0), allv.std(0) + 1e-6)
    return stats


def apply_feature_stats(blocks, stats, weights):
    """按全局统计量做 z-score，再按特征块权重拼接。"""
    parts = []
    for name, arr in blocks.items():
        mu, sd = stats[name]
        w = float(weights.get(name, 1.0))
        parts.append(((arr - mu) / sd * w).astype(np.float32))
    return np.concatenate(parts, 1)


def build_graph(seg, mean_lab, edge_boundary):
    """构建超像素邻接图，返回 edges [E,3] = (i, j, w)。

    w = exp(-||Lab_i-Lab_j||^2 / 2σ_c^2) * (1 + 边界梯度)，归一化到 [0,1]。
    共享边界梯度取两侧超像素边界梯度均值（避免引入像素级大索引数组）。
    """
    S = int(seg.max()) + 1

    a = np.concatenate([seg[:, :-1].reshape(-1), seg[:-1, :].reshape(-1)])
    b = np.concatenate([seg[:, 1:].reshape(-1), seg[1:, :].reshape(-1)])
    m = a != b
    a, b = a[m], b[m]
    if a.size == 0:
        return np.zeros((0, 3), np.float32)

    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    uniq, _, cnts = np.unique(lo * S + hi, return_inverse=True, return_counts=True)

    i_idx = (uniq // S).astype(np.int32)
    j_idx = (uniq % S).astype(np.int32)

    dcol = np.linalg.norm(
        mean_lab[i_idx].astype(np.float64) - mean_lab[j_idx].astype(np.float64), axis=1
    )
    pos = dcol[dcol > 1e-6]
    sigma_c = float(np.median(pos)) if pos.size else 1.0

    e_ij = 0.5 * (edge_boundary[i_idx] + edge_boundary[j_idx])
    e_norm = e_ij / max(float(e_ij.max()), 1e-6)

    w = np.exp(-(dcol ** 2) / (2 * sigma_c ** 2)) * (1.0 + e_norm)
    w = (w / max(float(w.max()), 1e-6)).astype(np.float32)
    return np.stack([i_idx, j_idx, w], 1)
