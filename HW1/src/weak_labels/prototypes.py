# -*- coding: utf-8 -*-
"""类别原型（单点/三角形硬约束）与背景原型（边界超像素）。"""
import numpy as np

from src.data.instances import triangle_mask
from src.weak_labels.slic_features import superpixel_triangle_overlap


def seed_superpixels_from_instances(seg, instances, rho_hard=0.9, rho_soft=0.5,
                                    tri_weight_cap=50.0):
    """实例 → 种子超像素。

    返回 (seeds, hard)：
        seeds: list[(sup_id, cls, weight)]，rho >= rho_soft 的超像素（含全部单点超像素）；
        hard:  dict[sup_id, cls]，单点超像素 + rho >= rho_hard 的超像素。
    三角形按面积加权（weight = min(area/1e4, cap)，在该三角形的种子超像素间均分）。
    """
    H, W = seg.shape
    seeds, hard = [], {}

    for inst in instances:
        if inst.kind == "point":
            x, y = inst.points[0]
            xi = int(np.clip(round(x), 0, W - 1))
            yi = int(np.clip(round(y), 0, H - 1))
            sup = int(seg[yi, xi])
            seeds.append((sup, int(inst.cls), 1.0))
            hard[sup] = int(inst.cls)
            continue

        tm = triangle_mask(inst.points, (H, W))
        if not tm.any():
            continue
        rho = superpixel_triangle_overlap(seg, tm)
        sups = np.nonzero(rho >= rho_soft)[0]
        if sups.size == 0:
            # 原图分辨率下超像素可能比三角形还大（rho 全部 < rho_soft）：
            # 退化为「三角形顶点 + 质心」所在超像素做硬约束，否则三角形对图割毫无贡献
            cx, cy = inst.center()
            picked = set()
            for x, y in list(inst.points) + [(cx, cy)]:
                xi = int(np.clip(round(x), 0, W - 1))
                yi = int(np.clip(round(y), 0, H - 1))
                picked.add(int(seg[yi, xi]))
            sups = np.array(sorted(picked), dtype=np.int64)
            w = float(min(inst.area / 1e4, tri_weight_cap)) / float(sups.size)
            for s in sups:
                sup = int(s)
                seeds.append((sup, int(inst.cls), w))
                hard[sup] = int(inst.cls)
            continue

        w = float(min(inst.area / 1e4, tri_weight_cap)) / float(sups.size)
        for s in sups:
            sup = int(s)
            seeds.append((sup, int(inst.cls), w))
            if rho[sup] >= rho_hard:
                hard[sup] = int(inst.cls)

    return seeds, hard


def seed_superpixels(seg, points):
    """点 → 所在超像素。

    返回 seeds: list[(sup_id, cls)]，points 为 [(x, y, cls), ...]（原图坐标）。
    """
    H, W = seg.shape
    seeds = []
    for x, y, cls in points:
        xi = int(np.clip(round(x), 0, W - 1))
        yi = int(np.clip(round(y), 0, H - 1))
        seeds.append((int(seg[yi, xi]), int(cls)))
    return seeds


def border_superpixels(seg, ratio, exclude_sups=()):
    """图像四周边界带内的超像素（排除含点的超像素）。"""
    H, W = seg.shape
    by = max(1, int(round(H * ratio)))
    bx = max(1, int(round(W * ratio)))
    band = np.zeros((H, W), dtype=bool)
    band[:by, :] = True
    band[-by:, :] = True
    band[:, :bx] = True
    band[:, -bx:] = True
    sups = np.unique(seg[band])
    excl = set(int(s) for s in exclude_sups)
    return np.array([s for s in sups if int(s) not in excl], dtype=np.int64)


def class_prototypes(feats_list, seeds_list, num_classes):
    """跨图合并同类种子超像素的特征，得到类别原型（加权均值）。

    feats_list: list of [S_i, D]（已归一化）
    seeds_list: list of [(sup_id, cls), ...] 或 [(sup_id, cls, weight), ...]
    返回 (protos [num_classes, D], valid [num_classes] bool)；背景类 0 不使用原型。
    """
    D = feats_list[0].shape[1]
    protos = np.zeros((num_classes, D), dtype=np.float32)
    valid = np.zeros(num_classes, dtype=bool)

    buckets = {c: [] for c in range(1, num_classes)}
    for feats, seeds in zip(feats_list, seeds_list):
        for item in seeds:
            sup, cls = int(item[0]), int(item[1])
            w = float(item[2]) if len(item) > 2 else 1.0
            if 1 <= cls < num_classes:
                buckets[cls].append((feats[sup].astype(np.float64), w))

    for c, items in buckets.items():
        if len(items) > 0:
            vecs = np.stack([v for v, _ in items], 0)
            ws = np.array([w for _, w in items], dtype=np.float64)
            ws = ws / max(ws.sum(), 1e-6)
            protos[c] = (vecs * ws[:, None]).sum(0).astype(np.float32)
            valid[c] = True
    return protos, valid


def unary_from_prototypes(feats, protos, valid, bg_proto, big=1e6):
    """原型距离 → 数据项 D [S, C]（未归一化，C = num_classes）。"""
    S = feats.shape[0]
    C = protos.shape[0]
    D = np.full((S, C), big, dtype=np.float32)

    # 背景（类 0）
    if bg_proto is not None:
        D[:, 0] = np.sum((feats - bg_proto[None, :]) ** 2, axis=1)

    for c in range(1, C):
        if valid[c]:
            D[:, c] = np.sum((feats - protos[c][None, :]) ** 2, axis=1)
    return D


def normalize_unary(D, sigma=None):
    """用中位距离缩放数据项，输出 (D_norm, sigma)。"""
    vals = D[D < 1e5]
    if sigma is None:
        sigma = float(np.median(vals)) if vals.size else 1.0
    sigma = max(sigma, 1e-6)
    return D / (2.0 * sigma ** 2), sigma


def hard_constraints(num_sups, seeds, bg_sups, num_classes):
    """构造硬约束数组：-1 表示无约束，否则为该类 ID。

    点所在超像素 = 点类别（最高优先级）；不含点的边界带超像素 = 背景。
    """
    hard = np.full(num_sups, -1, dtype=np.int32)
    for sup, cls in seeds:
        if 0 <= cls < num_classes:
            hard[sup] = cls
    for sup in bg_sups:
        if hard[sup] == -1:
            hard[sup] = 0
    return hard
