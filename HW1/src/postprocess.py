# -*- coding: utf-8 -*-
"""后处理与输出：调色板、连通域清理、SLIC 投票、letterbox、结果保存。"""
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# 0 背景 + 1~4 前景，固定配色（附录图保持一致）
PALETTE = np.array([
    [0, 0, 0],
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [67, 99, 216],
], dtype=np.uint8)


def remove_small_components(mask, num_classes, min_ratio=0.005):
    """删除面积小于图像 min_ratio 的孤立连通域（回填为背景）。"""
    out = mask.copy()
    total = float(mask.size)
    for c in range(1, num_classes):
        m = mask == c
        if not m.any():
            continue
        lab, n = ndi.label(m)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        for k in range(1, n + 1):
            if sizes[k] < min_ratio * total:
                out[lab == k] = 0
    return out


def keep_point_components(mask, points, num_classes, min_ratio=0.005):
    """保留点所在连通域；只删除无点覆盖且面积小的同类孤立块。"""
    out = mask.copy()
    total = float(mask.size)
    for x, y, cls in points:
        m = out == int(cls)
        if not m.any():
            continue
        lab, n = ndi.label(m)
        yi, xi = int(round(y)), int(round(x))
        if not (0 <= yi < lab.shape[0] and 0 <= xi < lab.shape[1]):
            continue
        keep = lab[yi, xi]
        if keep == 0:
            continue
        sizes = np.bincount(lab.ravel())
        for k in range(1, n + 1):
            if k != keep and sizes[k] < min_ratio * total:
                out[lab == k] = 0
    return out


def point_consistency(mask, points):
    """最终掩码在点处的类别是否与标注一致。"""
    ok = True
    for x, y, cls in points:
        yi, xi = int(round(y)), int(round(x))
        yi = int(np.clip(yi, 0, mask.shape[0] - 1))
        xi = int(np.clip(xi, 0, mask.shape[1] - 1))
        if int(mask[yi, xi]) != int(cls):
            ok = False
    return ok


def force_points(mask, points):
    """点处类别不符时强制回填（5×5 窗口），保证监督锚点不被抹掉。"""
    out = mask.copy()
    H, W = out.shape
    for x, y, cls in points:
        yi, xi = int(round(y)), int(round(x))
        y0, y1 = max(0, yi - 2), min(H, yi + 3)
        x0, x1 = max(0, xi - 2), min(W, xi + 3)
        out[y0:y1, x0:x1] = int(cls)
    return out


def slic_vote(prob, seg):
    """超像素级概率投票：prob [C,H,W] → labels [H,W]。"""
    C = prob.shape[0]
    S = int(seg.max()) + 1
    idx = seg.reshape(-1)
    cnt = np.bincount(idx, minlength=S).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)
    sums = np.stack(
        [np.bincount(idx, weights=prob[c].reshape(-1).astype(np.float64), minlength=S)[:S]
         for c in range(C)], 1)
    sup = (sums / cnt[:, None]).argmax(1)
    return sup[seg].astype(np.uint8)


def letterbox_image(img, size=1024, fill=0):
    """长边缩放到 size，短边居中 padding 成方形。"""
    H, W = img.shape[:2]
    s = size / float(max(H, W))
    nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    if img.ndim == 2:
        canvas = np.full((size, size), fill, dtype=img.dtype)
    else:
        canvas = np.full((size, size, img.shape[2]), fill, dtype=img.dtype)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    info = {"scale": s, "pad": (left, top), "new_size": (nw, nh), "orig_size": (W, H)}
    return canvas, info


def unletterbox_mask(mask, info):
    """裁掉 padding 并还原到原图尺寸。"""
    left, top = info["pad"]
    nw, nh = info["new_size"]
    W, H = info["orig_size"]
    crop = mask[top:top + nh, left:left + nw]
    return np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((W, H), Image.NEAREST))


def save_outputs(img_rgb, mask, out_dir, stem, meta=None):
    """保存 mask / color / overlay / json，返回文件路径。"""
    os.makedirs(out_dir, exist_ok=True)
    color = PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]
    overlay = (0.55 * img_rgb.astype(np.float32) + 0.45 * color.astype(np.float32))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    paths = {
        "mask": os.path.join(out_dir, f"{stem}_mask.png"),
        "color": os.path.join(out_dir, f"{stem}_color.png"),
        "overlay": os.path.join(out_dir, f"{stem}_overlay.png"),
        "json": os.path.join(out_dir, f"{stem}.json"),
    }
    Image.fromarray(mask.astype(np.uint8), mode="L").save(paths["mask"])
    Image.fromarray(color).save(paths["color"])
    Image.fromarray(overlay).save(paths["overlay"])

    payload = dict(meta or {})
    payload.update({
        "image": stem,
        "shape": list(mask.shape),
        "class_ratio": {str(int(k)): round(float(v) / mask.size, 4)
                        for k, v in zip(*np.unique(mask, return_counts=True))},
    })
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return paths
