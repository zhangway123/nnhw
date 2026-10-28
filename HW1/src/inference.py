# -*- coding: utf-8 -*-
"""整图推理（letterbox）：训练验证、自训练、测试共用。"""
import numpy as np
import torch

from src.postprocess import letterbox_image, slic_vote, unletterbox_mask
from src.weak_labels.slic_features import make_superpixels


@torch.no_grad()
def predict_tiled(model, img_rgb, device, tile=512, stride=384, batch=4,
                  use_amp=False, return_conf=False):
    """滑窗推理（原图分辨率），与训练时的裁剪尺度保持一致。

    大图直接 letterbox 到 1024 会把目标缩小 4~6 倍，和训练分布不符；
    这里按训练尺度逐块推理再拼接，重叠区取概率更大的结果。
    """
    H, W = img_rgb.shape[:2]
    tile = int(min(tile, max(H, W)))
    stride = int(max(1, min(stride, tile)))

    ys = list(range(0, max(H - tile, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, stride)) or [0]
    if ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    if xs[-1] + tile < W:
        xs.append(max(0, W - tile))

    label = np.zeros((H, W), dtype=np.uint8)
    best = np.zeros((H, W), dtype=np.float32)

    coords = [(y, x) for y in ys for x in xs]
    device_type = "cuda" if device.type == "cuda" else "cpu"

    for start in range(0, len(coords), batch):
        chunk = coords[start:start + batch]
        patches = []
        for (y0, x0) in chunk:
            patch = img_rgb[y0:y0 + tile, x0:x0 + tile]
            ph, pw = patch.shape[:2]
            if ph < tile or pw < tile:
                pad = np.zeros((tile, tile, 3), dtype=patch.dtype)
                pad[:ph, :pw] = patch
                patch = pad
            patches.append(patch)

        x = torch.from_numpy(np.stack(patches).astype(np.float32) / 255.0)
        x = x.permute(0, 3, 1, 2)
        x = ((x - 0.5) / 0.5).to(device)

        with torch.amp.autocast(device_type, enabled=use_amp and device.type == "cuda"):
            out = model(x)
        prob = out["logits"].softmax(1).float().cpu().numpy()

        for i, (y0, x0) in enumerate(chunk):
            ph = min(tile, H - y0)
            pw = min(tile, W - x0)
            p = prob[i, :, :ph, :pw]
            lab = p.argmax(0).astype(np.uint8)
            conf = p.max(0)
            win_lab = label[y0:y0 + ph, x0:x0 + pw]
            win_best = best[y0:y0 + ph, x0:x0 + pw]
            upd = conf > win_best
            win_lab[upd] = lab[upd]
            win_best[upd] = conf[upd]

    return (label, best) if return_conf else label


@torch.no_grad()
def predict_probs(model, img_rgb, size, device, use_amp=False):
    """返回 (prob [C,Size,Size], letterbox 信息)。"""
    lb, info = letterbox_image(img_rgb, size)
    x = torch.from_numpy(lb.astype(np.float32) / 255.0).permute(2, 0, 1)
    x = ((x - 0.5) / 0.5).unsqueeze(0).to(device)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(device_type, enabled=use_amp and device.type == "cuda"):
        out = model(x)
    prob = out["logits"].softmax(1)[0].float().cpu().numpy()
    return prob, info, out


def predict_mask(model, img_rgb, size, device, use_amp=False, use_slic=True,
                 n_segments=4000, compactness=15.0):
    """返回原图尺寸的类别掩码。"""
    prob, info, _ = predict_probs(model, img_rgb, size, device, use_amp)
    if use_slic:
        lb, _ = letterbox_image(img_rgb, size)
        seg, _ = make_superpixels(lb, n_segments, compactness)
        mask = slic_vote(prob, seg)
    else:
        mask = prob.argmax(0).astype(np.uint8)
    return unletterbox_mask(mask, info), prob, info
