# -*- coding: utf-8 -*-
"""端到端冒烟测试（不依赖真实数据集）：伪标签 → 模型 → 损失 → 数据加载 → 输出。"""
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.weak_labels.slic_features import (
    apply_feature_stats, build_graph, fit_feature_stats, make_superpixels,
    superpixel_features,
)
from src.weak_labels.prototypes import (
    border_superpixels, class_prototypes, hard_constraints, normalize_unary,
    seed_superpixels, unary_from_prototypes,
)
from src.weak_labels.graphcut import solve
from src.models.vq_unet import VQUNet
from src.losses import (consistency_loss, dice_loss, masked_ce, point_ce,
                        reconstruction_loss, tri_ce)
from src.data.dataset import SegDataset
from src.data.instances import Instance
from src.postprocess import (
    letterbox_image, remove_small_components, save_outputs, unletterbox_mask,
)


def make_synthetic(H=256, W=384):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[..., 0] = np.clip(180 + 40 * yy / H, 0, 255)
    img[..., 1] = np.clip(190 + 30 * xx / W, 0, 255)
    img[..., 2] = 210

    blobs = [((80, 70), (200, 30, 30)), ((140, 200), (30, 150, 40)), ((200, 310), (40, 60, 200))]
    pts = []
    for (cy, cx), color in blobs:
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= 28 ** 2
        img[m] = color
        pts.append((float(cx), float(cy), 1))
    return img, pts


def main():
    img, pts = make_synthetic()
    H, W = img.shape[:2]

    seg, lab = make_superpixels(img, 600, 15.0, 1.0)
    blocks, mean_lab, edge_bd = superpixel_features(img, lab, seg)
    stats = fit_feature_stats([blocks])
    feats = apply_feature_stats(blocks, stats, {"color": 1.0, "lbp": 0.5, "edge": 0.5, "pos": 0.1})
    print("superpixels:", int(seg.max()) + 1, "feat dim:", feats.shape[1])

    seeds = seed_superpixels(seg, pts)
    bg = border_superpixels(seg, 0.03, [s for s, _ in seeds])
    protos, valid = class_prototypes([feats], [seeds], 2)   # 0 背景 + 1 类
    D = unary_from_prototypes(feats, protos, valid, feats[bg].mean(0))
    D, sigma = normalize_unary(D)
    hard = hard_constraints(len(feats), seeds, bg, 2)
    edges = build_graph(seg, mean_lab, edge_bd)
    sup, energy, method = solve(D, edges, hard, 1.0, 2)
    mask = sup[seg]
    print(f"graphcut: {method}  edges={len(edges)}  energy={energy:.1f}  sigma={sigma:.3f}")
    ratio = float((mask == 1).mean())
    ok_points = all(mask[int(y), int(x)] == 1 for x, y, _ in pts)
    print(f"前景占比={ratio:.3f}  点一致={ok_points}")
    assert ok_points, "点硬约束失败"
    assert 0.0 < ratio < 0.6, "前景占比异常"

    # 模型前向/反向
    model = VQUNet(num_classes=2, enc_channels=(16, 32, 64, 128), codebook_size=32,
                   code_dim=16, downsample=16, groups=8)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out["logits"].shape == (2, 2, 64, 64), out["logits"].shape
    assert out["recon"].shape == (2, 3, 64, 64)
    loss = out["logits"].mean() + out["vq_loss"]
    loss.backward()
    print(f"model ok: logits {tuple(out['logits'].shape)}  used_codes={model.quant.last_used}")

    # 损失
    tgt = torch.zeros(2, 64, 64, dtype=torch.long)
    wgt = torch.zeros(2, 64, 64)
    wgt[:, 30:35, 30:35] = 1.0
    tgt[:, 30:35, 30:35] = 1
    prob = out["logits"].softmax(1)
    lp = point_ce(out["logits"], tgt, wgt)
    lt = tri_ce(out["logits"], tgt, wgt)
    lc = masked_ce(out["logits"], tgt, wgt)
    ld = dice_loss(prob, tgt, wgt, 2)
    lcons = consistency_loss(prob, prob * 0.9)
    lrec = reconstruction_loss(out["recon"], x)
    print(f"losses: point={lp.item():.3f} tri={lt.item():.3f} ce={lc.item():.3f} "
          f"dice={ld.item():.3f} consist={lcons.item():.4f} recon={lrec.item():.4f}")

    # 数据加载（伪标签 + 点）
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "fake.jpg")
        Image.fromarray(img).save(img_path)
        Image.fromarray(mask.astype(np.uint8), mode="L").save(os.path.join(tmp, "fake_label.png"))
        Image.fromarray(np.full((H, W), 230, dtype=np.uint8), mode="L").save(
            os.path.join(tmp, "fake_conf.png"))

        insts = [Instance(1, "point", [(x, y)]) for x, y, _ in pts]
        insts.append(Instance(1, "triangle", [(140.0, 80.0), (200.0, 80.0), (170.0, 140.0)],
                              area=1800.0))
        ds = SegDataset([img_path], {"fake": insts}, pseudo_dir=tmp, image_size=128,
                        num_classes=2, point_window=5, train=False)
        item = ds[0]
        print("dataset ok:", {k: (tuple(v.shape) if torch.is_tensor(v) else v)
                              for k, v in item.items() if k != "name"})
        assert item["weak"].shape == (3, 128, 128)
        assert item["pseudo"].shape == (128, 128)
        assert int(item["point_weight"].sum()) > 0
        assert int(item["tri_w"].sum()) > 0 and float(item["n_tri"]) >= 1.0

        # letterbox / 输出
        lb, info = letterbox_image(img, 256)
        back = unletterbox_mask(np.zeros((256, 256), dtype=np.uint8), info)
        assert back.shape == (H, W), back.shape
        clean = remove_small_components(mask.astype(np.uint8), 2, 0.005)
        out_dir = os.path.join(tmp, "pred")
        paths = save_outputs(img, clean, out_dir, "fake", meta={"note": "smoke"})
        for p in paths.values():
            assert os.path.isfile(p), p
        print("postprocess ok:", list(paths.keys()))

    print("ALL OK")


if __name__ == "__main__":
    main()
