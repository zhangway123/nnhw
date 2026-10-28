# -*- coding: utf-8 -*-
"""阶段 1：SLIC 超像素原型 + 多类图割 → 初始伪标签与置信度。

label.json 的 point 兼容单点（dict）与三角形（3 点 list）：
三角形内部是区域级正样本，既参与原型/硬约束，也在像素级强制写入。

用法：
    python src/weak_labels/make_pseudo.py                       # 全量 9 张标注图
    python src/weak_labels/make_pseudo.py --limit 2             # 只跑前 2 张（调试）
    python src/weak_labels/make_pseudo.py --max-side 1024       # 降分辨率快速验证
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.instances import (                                  # noqa: E402
    Instance,
    cap_triangle_area,
    load_instances,
    triangle_mask,
)
from src.utils import load_config, resolve, ensure_dir  # noqa: E402
from src.weak_labels.slic_features import (  # noqa: E402
    apply_feature_stats,
    build_graph,
    fit_feature_stats,
    make_superpixels,
    superpixel_features,
    superpixel_triangle_overlap,
)
from src.weak_labels.prototypes import (  # noqa: E402
    border_superpixels,
    class_prototypes,
    normalize_unary,
    seed_superpixels_from_instances,
    unary_from_prototypes,
)
from src.weak_labels.graphcut import solve  # noqa: E402
from src.postprocess import PALETTE  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--labels", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0 = 全部）")
    p.add_argument("--max-side", type=int, default=None, help="调试用：先把长边缩到该值")
    return p.parse_args()


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def resize_for_debug(rgb, instances, max_side):
    """调试用降分辨率；单点与三角形顶点用完全相同的缩放。"""
    if not max_side:
        return rgb, instances, 1.0
    H, W = rgb.shape[:2]
    if max(H, W) <= max_side:
        return rgb, instances, 1.0
    s = max_side / float(max(H, W))
    new = (max(1, int(round(W * s))), max(1, int(round(H * s))))
    img = Image.fromarray(rgb).resize(new, Image.BILINEAR)
    scaled = [Instance(i.cls, i.kind, [(x * s, y * s) for x, y in i.points], i.area * s * s)
              for i in instances]
    return np.asarray(img), scaled, s


def draw_instances(overlay, instances, scale=1.0):
    """在叠加图上画单点与三角形轮廓（颜色与类别一致）。"""
    im = Image.fromarray(overlay)
    dr = ImageDraw.Draw(im)
    for inst in instances:
        color = tuple(int(v) for v in PALETTE[int(inst.cls)])
        if inst.kind == "point":
            x, y = inst.points[0]
            x, y = x * scale, y * scale
            r = 6
            dr.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255), width=2)
            dr.ellipse([x - r + 2, y - r + 2, x + r - 2, y + r - 2], fill=color)
        else:
            pts = [(x * scale, y * scale) for x, y in inst.points]
            dr.polygon(pts, outline=(255, 255, 255), width=3)
            dr.line(pts + [pts[0]], fill=color, width=2)
    return np.asarray(im)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    pc = cfg["pseudo"]
    num_classes = int(cfg["data"]["num_classes"])
    rho_hard = float(pc.get("rho_hard", 0.9))
    rho_soft = float(pc.get("rho_soft", 0.5))
    tri_cap = float(pc.get("tri_area_ratio_cap", 0) or 0)
    labels_path = resolve(args.labels or cfg["data"]["labels"])
    train_dir = resolve(cfg["data"]["train_dir"])
    out_dir = ensure_dir(args.out or cfg["data"]["pseudo_dir"])
    vis_dir = ensure_dir(os.path.join(out_dir, "vis"))

    _, inst_map = load_instances(labels_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if args.limit:
        entries = entries[: args.limit]

    # ---------- 第一遍：SLIC + 原始特征 ----------
    cache = []
    for e in entries:
        t0 = time.time()
        name = e["image name"]
        stem = os.path.splitext(name)[0]
        rgb = load_rgb(os.path.join(train_dir, name))
        instances = inst_map[stem]
        if not instances:
            print(f"[warn] {name}: 没有有效实例，跳过")
            continue
        rgb, instances, scale = resize_for_debug(rgb, instances, args.max_side)

        H, W = rgb.shape[:2]
        instances, capped = cap_triangle_area(instances, (H, W), tri_cap)
        for cls, ratio in capped:
            print(f"[warn] {name}: 三角形面积占图 {ratio:.1%} 超过上限 {tri_cap:.0%}，"
                  f"退化为类 {cls} 的质心单点")
        n_seg = int(np.clip(round(H * W / pc["n_segments_div"]),
                            pc["n_segments_min"], pc["n_segments_max"]))
        seg, lab = make_superpixels(rgb, n_seg, pc["compactness"], pc["sigma"])
        blocks, mean_lab, edge_bd = superpixel_features(rgb, lab, seg)

        seeds, hard_seeds = seed_superpixels_from_instances(
            seg, instances, rho_hard=rho_hard, rho_soft=rho_soft)
        if len(seeds) == 0:
            print(f"[warn] {name}: 没有有效种子超像素，跳过")
            continue

        tri_masks = [triangle_mask(i.points, (H, W)) for i in instances if i.kind == "triangle"]

        # 背景不能从三角形里取（哪怕只覆盖 10%）
        excl = set(int(s) for s, _, _ in seeds)
        for tm in tri_masks:
            rho = superpixel_triangle_overlap(seg, tm)
            excl.update(int(s) for s in np.nonzero(rho >= 0.1)[0])
        bg_sups = border_superpixels(seg, pc["border_ratio"], sorted(excl))

        n_pt = sum(1 for i in instances if i.kind == "point")
        cache.append({
            "name": name, "stem": stem, "cls": int(e["label"]),
            "instances": instances, "scale": scale, "rgb": rgb, "shape": (H, W),
            "seg": seg.astype(np.int16), "mean_lab": mean_lab, "edge_bd": edge_bd,
            "blocks": blocks, "seeds": seeds, "hard_seeds": hard_seeds,
            "tri_masks": tri_masks, "bg_sups": bg_sups,
        })
        print(f"[1/2] {name}  {W}x{H}  S={int(seg.max()) + 1}  "
              f"实例={len(instances)}（单点 {n_pt} + 三角形 {len(tri_masks)}）  "
              f"用时 {time.time() - t0:.1f}s")
        del lab

    if not cache:
        print("没有可处理的图像")
        return

    # ---------- 全局标准化 + 类别原型 ----------
    stats = fit_feature_stats([c["blocks"] for c in cache])
    for c in cache:
        c["feats"] = apply_feature_stats(c["blocks"], stats, pc["feature_weights"])
    protos, valid = class_prototypes([c["feats"] for c in cache],
                                     [c["seeds"] for c in cache], num_classes)
    missing = [c for c in range(1, num_classes) if not valid[c]]
    if missing:
        print(f"[warn] 以下类别没有任何种子，无法建原型: {missing}")

    # ---------- 第二遍：图割 ----------
    summary = []
    for c in cache:
        t0 = time.time()
        feats = c["feats"]
        bg_sups = c["bg_sups"]
        if len(bg_sups) > 0:
            bg_proto = feats[bg_sups].mean(0)
        else:
            bg_proto = feats.mean(0)
            print(f"[warn] {c['name']}: 边界带无可用背景超像素，退化为全图均值")

        D = unary_from_prototypes(feats, protos, valid, bg_proto)
        D, sigma = normalize_unary(D)

        hard = np.full(len(feats), -1, dtype=np.int32)
        for sup, cls in c["hard_seeds"].items():
            hard[sup] = cls                      # 单点 / 三角形 rho>=0.9 优先
        for sup in bg_sups:
            if hard[sup] == -1:
                hard[sup] = 0                    # 不含种子的边界带 → 背景
        edges = build_graph(c["seg"].astype(np.int32), c["mean_lab"], c["edge_bd"])

        sup_labels, energy, method = solve(D, edges, hard, pc["lam"], num_classes)

        order = np.sort(D, axis=1)
        d1 = D[np.arange(len(sup_labels)), sup_labels]
        d2 = order[:, 1]
        margin = np.clip((d2 - d1) / (d2 + d1 + 1e-6), 0.0, 1.0)

        seg = c["seg"].astype(np.int32)
        mask = sup_labels[seg].astype(np.uint8)

        # 三角形内部一致率（强制写入前统计），并做像素级强制写入
        tri_cons = []
        for inst, tm in zip([i for i in c["instances"] if i.kind == "triangle"], c["tri_masks"]):
            tri_cons.append(float((mask[tm] == inst.cls).mean()))
            mask[tm] = inst.cls

        conf = np.ones_like(mask, dtype=np.uint8) * 255

        stem = c["stem"]
        Image.fromarray(mask, mode="L").save(os.path.join(out_dir, f"{stem}_label.png"))
        Image.fromarray(conf, mode="L").save(os.path.join(out_dir, f"{stem}_conf.png"))

        # 可视化：原图 + 伪标签色块 + 单点 + 三角形轮廓
        color = PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]
        overlay = (0.5 * c["rgb"] + 0.5 * color).astype(np.uint8)
        overlay = draw_instances(overlay, c["instances"])
        vh, vw = overlay.shape[:2]
        if max(vh, vw) > 1024:
            s = 1024.0 / max(vh, vw)
            overlay = np.asarray(Image.fromarray(overlay).resize(
                (int(vw * s), int(vh * s)), Image.BILINEAR))
        Image.fromarray(overlay).save(os.path.join(vis_dir, f"{stem}.jpg"), quality=90)

        # 统计
        areas = {int(k): float(v) for k, v in
                 zip(*np.unique(mask, return_counts=True))}
        total = float(mask.size)
        ratios = {k: v / total for k, v in areas.items()}
        points = [(x, y, i.cls) for i in c["instances"] if i.kind == "point"
                  for x, y in i.points]
        point_ok = all(mask[int(round(y)), int(round(x))] == cls for x, y, cls in points)
        tri_mean = float(np.mean(tri_cons)) if tri_cons else None
        tri_min = float(np.min(tri_cons)) if tri_cons else None

        summary.append({
            "image": c["name"], "stem": stem, "shape": list(c["shape"]),
            "scale": c["scale"],
            "num_instances": len(c["instances"]),
            "num_points": len(points), "num_triangles": len(tri_cons),
            "num_superpixels": int(seg.max()) + 1, "method": method,
            "energy": float(energy), "sigma": float(sigma),
            "class_ratio": {str(k): round(v, 4) for k, v in sorted(ratios.items())},
            "margin_p50": round(float(np.percentile(margin, 50)), 3),
            "margin_p90": round(float(np.percentile(margin, 90)), 3),
            "point_consistent": bool(point_ok),
            "tri_consistency": None if tri_mean is None else round(tri_mean, 4),
            "tri_consistency_min": None if tri_min is None else round(tri_min, 4),
            "seconds": round(time.time() - t0, 2),
        })
        print(f"[2/2] {stem}  {method}  点一致={point_ok}  "
              f"三角形一致={'-' if tri_mean is None else f'{tri_mean:.3f}'}  "
              f"类别占比={ {k: round(v, 3) for k, v in sorted(ratios.items())} }")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    bad = [s["image"] for s in summary if not s["point_consistent"]]
    low_tri = [s["image"] for s in summary
               if s["tri_consistency"] is not None and s["tri_consistency"] < 0.95]
    n_inst = sum(s["num_instances"] for s in summary)
    n_pt = sum(s["num_points"] for s in summary)
    n_tri = sum(s["num_triangles"] for s in summary)
    print(f"\n完成 {len(summary)} 张：{n_inst} 实例（单点 {n_pt} + 三角形 {n_tri}），"
          f"伪标签输出到 {out_dir}")
    if bad:
        print(f"[error] 点一致性失败（硬约束可能有 bug）: {bad}")
    else:
        print("点一致性检查全部通过")
    if low_tri:
        print(f"[warn] 三角形内部一致率 < 0.95（需调 lambda / 检查原型）: {low_tri}")


if __name__ == "__main__":
    main()
