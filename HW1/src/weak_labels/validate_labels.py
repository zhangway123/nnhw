# -*- coding: utf-8 -*-
"""校验 outputs/label.json：单点/三角形混排、越界、退化三角形、缺图。

用法：
    python src/weak_labels/validate_labels.py
    python src/weak_labels/validate_labels.py --labels outputs/label.json --save outputs/label_stats.json
"""
import argparse
import json
import os
import sys
from collections import Counter

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.instances import (  # noqa: E402
    MIN_TRI_AREA,
    parse_instances,
    polygon_area,
)
from src.utils import load_config, resolve  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=str, default=None, help="默认取 config 中的 data.labels")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--save", type=str, default="outputs/label_stats.json")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    labels_path = resolve(args.labels or cfg["data"]["labels"])
    train_dir = resolve(cfg["data"]["train_dir"])
    num_classes = int(cfg["data"]["num_classes"])

    with open(labels_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    errors, warnings = [], []
    seen = set()
    cls_inst = Counter()
    cls_points = Counter()
    cls_images = Counter()
    total_inst = total_pt = total_tri = total_raw = 0
    rows = []

    for e in entries:
        name = e.get("image name")
        label = e.get("label")
        raw = e.get("point", [])

        if name in seen:
            errors.append(f"{name}: 重复条目")
        seen.add(name)

        path = os.path.join(train_dir, name) if name else None
        if not path or not os.path.isfile(path):
            errors.append(f"{name}: 图像不存在于 {train_dir}")
            continue

        try:
            cid = int(label)
        except (TypeError, ValueError):
            errors.append(f"{name}: label 无法转 int（{label!r}）")
            continue
        if not (1 <= cid < num_classes):
            errors.append(f"{name}: label={cid} 超出 1..{num_classes - 1}")
        if len(raw) == 0:
            warnings.append(f"{name}: 没有任何实例")

        with Image.open(path) as im:
            W, H = im.size

        n_raw = 0
        n_pts = n_tri = 0
        for k, item in enumerate(raw):
            if isinstance(item, dict):
                n_raw += 1
                n_pts += 1
                x, y = item.get("x"), item.get("y")
                if x is None or y is None:
                    errors.append(f"{name}: 第 {k} 个点缺少 x/y")
                elif not (0 <= x < W and 0 <= y < H):
                    errors.append(f"{name}: 第 {k} 个点越界 ({x},{y}) / 图像 {W}x{H}")
                continue

            if not isinstance(item, (list, tuple)):
                errors.append(f"{name}: 第 {k} 个实例既不是 dict 也不是 list")
                continue
            if len(item) != 3:
                errors.append(f"{name}: 第 {k} 个三角形顶点数 = {len(item)}（应为 3）")
                continue
            n_raw += 3
            n_tri += 1
            pts = []
            for j, p in enumerate(item):
                if not isinstance(p, dict) or p.get("x") is None or p.get("y") is None:
                    errors.append(f"{name}: 第 {k} 个三角形第 {j} 个顶点缺少 x/y")
                    pts = []
                    break
                if not (0 <= p["x"] < W and 0 <= p["y"] < H):
                    errors.append(f"{name}: 第 {k} 个三角形第 {j} 个顶点越界 "
                                  f"({p['x']},{p['y']}) / 图像 {W}x{H}")
                pts.append((float(p["x"]), float(p["y"])))
            if len(pts) == 3:
                area = polygon_area(pts)
                if area < MIN_TRI_AREA:
                    errors.append(f"{name}: 第 {k} 个三角形退化（面积 {area:.1f} px²）")

        parsed = parse_instances(e, (W, H))
        n_parsed_pts = sum(1 for i in parsed if i.kind == "point")
        n_parsed_tri = sum(1 for i in parsed if i.kind == "triangle")
        if n_parsed_pts != n_pts or n_parsed_tri != n_tri:
            errors.append(f"{name}: 解析结果 {n_parsed_pts} 点 + {n_parsed_tri} 三角形"
                          f"，与原始 {n_pts} 点 + {n_tri} 三角形不一致（解析丢点）")

        cls_inst[cid] += n_pts + n_tri
        cls_points[cid] += n_pts
        cls_images[cid] += 1
        total_inst += n_pts + n_tri
        total_pt += n_pts
        total_tri += n_tri
        total_raw += n_raw
        rows.append({"image": name, "label": cid, "instances": n_pts + n_tri,
                     "points": n_pts, "triangles": n_tri, "raw_points": n_raw,
                     "size": [W, H]})

    print(f"标注文件: {labels_path}")
    print(f"训练目录: {train_dir}")
    print(f"图像数: {len(entries)}  实例数: {total_inst}"
          f"（单点 {total_pt} + 三角形 {total_tri}）  原始点数: {total_raw}  类别数: {num_classes - 1}")
    print("\n逐图统计:")
    for r in rows:
        print(f"  {r['image']:<22} label={r['label']}  实例={r['instances']} "
              f"(单点 {r['points']} + 三角形 {r['triangles']})  原始点={r['raw_points']}")
    print("\n各类统计:")
    for c in range(1, num_classes):
        print(f"  class {c}: {cls_images[c]} 张图 / {cls_inst[c]} 实例"
              f"（单点 {cls_points[c]}）" + ("  (缺失)" if cls_images[c] == 0 else ""))

    for w in warnings:
        print(f"[warn] {w}")
    for err in errors:
        print(f"[error] {err}")

    stats = {
        "labels": labels_path,
        "train_dir": train_dir,
        "num_images": len(entries),
        "num_instances": total_inst,
        "num_points": total_pt,
        "num_triangles": total_tri,
        "num_raw_points": total_raw,
        "per_class": {str(c): {"images": cls_images[c], "instances": cls_inst[c],
                               "points": cls_points[c]}
                      for c in range(1, num_classes)},
        "per_image": rows,
        "warnings": warnings,
        "errors": errors,
    }
    if args.save:
        out = resolve(args.save)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n统计已保存: {out}")

    if errors:
        print(f"\n校验未通过：{len(errors)} 个错误")
        sys.exit(1)
    print("\n校验通过")


if __name__ == "__main__":
    main()
