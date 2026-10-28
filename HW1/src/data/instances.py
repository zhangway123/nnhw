# -*- coding: utf-8 -*-
"""实例解析：label.json 的 point 元素既可能是单点 dict，也可能是 3 点三角形 list。

三角形是“内点集”（内部像素几乎全属同一只动物），不是完整掩码：
轮廓仍由图割与模型推断，评估时不得把它当成 GT。
"""
import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

MIN_TRI_AREA = 100.0


@dataclass
class Instance:
    cls: int                          # 图级 label
    kind: str                         # "point" | "triangle"
    points: list                      # [(x, y), ...]，1 个点或 3 个顶点（原图坐标）
    area: float = 0.0                 # 单点为 0；三角形为 |cross|/2

    def center(self):
        """单点返回自身；三角形返回质心（供“实例中心裁剪”与连通域保留）。"""
        if self.kind == "point":
            return float(self.points[0][0]), float(self.points[0][1])
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def polygon_area(points):
    """鞋带公式求多边形面积。"""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def parse_instances(entry, img_size=None):
    """把一个 label.json 条目解析成实例列表。

    img_size = (W, H)；给出时对越界顶点做裁剪并 warn。
    三角形顶点数 != 3 或面积 < 100 px² 时跳过并 warn。
    """
    name = entry.get("image name")
    cls = int(entry["label"])
    W = H = None
    if img_size is not None:
        W, H = int(img_size[0]), int(img_size[1])

    out = []
    for k, item in enumerate(entry.get("point", [])):
        if isinstance(item, dict):
            kind = "point"
            pts = [(float(item["x"]), float(item["y"]))]
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            kind = "triangle"
            pts = [(float(p["x"]), float(p["y"])) for p in item]
        else:
            print(f"[warn] {name}: 第 {k} 个实例格式无法识别（需 dict 或 3 点 list），跳过")
            continue

        if W is not None:
            clipped = []
            for x, y in pts:
                cx = float(np.clip(x, 0.0, W - 1.0))
                cy = float(np.clip(y, 0.0, H - 1.0))
                if (cx, cy) != (x, y):
                    print(f"[warn] {name}: 第 {k} 个实例顶点越界 ({x},{y}) → ({cx},{cy})")
                clipped.append((cx, cy))
            pts = clipped

        area = 0.0
        if kind == "triangle":
            area = polygon_area(pts)
            if area < MIN_TRI_AREA:
                print(f"[warn] {name}: 第 {k} 个三角形退化（面积 {area:.1f} px²），跳过")
                continue
        out.append(Instance(cls=cls, kind=kind, points=pts, area=float(area)))
    return out


def load_instances(labels_path):
    """返回 (原始条目列表, {stem: [Instance]})。"""
    with open(labels_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    mapping = {}
    for e in entries:
        stem = os.path.splitext(e["image name"])[0]
        mapping[stem] = parse_instances(e)
    return entries, mapping


def triangle_mask(points, shape):
    """在 shape=(H, W) 网格上填充三角形，返回布尔掩码。"""
    H, W = int(shape[0]), int(shape[1])
    canvas = Image.new("L", (W, H), 0)
    ImageDraw.Draw(canvas).polygon([(float(x), float(y)) for x, y in points], fill=1)
    return np.asarray(canvas, dtype=np.uint8).astype(bool)


def _clip_to_box(poly, xmin, ymin, xmax, ymax):
    """Sutherland–Hodgman：把凸多边形裁剪到矩形。"""
    def clip(poly, inside, intersect):
        if not poly:
            return []
        out = []
        prev = poly[-1]
        prev_in = inside(prev)
        for cur in poly:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
            prev, prev_in = cur, cur_in
        return out

    def isect_v(p, q, xc):
        x1, y1 = p
        x2, y2 = q
        if abs(x2 - x1) < 1e-9:
            return (xc, y1)
        t = (xc - x1) / (x2 - x1)
        return (xc, y1 + t * (y2 - y1))

    def isect_h(p, q, yc):
        x1, y1 = p
        x2, y2 = q
        if abs(y2 - y1) < 1e-9:
            return (x1, yc)
        t = (yc - y1) / (y2 - y1)
        return (x1 + t * (x2 - x1), yc)

    poly = list(poly)
    poly = clip(poly, lambda p: p[0] >= xmin, lambda p, q: isect_v(p, q, xmin))
    poly = clip(poly, lambda p: p[0] <= xmax, lambda p, q: isect_v(p, q, xmax))
    poly = clip(poly, lambda p: p[1] >= ymin, lambda p, q: isect_h(p, q, ymin))
    poly = clip(poly, lambda p: p[1] <= ymax, lambda p, q: isect_h(p, q, ymax))
    return poly


def instances_to_crop(instances, x0, y0, size, flip=False):
    """把实例坐标变换到 size×size 裁剪框内（可选水平翻转）。

    单点出框丢弃；三角形与裁剪框交集面积 / 自身面积 < 0.5 丢弃，
    否则保留并把顶点 clip 到框内。
    """
    size = float(size)
    out = []
    for inst in instances:
        pts = [(x - x0, y - y0) for x, y in inst.points]

        if inst.kind == "point":
            x, y = pts[0]
            if not (0 <= x < size and 0 <= y < size):
                continue
            nx = size - 1 - x if flip else x
            out.append(Instance(inst.cls, "point", [(nx, y)], 0.0))
            continue

        clipped = _clip_to_box(pts, 0.0, 0.0, size, size)
        if len(clipped) < 3:
            continue
        if polygon_area(clipped) < 0.5 * inst.area:
            continue
        if flip:
            clipped = [(size - 1 - x, y) for x, y in clipped]
        out.append(Instance(inst.cls, "triangle", clipped, inst.area))
    return out


def instance_centers(instances):
    """[(x, y, cls), ...]；单点返回自身，三角形返回质心。"""
    out = []
    for inst in instances:
        cx, cy = inst.center()
        out.append((cx, cy, inst.cls))
    return out


def cap_triangle_area(instances, img_size, ratio_cap=0.05):
    """把面积占图比例超过 ratio_cap 的三角形退化为质心单点。

    超大三角形整块回填会给出“大片区域都属于某一类”的强先验，导致该类在
    推理时外扩（实测 0838 占图 10.4% 的三角形把类 2 放大到 50%）。
    返回 (新实例列表, 被收缩的 [(cls, 面积占比), ...])，调用方可据此打日志。
    """
    H, W = int(img_size[0]), int(img_size[1])
    if ratio_cap is None or ratio_cap <= 0:
        return list(instances), []
    cap_area = float(ratio_cap) * float(H) * float(W)

    out, capped = [], []
    for inst in instances:
        if inst.kind == "triangle" and inst.area > cap_area:
            cx, cy = inst.center()
            out.append(Instance(inst.cls, "point", [(cx, cy)], 0.0))
            capped.append((int(inst.cls), inst.area / float(H * W)))
        else:
            out.append(inst)
    return out, capped
