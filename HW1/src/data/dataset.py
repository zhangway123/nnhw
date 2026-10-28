# -*- coding: utf-8 -*-
"""图像 + 实例（单点/三角形）+ 伪标签的联合加载与增强（弱/强两路）。"""
import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from src.data.instances import (
    Instance,
    cap_triangle_area,
    instance_centers,
    instances_to_crop,
    triangle_mask,
)

IGNORE = 255


def load_points(labels_path):
    """读取 label.json → {stem: [(x, y, cls), ...]}，并返回条目列表。

    兼容旧调用：把三角形的 3 个顶点也当作普通点返回（仅供点准确率统计/兼容）。
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    points_map = {}
    for e in entries:
        stem = os.path.splitext(e["image name"])[0]
        cls = int(e["label"])
        pts = []
        for item in e.get("point", []):
            if isinstance(item, dict):
                pts.append((float(item["x"]), float(item["y"]), cls))
            else:
                pts.extend((float(p["x"]), float(p["y"]), cls) for p in item)
        points_map[stem] = pts
    return entries, points_map


def _load_mask(path, shape):
    """读取伪标签/置信度图；尺寸不一致时按最近邻对齐到图像尺寸。"""
    if path is None or not os.path.isfile(path):
        return None
    with Image.open(path) as im:
        arr = np.asarray(im, dtype=np.uint8)
    if arr.shape[:2] != shape[:2]:
        arr = np.asarray(Image.fromarray(arr).resize((shape[1], shape[0]), Image.NEAREST))
    return arr


def _scale_instances(instances, s):
    return [Instance(i.cls, i.kind, [(x * s, y * s) for x, y in i.points], i.area * s * s)
            for i in instances]


class SegDataset(Dataset):
    def __init__(self, images, instances_map, pseudo_dir=None, image_size=512,
                 num_classes=5, point_window=5, train=True, with_strong=True,
                 point_crop_prob=0.6, max_side=1024, ignore=IGNORE,
                 tri_area_ratio_cap=0.05):
        self.images = list(images)
        self.instances_map = instances_map or {}
        self.pseudo_dir = pseudo_dir
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.r = int(point_window) // 2
        self.train = bool(train)
        self.with_strong = bool(with_strong)
        self.point_crop_prob = float(point_crop_prob)
        self.max_side = int(max_side or 0)
        self.ignore = ignore
        self.tri_area_ratio_cap = float(tri_area_ratio_cap or 0)

    def __len__(self):
        return len(self.images)

    # ------------------------------------------------------------------
    def _pseudo_paths(self, stem):
        if not self.pseudo_dir:
            return None, None
        lab = os.path.join(self.pseudo_dir, f"{stem}_label.png")
        conf = os.path.join(self.pseudo_dir, f"{stem}_conf.png")
        return (lab if os.path.isfile(lab) else None,
                conf if os.path.isfile(conf) else None)

    def _crop_origin(self, H, W, centers):
        """训练时按一定概率以实例中心裁剪；否则随机裁剪。

        原图约 6000×4000、目标极小，纯随机 512 裁剪几乎切不到动物，
        因此用实例中心裁剪保证点/三角形监督与伪标签都落在目标附近。
        """
        size = self.image_size
        if H < size or W < size:
            return 0, 0
        if not self.train:
            return (W - size) // 2, (H - size) // 2

        if centers and random.random() < self.point_crop_prob:
            x, y, _ = random.choice(centers)
            # 抖动范围取 ±size/2：让实例在裁剪框内均匀分布，
            # 否则目标总在中心，模型会学成“中心即前景”的捷径
            x0 = int(round(x - size / 2 + random.uniform(-size / 2, size / 2)))
            y0 = int(round(y - size / 2 + random.uniform(-size / 2, size / 2)))
        else:
            x0 = random.randint(0, W - size)
            y0 = random.randint(0, H - size)
        x0 = int(np.clip(x0, 0, W - size))
        y0 = int(np.clip(y0, 0, H - size))
        return x0, y0

    def __getitem__(self, idx):
        path = self.images[idx]
        stem = os.path.splitext(os.path.basename(path))[0]

        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.uint8)
        H, W = img.shape[:2]
        instances = list(self.instances_map.get(stem, []))

        # 统一长边到 max_side：与推理 letterbox 的尺度保持一致，
        # 否则训练看到的是原分辨率的大目标、推理看到的是缩小后的小目标。
        if self.max_side and max(H, W) > self.max_side:
            s = self.max_side / float(max(H, W))
            nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
            img = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
            instances = _scale_instances(instances, s)
            H, W = nh, nw

        size = self.image_size
        # 小图先放大到至少 size，保证能裁出窗口
        if H < size or W < size:
            s = max(size / float(H), size / float(W))
            nh, nw = max(size, int(round(H * s))), max(size, int(round(W * s)))
            img = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
            instances = _scale_instances(instances, s)
            H, W = nh, nw

        lab_path, conf_path = self._pseudo_paths(stem)
        pseudo = _load_mask(lab_path, (H, W))
        conf = _load_mask(conf_path, (H, W))

        if self.tri_area_ratio_cap > 0:
            instances, _ = cap_triangle_area(instances, (H, W), self.tri_area_ratio_cap)

        x0, y0 = self._crop_origin(H, W, instance_centers(instances))
        img = img[y0:y0 + size, x0:x0 + size]
        if pseudo is not None:
            pseudo = pseudo[y0:y0 + size, x0:x0 + size]
        if conf is not None:
            conf = conf[y0:y0 + size, x0:x0 + size]

        flip = self.train and random.random() < 0.5
        insts = instances_to_crop(instances, x0, y0, size, flip=flip)
        if flip:
            img = np.ascontiguousarray(img[:, ::-1])
            if pseudo is not None:
                pseudo = np.ascontiguousarray(pseudo[:, ::-1])
            if conf is not None:
                conf = np.ascontiguousarray(conf[:, ::-1])

        # 三角形监督目标（区域级，先画），单点窗口随后覆盖（点优先）
        tri_target = np.full((size, size), self.ignore, dtype=np.int64)
        tri_weight = np.zeros((size, size), dtype=np.float32)
        n_tri = 0
        for inst in insts:
            if inst.kind != "triangle":
                continue
            tm = triangle_mask(inst.points, (size, size))
            if not tm.any():
                continue
            tri_target[tm] = int(inst.cls)
            tri_weight[tm] = 1.0
            n_tri += 1

        # 点监督目标与权重（5×5 窗口）
        target = np.full((size, size), self.ignore, dtype=np.int64)
        weight = np.zeros((size, size), dtype=np.float32)
        n_point = 0
        for inst in insts:
            if inst.kind != "point":
                continue
            n_point += 1
            x, y = inst.points[0]
            xi, yi = int(round(x)), int(round(y))
            x_lo, x_hi = max(0, xi - self.r), min(size, xi + self.r + 1)
            y_lo, y_hi = max(0, yi - self.r), min(size, yi + self.r + 1)
            target[y_lo:y_hi, x_lo:x_hi] = int(inst.cls)
            weight[y_lo:y_hi, x_lo:x_hi] = 1.0
        if n_point > 0:
            tri_weight[weight > 0] = 0.0       # 点窗口与三角形重叠时以点为准

        weak = self._to_tensor(img)
        strong = self._to_tensor(self._photometric(img)) if (self.train and self.with_strong) else weak

        if pseudo is None:
            pseudo_t = torch.full((size, size), self.ignore, dtype=torch.long)
            conf_t = torch.zeros((size, size), dtype=torch.float32)
            has_pseudo = False
        else:
            pseudo_t = torch.from_numpy(pseudo.astype(np.int64))
            if conf is None:
                conf_t = (pseudo_t != 0).float()
            else:
                conf_t = torch.from_numpy(conf.astype(np.float32) / 255.0)
            has_pseudo = True

        return {
            "weak": weak,
            "strong": strong,
            "point": torch.from_numpy(target),
            "point_weight": torch.from_numpy(weight),
            "tri": torch.from_numpy(tri_target),
            "tri_w": torch.from_numpy(tri_weight),
            "n_points": torch.tensor(float(n_point), dtype=torch.float32),
            "n_tri": torch.tensor(float(n_tri), dtype=torch.float32),
            "pseudo": pseudo_t,
            "conf": conf_t,
            "has_pseudo": torch.tensor(has_pseudo),
            "name": stem,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _to_tensor(img_u8):
        x = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1)
        return (x - 0.5) / 0.5

    @staticmethod
    def _photometric(img_u8):
        """强增强：只做光度变换，几何与弱增强保持一致。"""
        im = Image.fromarray(img_u8)
        im = ImageEnhance.Brightness(im).enhance(random.uniform(0.7, 1.3))
        im = ImageEnhance.Contrast(im).enhance(random.uniform(0.7, 1.3))
        im = ImageEnhance.Color(im).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        arr = np.asarray(im, dtype=np.float32)
        arr += np.random.normal(0.0, 4.0, arr.shape)
        return np.clip(arr, 0, 255).astype(np.uint8)
