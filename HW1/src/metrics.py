# -*- coding: utf-8 -*-
"""分割指标：mIoU / mAcc / PA / Dice / Boundary IoU / 每类 IoU。"""
import numpy as np
from scipy import ndimage as ndi


class SegMetrics:
    def __init__(self, num_classes, ignore=255, boundary_d=2):
        self.C = int(num_classes)
        self.ignore = ignore
        self.boundary_d = int(boundary_d)
        self.cm = np.zeros((self.C, self.C), dtype=np.float64)
        self.b_inter = np.zeros(self.C, dtype=np.float64)
        self.b_union = np.zeros(self.C, dtype=np.float64)

    def update(self, pred, gt):
        pred = np.asarray(pred).astype(np.int64)
        gt = np.asarray(gt).astype(np.int64)
        valid = gt != self.ignore

        idx = gt[valid] * self.C + pred[valid]
        self.cm += np.bincount(idx, minlength=self.C * self.C).reshape(self.C, self.C)

        for c in range(self.C):
            pm = pred == c
            gm = (gt == c) & valid
            if not pm.any() and not gm.any():
                continue
            band = ndi.binary_dilation(gm, iterations=self.boundary_d) | \
                ndi.binary_dilation(pm, iterations=self.boundary_d)
            inter = np.logical_and(np.logical_and(pm, gm), band).sum()
            union = np.logical_and(np.logical_or(pm, gm), band).sum()
            self.b_inter[c] += float(inter)
            self.b_union[c] += float(union)

    def compute(self):
        cm = self.cm
        tp = np.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        present = cm.sum(1) > 0

        iou = np.where((tp + fp + fn) > 0, tp / np.maximum(tp + fp + fn, 1e-9), np.nan)
        dice = np.where((2 * tp + fp + fn) > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9), np.nan)
        acc = np.where(present, tp / np.maximum(cm.sum(1), 1e-9), np.nan)
        biou = np.where(self.b_union > 0, self.b_inter / np.maximum(self.b_union, 1e-9), np.nan)

        def _mean(a):
            v = a[~np.isnan(a)]
            return float(v.mean()) if v.size else float("nan")

        total = cm.sum()
        return {
            "mIoU": _mean(iou),
            "mAcc": _mean(acc),
            "PA": float(tp.sum() / max(total, 1e-9)),
            "Dice": _mean(dice),
            "BoundaryIoU": _mean(biou),
            "per_class_iou": {int(i): (None if np.isnan(v) else float(v))
                              for i, v in enumerate(iou)},
            "per_class_dice": {int(i): (None if np.isnan(v) else float(v))
                               for i, v in enumerate(dice)},
        }


def point_accuracy(pred, points):
    """预测掩码在点处与标注类别一致的比例。"""
    if not points:
        return float("nan")
    H, W = pred.shape
    ok = 0
    for x, y, cls in points:
        yi = int(np.clip(int(round(y)), 0, H - 1))
        xi = int(np.clip(int(round(x)), 0, W - 1))
        ok += int(pred[yi, xi]) == int(cls)
    return ok / float(len(points))
