# -*- coding: utf-8 -*-
"""损失：点窗口 CE + 三角形内部 CE + 高置信伪标签 CE/Dice + 一致性 + 重构正则。"""
import torch
import torch.nn.functional as F


def point_ce(logits, target, weight, ignore=255):
    """点窗口 CE：逐样本按窗口像素数归一，再按 batch 平均。"""
    ce = F.cross_entropy(logits, target, ignore_index=ignore, reduction="none")
    denom = weight.sum(dim=(1, 2)) + 1e-6
    per_sample = (ce * weight).sum(dim=(1, 2)) / denom
    return per_sample.mean()


def tri_ce(logits, target, weight, ignore=255):
    """三角形内部 CE：每个实例先按自身面积（像素权重）归一，再按 batch 平均。

    batch 内没有三角形时 tri_w.sum()==0，该项自动为 0（不报错）。
    """
    ce = F.cross_entropy(logits, target, ignore_index=ignore, reduction="none")
    denom = weight.sum(dim=(1, 2)) + 1e-6
    per_sample = (ce * weight).sum(dim=(1, 2)) / denom
    has = (weight.sum(dim=(1, 2)) > 0).float()
    return (per_sample * has).sum() / (has.sum() + 1e-6)


def masked_ce(logits, target, mask, class_weight=None, ignore=255):
    """仅在高置信伪标签像素上计算 CE。"""
    ce = F.cross_entropy(logits, target, weight=class_weight,
                         ignore_index=ignore, reduction="none")
    m = mask.float()
    return (ce * m).sum() / (m.sum() + 1e-6)


def dice_loss(probs, target, mask, num_classes, ignore=255):
    """逐类 Dice，只对 batch 内出现的类取平均。"""
    valid = (mask > 0) & (target != ignore)
    if valid.sum() == 0:
        return probs.sum() * 0.0
    t = target.clamp(0, num_classes - 1)
    onehot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float()
    v = valid.unsqueeze(1).float()
    p = probs * v
    onehot = onehot * v

    dims = (0, 2, 3)
    inter = (p * onehot).sum(dims)
    denom = p.sum(dims) + onehot.sum(dims)
    dice = (2.0 * inter + 1.0) / (denom + 1.0)

    present = (onehot.sum(dims) > 0).float()
    return 1.0 - (dice * present).sum() / (present.sum() + 1e-6)


def consistency_loss(prob_weak, prob_strong):
    """弱/强两路一致性，弱侧 stop-grad。"""
    return F.mse_loss(prob_strong, prob_weak.detach())


def reconstruction_loss(recon, x):
    """x 为网络输入（已按 mean/std 归一化到 [-1,1]），先还原到 [0,1]。"""
    target = x * 0.5 + 0.5
    return F.mse_loss(recon, target)
