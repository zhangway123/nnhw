# -*- coding: utf-8 -*-
"""
HW1 无监督流程（nonsupervise.md）—— 第一部分：数据预处理
    是train_eval_photos文件夹下的图像!
    1.1 生成 dataset.json：number / 图像名 / 长宽 / label(暂空) / 图像路径
    1.2 随机留出法（random holdout）：训练集 80% / 评估集 20%，保存 split.json
    1.3 图像缩放至 (512,512)，RGB 归一化到 (0,1)（在 VQDataset 加载时完成）

运行：python data_preprocess.py --data_dir train_eval_photos
后续：python train_vqvae.py（读取本步骤产物 dataset.json / split.json）
"""
import os
import json
import random
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="train_eval_photos",
                   help="图像文件夹（数据更多时可改为 train_eval_photos）")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--eval_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_dataset_json(data_dir, out_dir):
    """1.1 生成 JSON：number / image name / length & width / label(暂空) / image path"""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(exts))
    records = []
    for i, fn in enumerate(files):
        fp = os.path.join(data_dir, fn)
        with Image.open(fp) as im:
            w, h = im.size
        records.append({"number": i,
                        "image name": fn,
                        "width": w, "height": h,
                        "label": "",                          # 暂时留空
                        "path": os.path.abspath(fp)})
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records, json_path


def make_split(records, eval_ratio, seed):
    """1.2 随机留出法：训练集 80% / 评估集 20%（bootstrap 采样为备选）"""
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n_eval = int(len(idx) * eval_ratio)
    return idx[n_eval:], idx[:n_eval]                          # train_idx, eval_idx


def load_image(path, size=512):
    """1.3 缩放至 (512,512)，RGB 三个通道归一化到 (0,1)"""
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0            # (512,512,3)


class VQDataset(Dataset):
    """读图 → 缩放归一化 → CHW tensor"""
    def __init__(self, records, idx, size=512):
        self.items = [records[i] for i in idx]
        self.size = size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = load_image(self.items[i]["path"], self.size)       # HWC, [0,1]
        return torch.from_numpy(x).permute(2, 0, 1)            # CHW


def main():
    args = parse_args()
    records, _ = build_dataset_json(args.data_dir, args.out_dir)
    train_idx, eval_idx = make_split(records, args.eval_ratio, args.seed)
    split_path = os.path.join(args.out_dir, "split.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"train": train_idx, "eval": eval_idx}, f)
    print(f"总样本 {len(records)}，训练 {len(train_idx)}，评估 {len(eval_idx)}")
    print(f"产物：{os.path.join(args.out_dir, 'dataset.json')}、{split_path}")


if __name__ == "__main__":
    main()
