# -*- coding: utf-8 -*-
"""通用工具：配置、路径、随机种子、设备。"""
import os
import random

import numpy as np
import torch
import yaml

# HW1 根目录（src 的上一级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path):
    """相对路径统一按 HW1 根目录解析。"""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_config(path=None):
    path = resolve(path or "configs/config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    path = resolve(path)
    os.makedirs(path, exist_ok=True)
    return path


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg=None):
    want = (cfg or {}).get("device", "auto")
    if want == "cpu":
        return torch.device("cpu")
    if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] 配置要求 cuda，但当前环境不可用，回退到 cpu")
    return torch.device("cpu")


def list_images(directory, exts=(".jpg", ".jpeg", ".png")):
    directory = resolve(directory)
    names = sorted(n for n in os.listdir(directory) if n.lower().endswith(exts))
    return [os.path.join(directory, n) for n in names]
