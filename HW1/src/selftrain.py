# -*- coding: utf-8 -*-
"""阶段 4：自训练（EMA 教师重生成伪标签 → 置信度过滤 → 点优先 → 继续训练）。

用法：
    python src/selftrain.py --rounds 3
    python src/selftrain.py --init checkpoints/best.pth --rounds 1 --limit 3
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.instances import (                               # noqa: E402
    cap_triangle_area,
    load_instances,
    triangle_mask,
)
from src.inference import predict_probs, predict_tiled         # noqa: E402
from src.postprocess import (                                  # noqa: E402
    force_points,
    keep_point_components,
    letterbox_image,
    remove_small_components,
    slic_vote,
    unletterbox_mask,
)
from src.train import _find_image, build_model, run_training     # noqa: E402
from src.utils import (                                        # noqa: E402
    ensure_dir,
    get_device,
    list_images,
    load_config,
    resolve,
    seed_everything,
)
from src.weak_labels.slic_features import make_superpixels     # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--init", type=str, default="checkpoints/best.pth")
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="每轮训练轮数（覆盖 config）")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 张图（调试）")
    return p.parse_args()


def _unletterbox_float(arr, info):
    left, top = info["pad"]
    nw, nh = info["new_size"]
    W, H = info["orig_size"]
    crop = arr[top:top + nh, left:left + nw]
    out = np.asarray(Image.fromarray(crop.astype(np.float32), mode="F")
                     .resize((W, H), Image.NEAREST))
    return np.array(out, dtype=np.float32, copy=True)


@torch.no_grad()
def generate_pseudo(model, images, out_dir, size, device, num_classes,
                    instances_map, min_ratio=0.0005, use_amp=False, use_slic=True,
                    use_tiled=True, tile=512, stride=384, tri_area_ratio_cap=0.05):
    ensure_dir(out_dir)
    stats = []
    model.eval()
    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"))

        if use_tiled:
            mask, conf = predict_tiled(model, img, device, tile=tile, stride=stride,
                                       use_amp=use_amp, return_conf=True)
        else:
            prob, info, _ = predict_probs(model, img, size, device, use_amp)
            if use_slic:
                lb, _ = letterbox_image(img, size)
                seg, _ = make_superpixels(lb, 4000, 15.0)
                mask_lb = slic_vote(prob, seg)
            else:
                mask_lb = prob.argmax(0).astype(np.uint8)
            mask = unletterbox_mask(mask_lb, info)
            conf = _unletterbox_float(prob.max(0), info)

        mask = remove_small_components(mask, num_classes, min_ratio)
        instances = instances_map.get(stem, [])
        if instances:
            instances, _ = cap_triangle_area(instances, mask.shape, tri_area_ratio_cap)
            # 实例锚点：单点用自身，三角形用质心，保留其所在连通域
            anchors = [inst.center() + (inst.cls,) for inst in instances]
            mask = keep_point_components(mask, anchors, num_classes, min_ratio)
            for inst in instances:
                if inst.kind == "point":
                    mask = force_points(mask, [(x, y, inst.cls) for x, y in inst.points])
                    for x, y in inst.points:
                        yi, xi = int(round(y)), int(round(x))
                        conf[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3] = 1.0
                else:
                    tm = triangle_mask(inst.points, mask.shape)
                    mask[tm] = inst.cls          # 三角形内部整块回填
                    conf[tm] = 1.0               # 不参与置信度过滤

        Image.fromarray(mask.astype(np.uint8), mode="L").save(
            os.path.join(out_dir, f"{stem}_label.png"))
        Image.fromarray((np.clip(conf, 0, 1) * 255).astype(np.uint8), mode="L").save(
            os.path.join(out_dir, f"{stem}_conf.png"))

        ratios = {int(k): float(v) / mask.size
                  for k, v in zip(*np.unique(mask, return_counts=True))}
        stats.append({"image": stem, "class_ratio": {str(k): round(v, 4)
                                                     for k, v in sorted(ratios.items())}})
        print(f"  伪标签 {stem}: { {k: round(v, 3) for k, v in sorted(ratios.items())} }")
    return stats


def ema_update(teacher, student, decay):
    with torch.no_grad():
        for t, s in zip(teacher.parameters(), student.parameters()):
            t.mul_(decay).add_(s.detach(), alpha=1.0 - decay)
        for t, s in zip(teacher.buffers(), student.buffers()):
            t.copy_(s)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    dev = torch.device(args.device) if args.device else get_device(cfg)
    rounds = int(args.rounds or cfg["selftrain"]["rounds"])
    size = int(cfg["selftrain"]["infer_size"])
    decay = float(cfg["selftrain"]["ema_decay"])
    num_classes = int(cfg["data"]["num_classes"])

    labels_path = resolve(cfg["data"]["labels"])
    _, inst_map = load_instances(labels_path)
    train_dir = resolve(cfg["data"]["train_dir"])
    val_stem = os.path.splitext(cfg["data"]["val_image"])[0]

    images = [p for p in sorted(list_images(train_dir))
              if os.path.splitext(os.path.basename(p))[0] != val_stem]
    if args.limit:
        images = images[: args.limit]
    # 验证图也要生成伪标签：否则每轮的 val mIoU 无从计算，无法判断是否漂移
    pseudo_images = images
    val_path = _find_image(train_dir, val_stem)
    if val_path is not None:
        pseudo_images = images + [val_path]

    model = build_model(cfg, num_classes).to(dev)
    ckpt = torch.load(resolve(args.init), map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    teacher = copy.deepcopy(model).eval()

    init_ckpt = args.init
    all_stats = []
    for r in range(1, rounds + 1):
        print(f"\n===== 自训练第 {r}/{rounds} 轮：教师生成伪标签（{len(pseudo_images)} 张，含验证图） =====")
        out_dir = resolve(f"outputs/pseudo_r{r}")
        stats = generate_pseudo(teacher, pseudo_images, out_dir, size, dev, num_classes,
                                inst_map,
                                use_tiled=bool(cfg["infer"].get("use_tiled", True)),
                                tile=int(cfg["infer"].get("tile", 512)),
                                stride=int(cfg["infer"].get("stride", 384)),
                                tri_area_ratio_cap=float(
                                    cfg["pseudo"].get("tri_area_ratio_cap", 0) or 0))
        all_stats.append({"round": r, "pseudo_dir": out_dir, "images": stats})

        print(f"===== 自训练第 {r}/{rounds} 轮：继续训练 =====")
        _, best = run_training(
            cfg, images=images, pseudo_dir=out_dir, init_ckpt=init_ckpt,
            tag=f"r{r}_", device=dev, epochs=args.epochs,
        )

        student_ckpt = torch.load(resolve(best), map_location=dev, weights_only=False)
        student = build_model(cfg, num_classes).to(dev)
        student.load_state_dict(student_ckpt["model"])
        ema_update(teacher, student, decay)
        init_ckpt = best

    with open(resolve("outputs/selftrain_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n自训练完成，{rounds} 轮；汇总 outputs/selftrain_summary.json")


if __name__ == "__main__":
    main()
