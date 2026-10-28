# -*- coding: utf-8 -*-
"""阶段 5：仅对 test_photo 推理并输出语义分割图（mask / color / overlay / json）。

用法：
    python src/test.py --ckpt checkpoints/best.pth --data test_photo --out outputs/pred/test_photo
    python src/test.py --ckpt checkpoints/best.pth --gt path/to/gt_dir   # 有真值时额外算指标
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.instances import load_instances                    # noqa: E402
from src.inference import predict_mask, predict_tiled            # noqa: E402
from src.metrics import SegMetrics                              # noqa: E402
from src.postprocess import remove_small_components, save_outputs  # noqa: E402
from src.train import build_model                               # noqa: E402
from src.utils import ensure_dir, get_device, list_images, load_config, resolve  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--ckpt", type=str, default="checkpoints/best.pth")
    p.add_argument("--data", type=str, default=None, help="默认取 config 的 data.test_dir")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--gt", type=str, default=None, help="可选：真值目录（同名 PNG）")
    p.add_argument("--no-slic", action="store_true", help="关闭 SLIC 投票后处理")
    p.add_argument("--no-tiled", action="store_true", help="退回 letterbox 整图推理")
    p.add_argument("--tiled", action="store_true", help="强制启用滑窗推理")
    p.add_argument("--min-ratio", type=float, default=None,
                   help="去小连通域阈值（相对图像面积）；默认取 config，大图小目标不能用 0.005")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--labeled-only", action="store_true",
                   help="只跑 label.json 里有标注的图（人工核验用）")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dev = torch.device(args.device) if args.device else get_device(cfg)
    num_classes = int(cfg["data"]["num_classes"])
    size = int(args.size or cfg["infer"]["size"])
    min_ratio = float(args.min_ratio if args.min_ratio is not None
                      else cfg["infer"].get("min_ratio", 0.0005))
    data_dir = resolve(args.data or cfg["data"]["test_dir"])
    out_dir = ensure_dir(args.out or cfg["infer"]["out_dir"])

    model = build_model(cfg, num_classes).to(dev)
    ckpt = torch.load(resolve(args.ckpt), map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    print(f"设备: {dev}  权重: {args.ckpt}  测试目录: {data_dir}")

    images = sorted(list_images(data_dir))
    if args.labeled_only:
        _, inst_map = load_instances(resolve(cfg["data"]["labels"]))
        images = [p for p in images
                  if os.path.splitext(os.path.basename(p))[0] in inst_map]
    if args.limit:
        images = images[: args.limit]
    if len(images) == 0:
        raise RuntimeError(f"{data_dir} 下没有图像")

    metrics = SegMetrics(num_classes, ignore=int(cfg["data"]["ignore_index"])) if args.gt else None
    summary = []

    for path in images:
        t0 = time.time()
        stem = os.path.splitext(os.path.basename(path))[0]
        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"))

        use_tiled = (args.tiled or bool(cfg["infer"].get("use_tiled", False))) \
            and not args.no_tiled
        if use_tiled:
            mask = predict_tiled(
                model, img, dev,
                tile=int(cfg["infer"].get("tile", 512)),
                stride=int(cfg["infer"].get("stride", 384)),
                use_amp=bool(cfg["train"]["amp"]) and dev.type == "cuda",
            )
            info = {"orig_size": (img.shape[1], img.shape[0]), "tiled": True,
                    "tile": int(cfg["infer"].get("tile", 512)),
                    "stride": int(cfg["infer"].get("stride", 384))}
        else:
            mask, _, info = predict_mask(
                model, img, size, dev,
                use_amp=bool(cfg["train"]["amp"]) and dev.type == "cuda",
                use_slic=not args.no_slic,
            )
            info = dict(info, tiled=False)
        mask = remove_small_components(mask, num_classes, min_ratio)

        meta = {
            "image_name": os.path.basename(path),
            "orig_size": list(info["orig_size"]),
            "tiled": bool(info.get("tiled", False)),
        }
        if info.get("tiled"):
            meta.update({"tile": info.get("tile"), "stride": info.get("stride")})
        else:
            meta.update({
                "letterbox": {"size": size, "scale": info["scale"],
                              "pad": list(info["pad"])},
                "slic_vote": not args.no_slic,
            })
        paths = save_outputs(img, mask, out_dir, stem, meta=meta)

        if metrics is not None:
            gt_path = os.path.join(resolve(args.gt), f"{stem}.png")
            if os.path.isfile(gt_path):
                gt = np.asarray(Image.open(gt_path), dtype=np.uint8)
                if gt.shape != mask.shape:
                    gt = np.asarray(Image.fromarray(gt).resize(
                        (mask.shape[1], mask.shape[0]), Image.NEAREST))
                metrics.update(mask, gt)
            else:
                print(f"[warn] 缺少真值 {gt_path}")

        elapsed = time.time() - t0
        ratios = {int(k): round(float(v) / mask.size, 4)
                  for k, v in zip(*np.unique(mask, return_counts=True))}
        summary.append({"image": stem, "mask": paths["mask"], "class_ratio": ratios,
                        "seconds": round(elapsed, 2)})
        print(f"{stem}: {mask.shape[1]}x{mask.shape[0]}  用时 {elapsed:.1f}s  "
              f"类别占比 {ratios}")

    payload = {"out_dir": out_dir, "num_images": len(summary), "items": summary}
    if metrics is not None:
        payload["metrics"] = metrics.compute()
        print("\n评估指标:")
        for k, v in payload["metrics"].items():
            if isinstance(v, dict):
                continue
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print("  每类 IoU:", {k: (round(v, 4) if v is not None else None)
                             for k, v in payload["metrics"]["per_class_iou"].items()})

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n完成 {len(summary)} 张，结果输出到 {out_dir}")


if __name__ == "__main__":
    main()
