# -*- coding: utf-8 -*-
"""阶段 3：VQ-U-Net 弱监督训练（点 CE + 伪标签 CE/Dice + 一致性 + 重构正则）。

用法：
    python src/train.py
    python src/train.py --epochs 2 --smoke
    python src/train.py --init checkpoints/best.pth --pseudo-dir outputs/pseudo_r1 --tag r1_
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset import SegDataset, load_points          # noqa: E402
from src.data.instances import load_instances, triangle_mask   # noqa: E402
from src.inference import predict_mask, predict_tiled        # noqa: E402
from src.losses import (                                      # noqa: E402
    consistency_loss,
    dice_loss,
    masked_ce,
    point_ce,
    reconstruction_loss,
    tri_ce,
)
from src.metrics import point_accuracy                        # noqa: E402
from src.models.vq_unet import VQUNet                         # noqa: E402
from src.utils import (                                       # noqa: E402
    ensure_dir,
    get_device,
    list_images,
    load_config,
    resolve,
    seed_everything,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--pseudo-dir", type=str, default=None)
    p.add_argument("--init", type=str, default=None, help="初始化权重（自训练用）")
    p.add_argument("--tag", type=str, default="", help="产物前缀，避免覆盖")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--smoke", action="store_true", help="冒烟测试：限量 step + 小尺寸")
    return p.parse_args()


def build_model(cfg, num_classes):
    m = cfg["model"]
    return VQUNet(
        num_classes=num_classes,
        enc_channels=tuple(m["enc_channels"]),
        codebook_size=m["codebook_size"],
        code_dim=m["code_dim"],
        beta=m["beta"],
        ema_decay=m["ema_decay"],
        downsample=m["downsample"],
        groups=m["groups"],
        restart_every=m.get("restart_every", 100),
    )


def compute_class_weights(pseudo_dir, stems, num_classes, stride=4):
    """按类别频率的平方根反比加权。

    直接用 1/freq 会把背景压到 0.1，模型会退化成“全前景”；平方根更稳。
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for stem in stems:
        path = os.path.join(pseudo_dir, f"{stem}_label.png")
        if not os.path.isfile(path):
            continue
        arr = np.asarray(Image.open(path), dtype=np.uint8)[::stride, ::stride]
        counts += np.bincount(arr.reshape(-1), minlength=num_classes)[:num_classes]
    freq = counts / max(counts.sum(), 1.0)
    w = np.ones_like(freq)
    present = freq > 0
    if present.any():
        w[present] = 1.0 / np.sqrt(freq[present])
        w[present] /= w[present].mean()
    return torch.tensor(np.clip(w, 0.2, 5.0), dtype=torch.float32)


def _find_image(directory, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(directory, stem + ext)
        if os.path.isfile(path):
            return path
    return None


@torch.no_grad()
def validate(model, image_path, points, instances, pseudo_path, conf_path, num_classes,
             size, device, tau, use_amp, use_tiled=True, tile=512, stride=384):
    model.eval()
    with Image.open(image_path) as im:
        img = np.asarray(im.convert("RGB"))
    if use_tiled:
        mask = predict_tiled(model, img, device, tile=tile, stride=stride,
                             use_amp=use_amp)
    else:
        mask, _, _ = predict_mask(model, img, size, device, use_amp=use_amp, use_slic=False)

    acc = point_accuracy(mask, points) if points else float("nan")

    tri_cons = [(float((mask[triangle_mask(i.points, mask.shape)] == i.cls).mean()))
                for i in instances if i.kind == "triangle"]
    tri_acc = float(np.mean(tri_cons)) if tri_cons else float("nan")

    miou = float("nan")
    if pseudo_path and os.path.isfile(pseudo_path):
        gt = np.asarray(Image.open(pseudo_path), dtype=np.uint8)
        if gt.shape != mask.shape:
            gt = np.asarray(Image.fromarray(gt).resize(mask.shape[::-1], Image.NEAREST))
        if conf_path and os.path.isfile(conf_path):
            conf = np.asarray(Image.open(conf_path), dtype=np.uint8)
            if conf.shape != mask.shape:
                conf = np.asarray(Image.fromarray(conf).resize(mask.shape[::-1], Image.NEAREST))
            sel = conf.astype(np.float32) / 255.0 >= tau
        else:
            sel = np.ones_like(gt, dtype=bool)
        if sel.any():
            g = gt[sel].astype(np.int64)
            p = mask[sel].astype(np.int64)
            cm = np.bincount(g * num_classes + p,
                             minlength=num_classes * num_classes).reshape(num_classes, num_classes)
            tp = np.diag(cm).astype(np.float64)
            fp = cm.sum(0) - tp
            fn = cm.sum(1) - tp
            iou = tp / np.maximum(tp + fp + fn, 1.0)
            present = cm.sum(1) > 0
            miou = float(iou[present].mean()) if present.any() else 0.0

    model.train()
    # 按文档 4.3：早停与选模以验证 mIoU 为准（点只有 1 个，单独用会把“全前景”选成最优）
    score = miou if not np.isnan(miou) else (0.0 if np.isnan(acc) else acc)
    return score, acc, miou, tri_acc, mask


def run_training(cfg, images=None, pseudo_dir=None, init_ckpt=None, tag="",
                 epochs=None, batch_size=None, image_size=None, max_steps=0,
                 device=None):
    seed_everything(cfg.get("seed", 42))
    dev = device or get_device(cfg)
    tc = cfg["train"]
    num_classes = int(cfg["data"]["num_classes"])

    image_size = int(image_size or tc["image_size"])
    batch_size = int(batch_size or tc["batch_size"])
    epochs = int(epochs or tc["epochs"])
    pseudo_dir = resolve(pseudo_dir or cfg["data"]["pseudo_dir"])
    train_dir = resolve(cfg["data"]["train_dir"])
    labels_path = resolve(cfg["data"]["labels"])
    val_stem = os.path.splitext(cfg["data"]["val_image"])[0]
    tau = float(tc["tau"])

    _, points_map = load_points(labels_path)
    _, inst_map = load_instances(labels_path)

    if images is None:
        images = sorted(list_images(train_dir))
        images = [p for p in images
                  if os.path.isfile(os.path.join(
                      pseudo_dir, os.path.splitext(os.path.basename(p))[0] + "_label.png"))]
    images = [p for p in images
              if os.path.splitext(os.path.basename(p))[0] != val_stem]
    if len(images) == 0:
        raise RuntimeError(f"{pseudo_dir} 下没有可训练样本（需 {stem}_label.png）")

    base_count = len(images)
    target = int(tc.get("samples_per_epoch", 128))
    repeat = max(1, int(np.ceil(target / max(base_count, 1))))
    images = images * repeat

    print(f"设备: {dev}  训练样本: {len(images)}（{base_count} 张 × {repeat}）  伪标签目录: {pseudo_dir}")
    if dev.type == "cpu":
        print("[warn] 当前为 CPU 训练，速度会明显偏慢；确认 CUDA 驱动/PyTorch cu128 是否正常")

    ds = SegDataset(images, inst_map, pseudo_dir=pseudo_dir,
                    image_size=image_size, num_classes=num_classes,
                    point_window=int(tc["point_window"]), train=True,
                    point_crop_prob=float(tc.get("point_crop_prob", 0.6)),
                    max_side=int(cfg["data"].get("max_side", 0)),
                    tri_area_ratio_cap=float(cfg["pseudo"].get("tri_area_ratio_cap", 0) or 0))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=int(tc["num_workers"]), drop_last=False,
                    pin_memory=(dev.type == "cuda"))

    model = build_model(cfg, num_classes).to(dev)
    if init_ckpt:
        ckpt = torch.load(resolve(init_ckpt), map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"初始化权重: {init_ckpt}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]),
                            weight_decay=float(tc["weight_decay"]))

    total_steps = max_steps if max_steps > 0 else max(1, epochs * len(dl))
    warm = max(1, int(total_steps * float(tc["warmup_ratio"])))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + np.cos(np.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    use_amp = bool(tc["amp"]) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    cw = compute_class_weights(pseudo_dir, [os.path.splitext(os.path.basename(p))[0]
                                            for p in images], num_classes).to(dev)
    print("类别权重:", [round(float(v), 3) for v in cw.cpu()])

    ckpt_dir = ensure_dir(tc["ckpt_dir"])
    log_dir = ensure_dir(tc["log_dir"])
    log_path = os.path.join(log_dir, f"train_log{('_' + tag) if tag else ''}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "loss", "point", "tri", "pseudo_ce", "dice", "consist", "recon",
            "vq", "lr", "used_codes", "val_score", "val_point", "val_miou",
            "val_tri", "sec",
        ])

    val_path = _find_image(train_dir, val_stem)
    val_pseudo = os.path.join(pseudo_dir, f"{val_stem}_label.png")
    val_conf = os.path.join(pseudo_dir, f"{val_stem}_conf.png")
    val_points = points_map.get(val_stem, [])
    val_instances = inst_map.get(val_stem, [])
    infer_size = int(cfg["selftrain"]["infer_size"])

    best_score = -1.0
    best_epoch = 0
    patience = int(tc.get("patience", 0))
    step = 0
    model.train()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        agg = {k: 0.0 for k in ("loss", "point", "tri", "pseudo_ce", "dice",
                                "consist", "recon", "vq")}
        n_batch = 0

        for batch in dl:
            weak = batch["weak"].to(dev, non_blocking=True)
            strong = batch["strong"].to(dev, non_blocking=True)
            point = batch["point"].to(dev, non_blocking=True)
            weight = batch["point_weight"].to(dev, non_blocking=True)
            tri = batch["tri"].to(dev, non_blocking=True)
            tri_w = batch["tri_w"].to(dev, non_blocking=True)
            pseudo = batch["pseudo"].to(dev, non_blocking=True)
            conf = batch["conf"].to(dev, non_blocking=True)
            has_pseudo = bool(batch["has_pseudo"].any().item())

            with torch.amp.autocast("cuda", enabled=use_amp):
                out_w = model(weak)
                logits_w = out_w["logits"]
                prob_w = logits_w.softmax(1)

                l_point = point_ce(logits_w, point, weight,
                                   ignore=int(cfg["data"]["ignore_index"]))
                l_tri = tri_ce(logits_w, tri, tri_w,
                               ignore=int(cfg["data"]["ignore_index"]))

                l_ce = logits_w.sum() * 0.0
                l_dice = logits_w.sum() * 0.0
                if has_pseudo:
                    mask = (conf >= tau) & (pseudo != int(cfg["data"]["ignore_index"]))
                    if mask.any():
                        l_ce = masked_ce(logits_w, pseudo, mask, class_weight=cw,
                                         ignore=int(cfg["data"]["ignore_index"]))
                        l_dice = dice_loss(prob_w, pseudo, mask, num_classes,
                                           ignore=int(cfg["data"]["ignore_index"]))

                out_s = model(strong)
                l_consist = consistency_loss(prob_w, out_s["logits"].softmax(1))
                l_recon = reconstruction_loss(out_w["recon"], weak)
                l_vq = out_w["vq_loss"] + out_s["vq_loss"]

                loss = (float(tc["lambda_point"]) * l_point
                        + float(tc.get("lambda_tri", 2.0)) * l_tri
                        + float(tc["lambda_pseudo"]) * (l_ce + l_dice)
                        + float(tc["lambda_consist"]) * l_consist
                        + float(tc["lambda_recon"]) * l_recon
                        + l_vq)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            step += 1
            n_batch += 1
            agg["loss"] += float(loss.item())
            agg["point"] += float(l_point.item())
            agg["tri"] += float(l_tri.item())
            agg["pseudo_ce"] += float(l_ce.item()) if torch.is_tensor(l_ce) else 0.0
            agg["dice"] += float(l_dice.item()) if torch.is_tensor(l_dice) else 0.0
            agg["consist"] += float(l_consist.item())
            agg["recon"] += float(l_recon.item())
            agg["vq"] += float(l_vq.item())

            if max_steps > 0 and step >= max_steps:
                break

        avg = {k: v / max(n_batch, 1) for k, v in agg.items()}
        score, acc, miou, tri_acc, _ = validate(
            model, val_path, val_points, val_instances, val_pseudo, val_conf,
            num_classes, infer_size, dev, tau, use_amp,
            use_tiled=bool(cfg["infer"].get("use_tiled", True)),
            tile=int(cfg["infer"].get("tile", 512)),
            stride=int(cfg["infer"].get("stride", 384)))

        lr_now = opt.param_groups[0]["lr"]
        used = getattr(model.quant, "last_used", 0)
        sec = time.time() - t0
        print(f"[epoch {epoch}/{epochs}] loss={avg['loss']:.4f} point={avg['point']:.4f} "
              f"tri={avg['tri']:.4f} "
              f"pseudo={avg['pseudo_ce']:.4f} dice={avg['dice']:.4f} "
              f"consist={avg['consist']:.4f} recon={avg['recon']:.4f} vq={avg['vq']:.4f} "
              f"codes={used} val_score={score:.3f} (point={acc:.3f}, mIoU={miou:.3f}, "
              f"tri={tri_acc:.3f}) {sec:.1f}s")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, round(avg["loss"], 6), round(avg["point"], 6),
                round(avg["tri"], 6),
                round(avg["pseudo_ce"], 6), round(avg["dice"], 6),
                round(avg["consist"], 6), round(avg["recon"], 6), round(avg["vq"], 6),
                round(lr_now, 8), used, round(score, 4),
                round(float(acc), 4), round(float(miou), 4), round(float(tri_acc), 4),
                round(sec, 2),
            ])

        payload = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                   "val_score": score}
        torch.save(payload, os.path.join(ckpt_dir, f"{tag}last.pth"))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(payload, os.path.join(ckpt_dir, f"{tag}best.pth"))
            print(f"  ↳ 保存最佳权重 {tag}best.pth")

        if patience > 0 and (epoch - best_epoch) >= patience:
            print(f"  ↳ 早停：{patience} 轮验证分数未提升")
            break

        if max_steps > 0 and step >= max_steps:
            break

    print(f"训练完成，最佳 val_score={best_score:.3f}，日志 {log_path}")
    return best_score, os.path.join(ckpt_dir, f"{tag}best.pth")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dev = torch.device(args.device) if args.device else None
    run_training(
        cfg,
        pseudo_dir=args.pseudo_dir,
        init_ckpt=args.init,
        tag=args.tag,
        epochs=args.epochs or (1 if args.smoke else None),
        batch_size=args.batch_size or (2 if args.smoke else None),
        image_size=args.image_size or (256 if args.smoke else None),
        max_steps=2 if args.smoke else 0,
        device=dev,
    )


if __name__ == "__main__":
    main()
