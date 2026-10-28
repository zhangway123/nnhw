# 一键流程：校验标注 → 生成伪标签 → 训练 → 自训练 → 测试输出
$ErrorActionPreference = "Stop"

Write-Host "== 0. 校验 label.json =="
python src/weak_labels/validate_labels.py

Write-Host "== 1. 生成初始伪标签（SLIC + 图割） =="
python src/weak_labels/make_pseudo.py

Write-Host "== 2. 弱监督训练 VQ-U-Net =="
python src/train.py

Write-Host "== 3. 自训练（2~3 轮） =="
python src/selftrain.py --rounds 3

Write-Host "== 4. 仅对 test_photo 推理并输出分割图 =="
python src/test.py --ckpt checkpoints/r3_best.pth --data test_photo --out outputs/pred/test_photo

Write-Host "== 全部完成 =="
