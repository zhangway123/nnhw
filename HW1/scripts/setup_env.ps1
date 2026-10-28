# 环境安装与自检（Windows PowerShell）
$ErrorActionPreference = "Stop"

Write-Host "== 1. 安装依赖 =="
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "== 2. 自检 =="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import numpy, scipy, skimage, maxflow; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| skimage', skimage.__version__, '| maxflow ok')"

Write-Host "== 完成：若 cuda 为 False，请检查 NVIDIA 驱动与 PyTorch cu128 安装 =="
