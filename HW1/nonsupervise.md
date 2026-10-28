# 弱监督学习流程说明

> 文件名沿用 `nonsupervise.md`，内容已改为**点弱监督**流程：人工点标注（`outputs/label.json`）→ SLIC 超像素原型 + 图割伪标签 → VQ-U-Net 端到端训练 → 自训练 → 后处理与推理。
> 原「纯 VQ-VAE 重构预训练 + 码本聚类」路线已废弃，仅保留为消融 E2 的反例。

- 硬件 GPU：RTX 5060 Ti 16GB（Blackwell）
- 主要方法：点弱监督 + 超像素/图割伪标签 + VQ-U-Net（在线码本）+ 自训练
- 人工标注：`outputs/label.json`（9 张图、**19 个实例**：15 个单点 + 4 个三角形，共 27 个原始点；4 个类别 ID）
- 标签 ID：0 = 背景（由边界超像素生成，不人工标注）；1\~4 = 前景类别 ID，只作标签编号，不做物种命名

***

## 1. 人工标注 label.json

### 1.1 文件格式（2026-10-26 更新：支持“三角形实例”）

`outputs/label.json` 为数组，每项对应一张图。**`point` 数组的每个元素代表一个动物实例**，元素有两种写法：

1. **单点实例**：`{"x": ..., "y": ...}` —— 只知道该点是动物；
2. **三角形实例**：`[{x,y}, {x,y}, {x,y}]` —— 3 个点围成的三角形，**其内部像素几乎全部属于同一只动物**。三角形被动物包住，是保守的内点集（不是完整轮廓）。

真实示例（`5N0A0053_1.jpg`：3 个三角形实例 + 4 个单点实例 = 7 只动物）：

```json
{
  "image name": "5N0A0053_1.jpg",
  "label": "1",
  "point": [
    [{"x": 1255, "y": 2327}, {"x": 1248, "y": 2164}, {"x": 1219, "y": 2467}],
    [{"x": 662,  "y": 2839}, {"x": 587,  "y": 2600}, {"x": 613,  "y": 2954}],
    [{"x": 1602, "y": 3270}, {"x": 1542, "y": 3131}, {"x": 1566, "y": 3419}],
    {"x": 2284, "y": 2959},
    {"x": 3785, "y": 2383},
    {"x": 2061, "y": 2056},
    {"x": 2515, "y": 2222}
  ]
}
```

| 字段 | 含义 | 使用注意 |
| --- | --- | --- |
| `image name` | 图像文件名（不含目录） | 与 `train_eval_photos/` 下文件名对齐，扩展名可能不同，按 stem 匹配 |
| `label` | 图像级类别 ID，字符串 `"1"`~`"4"` | 必须 `int()`；**同一张图的所有实例同属该类** |
| `point` | 实例列表，元素为 dict（单点）或 list（3 点三角形） | **必须同时兼容两种类型**，不能假设全是 dict |
| 坐标 | `x/y` 均为**原图分辨率** | 裁剪 / 翻转 / 缩放时，三角形 3 个顶点与单点用完全相同的变换 |

解析规则：

- 一个元素 = 一个实例，不要假设「一张图只有一个实例」；
- 三角形固定 3 个顶点，**顶点顺序无关**，内部判定用射线法或直接填充凸包；
- 退化三角形（三点近似共线、面积过小）视为无效实例并跳过，同时记录警告；
- 同一实例的 3 个点类别相同（继承图级 `label`），不存在“一个三角形跨两类”。

### 1.2 标注现状统计

| 图像 | label | 实例数 | 单点 | 三角形 | 原始点数 |
| --- | --- | --- | --- | --- | --- |
| 5N0A0053_1.jpg | 1 | 7 | 4 | 3 | 13 |
| 5N0A0065_1.jpg | 1 | 4 | 4 | 0 | 4 |
| 5N0A1577_1.jpg | 1 | 1 | 1 | 0 | 1 |
| 5N0A0827_1.jpg | 2 | 1 | 0 | 1 | 3 |
| 5N0A0840_1.jpg | 2 | 1 | 1 | 0 | 1 |
| 5N0A0856_1.jpg | 3 | 1 | 1 | 0 | 1 |
| 5N0A0857_1.jpg | 3 | 1 | 1 | 0 | 1 |
| 5N0A6430_1.jpg | 4 | 1 | 1 | 0 | 1 |
| 5N0A6425_1.jpg | 4 | 2 | 2 | 0 | 2 |

合计：**9 张图 / 19 个实例（15 单点 + 4 三角形）/ 27 个原始点 / 4 类**。

分类别看：类别 1 有 3 图共 12 个实例（含 3 个三角形）；类别 2 有 2 图共 2 个实例（含 1 个三角形）；类别 3、4 各 2 图 2~3 个实例，均为单点。

### 1.3 标注语义与可信度优先级

| 优先级 | 标注形式 | 可作为什么监督 |
| --- | --- | --- |
| 1（最高） | **三角形内部像素** | 区域级正样本：伪标签 CE/Dice、图割硬约束、自训练强制覆盖 |
| 2 | 单点周围 5×5 窗口 | 点级正样本：点 CE |
| 3 | 图割伪标签 | 弱样本：高置信像素参与 CE/Dice |
| 4 | 模型预测（自训练） | 需置信度过滤 |

要点：

- 三角形是**内点集**，不是完整掩码。动物轮廓（尤其边界）仍需图割与模型推断，不能在评估里把三角形当 GT；
- 三角形面积差异很大（0827 的三角形几乎横跨整幅图），因此按面积归一化权重，避免大三角形压过单点实例；
- 单点与三角形并存时，**三角形区域的监督强度更高**，梯度上应体现出来（见 4.2）。

### 1.4 数据划分

| 用途 | 内容 | 说明 |
| --- | --- | --- |
| 训练集 | `train_eval_photos/` 全部图像 | 其中 9 张有标注；8 张参与训练（18 个实例：4 三角形 + 14 单点） |
| 验证 | `5N0A1577_1.jpg`（1 个单点实例） | 从训练集中留出，仅用于早停与调参，不参与训练 |
| 无标签部分 | `train_eval_photos/` 其余图像（约 150 张） | 无标注，仅在自训练阶段以伪标签参与训练 |
| 测试集 | `test_photo/`（6 张） | 只推理并输出分割图，不参与训练与调参 |

留出 1577 而不是含 7 个实例的 0053：保住训练集里类别 1 的多实例样本，同时保证 4 个类别在训练集中都出现。

`test_photo` 没有真值，只产出分割结果图（见 6.3）；需要像素级指标时按第 7 节另建评估子集。

***

## 2. 数据预处理

沿用 `data_preprocess.py` 产出的 `outputs/dataset.json`（number / image name / length & width / label / image path）：训练目录固定为 `train_eval_photos/`，测试目录固定为 `test_photo/`，划分以 1.3 为准，不再使用 `data_preprocess.py` 的随机 8:2 划分。

| 阶段       | 尺寸策略                                        |
| -------- | ------------------------------------------- |
| 伪标签（SLIC/图割） | 原图分辨率，不缩放（保留边界细节）                          |
| 训练       | 随机裁剪 512×512（E5 可试 640），不足则先 resize 再裁      |
| 推理       | letterbox 到 1024，长边对齐，短边居中 padding          |

归一化：像素到 `[0,1]` 后按 `mean=0.5, std=0.5` 标准化（无预训练权重，不强制 ImageNet 统计量，但全流程统一）。

**点的坐标变换**（最容易出错的地方）：

```
随机裁剪：(x, y) → (x - x0, y - y0)，越界点丢弃
缩放：    (x, y) → (x * s, y * s)
letterbox：(x, y) → (x * s + pad_x, y * s + pad_y)
```

所有几何增强（翻转、裁剪、旋转、仿射）必须对点做同样变换，而颜色类增强不动点坐标。

增强分两路：

- **弱增强**（供教师/一致性目标）：水平翻转 + 随机裁剪；
- **强增强**（供学生）：上述 + 颜色抖动 + 高斯模糊 + 亮度对比度扰动 + 少量噪声。

***

## 3. 初始伪标签：SLIC + 原型 + 图割

产物目录：`outputs/pseudo/{stem}_label.png`（uint8，0\~4）、`outputs/pseudo/{stem}_conf.npy`（float32 置信度图）。

### 3.1 SLIC 超像素

- 在 Lab 空间执行 `skimage.segmentation.slic`；
- `n_segments = clip(round(H*W/1200), 1500, 6000)`，`compactness = 15`，`sigma = 1`；
- 保留原图分辨率，不做预缩放。

### 3.2 超像素特征

| 特征块 | 内容                                                 | 权重  |
| --- | -------------------------------------------------- | --- |
| 颜色  | Lab 三通道的均值与标准差                                      | 1.0 |
| 纹理  | 灰度 LBP 直方图（uniform，P=8，R=1，10 bin）                   | 0.5 |
| 边缘  | 超像素内 Sobel 梯度均值 + 超像素边界上的平均梯度                       | 0.5 |
| 位置  | 归一化质心 `(x/W, y/H)`                                  | 0.1 |

各特征块先各自 z-score，再按权重拼接。位置权重必须低，否则图割会退化成「按位置分区」。

### 3.3 实例种子与类别原型（硬约束）

`label.json` 每图只有一个 `label`，所以该图所有实例同属类别 $c$。先把每个实例转成**种子超像素**：

| 实例类型 | 种子超像素定义 |
| --- | --- |
| 单点 | 包含该点的超像素（1 个） |
| 三角形 | 三角形内部覆盖率 $\rho_i \ge 0.5$ 的超像素，$\rho_i$ = 该超像素落在三角形内的像素数 / 超像素总像素数 |

- 类别原型 $p_c$ = 该类**所有实例种子超像素**特征的均值（跨图合并，三角形按面积加权、单点按 1 计）；
- 硬约束分两档，避免三角形顶点附近超像素溢出到背景：
  - 单点所在超像素：$D_i(c)=0$，$D_i(c'\neq c)=+\infty$；
  - 三角形内 $\rho_i \ge 0.9$ 的超像素：同样硬约束为 $c$；
  - $0.5 \le \rho_i < 0.9$ 的超像素：**只加数据项**，不加硬约束；
- 这是全流程唯一的人工真值，任何后处理都不得覆盖。

### 3.4 背景原型（边界超像素）

- 取图像四周边界带（外侧约 3% 宽度）内、**不含任何实例种子**（单点或三角形）的超像素；
- 其特征均值作为背景原型 $p_{bg}$；
- 对不含种子的边界带超像素额外加背景硬约束，抑制图割把背景翻成动物。

### 3.5 多类图割

在超像素邻接图上最小化：

$$
E(x)=\sum_i D_i(x_i)+\lambda\sum_{i\sim j} w_{ij}\,\mathbb{1}[x_i\neq x_j]
$$

- 数据项 $D_i(c)=\lVert f_i-p_c\rVert^2/(2\sigma^2)$，$\sigma$ 取特征距离的中位数；
- 硬约束来源：单点超像素、三角形 $\rho\ge0.9$ 的超像素、边界带背景超像素；
- 平滑项 $w_{ij}=\exp\big(-\lVert \bar c_i-\bar c_j\rVert^2/(2\sigma_c^2)\big)\cdot(1+e_{ij})$，$\bar c$ 为超像素平均 Lab 颜色，$e_{ij}$ 为共享边界平均梯度；
- $\lambda$ 默认 1.0（0.5\~2.0 搜索）；
- 用 `PyMaxflow` 的 alpha-expansion 求解多标签近似最优；
- 超像素标签回填到像素，得到初始伪标签。

### 3.6 置信度

- 每像素置信度 = 该像素所属超像素在图割标签上的 $\max_c \mathrm{softmax}(-D_i)(c)$；
- 硬约束超像素（单点、三角形 $\rho\ge0.9$、边界带）置信度置 1；
- 三角形内部（$\rho\ge0.5$）置信度同样置 1；
- 低置信像素在训练中不参与伪标签损失。

### 3.7 必做的质量检查

1. 叠加可视化：原图 + 伪标签色块 + 单点位置 + **三角形轮廓**，逐图检查；
2. 点一致性：伪标签在单点处、以及三角形三个顶点处的类别必须等于实例类别（应为 100%）；
3. **三角形内部一致率**：三角形内部被判成其它类的像素比例应 < 5%；
4. 类别面积占比：若某一类占图面积 > 80%，基本可判定原型污染，需调 $\lambda$ 或检查背景原型。

***

## 4. VQ-U-Net 模型与训练

### 4.1 结构

```
x [B,3,512,512]
 → Encoder：4 stage，通道 32/64/128/256，每 stage 2×[Conv3×3 + GroupNorm + SiLU]，stage 间 stride 2
 → Conv1×1 降到 D 维（默认 32）
 → VQ 瓶颈：最近邻查表（K 个码），直通估计，EMA 更新，死码重启
 → Decoder：4 stage，上采样 + 与 encoder 跳连 concat + Conv3×3 + GroupNorm + SiLU
 → Conv1×1 分类头 → 5 类 logits（0 背景 + 1~4），上采样到输入尺寸
 → 可选重构分支：解码器末级特征 → Conv1×1 → 3 通道 → Sigmoid（仅正则）
```

| 组件    | 设置                                                    |
| ----- | ----------------------------------------------------- |
| 归一化   | GroupNorm（batch 小，BN 不稳定）                             |
| 激活    | SiLU                                                  |
| VQ 位置 | ×16 瓶颈（默认）；E5 对比 ×8                                    |
| 码本    | K = 128（默认），D = 32；E5 对比 64 / 256                       |
| 码本更新  | EMA decay 0.99；每 100 step 死码重启（用当前 batch 的 $z_e$ 重置未用码） |
| 承诺损失  | $\beta = 0.25$，作用在编码器一侧                               |
| 初始化   | **完全随机**，不加载任何预训练权重                                   |
| 参数量   | 轻量级（通道 32\~256），512 输入下 16GB 显存充裕                     |

**VQ 的角色被严格限定**：特征离散化 + 瓶颈正则。它不生成伪标签，也不提供类别原型，因此不会重演「码本被背景占据 → 聚类退化为背景分组」的老问题。

### 4.2 损失

$$
\mathcal{L}=\lambda_s\left(\mathcal{L}_{tri}+\mathcal{L}_{point}\right)+\lambda_p\mathcal{L}_{pseudo}+\lambda_c\mathcal{L}_{consist}+\lambda_r\mathcal{L}_{recon}
$$

| 损失项 | 定义 | 默认权重 |
| --- | --- | --- |
| $\mathcal{L}_{tri}$ | **三角形实例**：内部像素做 CE（可选加 Dice），每个实例先按自身面积归一，再按实例数平均 | 2.0 |
| $\mathcal{L}_{point}$ | **单点实例**：以点为中心的 5×5 窗口做 CE，按窗口像素数归一，再按实例数平均 | 2.0 |
| $\mathcal{L}_{pseudo}$ | 仅在置信度 $\ge\tau$ 的像素上做 CE（低置信 `ignore_index`）+ 逐类 Dice 平均 | 1.0 |
| $\mathcal{L}_{consist}$ | 弱/强两路预测一致性（MSE），目标侧 stop-grad | 0.2 |
| $\mathcal{L}_{recon}$ | 解码器末级特征重建输入（MSE），**仅作正则** | 0.03 |

要点：

- 三角形内部是**区域级**真值，可靠性高于单点窗口，因此单独成项、权重更高；
- 三角形按面积归一，避免 0827 那种「横跨整幅图」的大三角形压过其它实例；
- 单点窗口保持 5×5\~11×11，窗口过大会把点当区块真值，污染邻近背景；
- 若一个 batch 内没有三角形（多数图只有单点），$\mathcal{L}_{tri}$ 记 0，不能报错；
- 伪标签 CE 按类别频率的平方根反比加权（`1/sqrt(freq)`，裁剪 `[0.2, 5]`）；
- Dice 按类平均，缓解极端不平衡；
- 一旦出现「重建指标变好、分割变差」，说明 $\lambda_r$ 过大，降到 0.01 或关闭。

### 4.3 训练配置

| 项 | 取值 |
| --- | --- |
| 优化器 | AdamW，lr = 1e-3，weight decay = 1e-4 |
| 调度器 | Cosine Annealing + 前 5% warmup |
| Batch | 8\~16（512 输入） |
| 采样 | `samples_per_epoch=128`，不足则重复采样（8 张图 → ×16） |
| 裁剪 | 实例中心裁剪概率 0.4，抖动 ±size/2（防止「中心即前景」捷径） |
| Epochs | 30\~80，早停（验证 mIoU 连续 N 轮不升） |
| 精度 | AMP（fp16） |
| 随机种子 | 42 |

早期如果验证 mIoU 上不去，先查四件事：三角形是否正确解析、点/三角形坐标是否随增强变换、伪标签是否正确、训练与推理尺度是否一致。

***

## 5. 自训练

| 项目 | 做法 |
| --- | --- |
| 轮次 | 2\~3 轮 |
| 教师 | 当前模型的 EMA 副本（decay 0.99） |
| 数据 | `train_eval_photos/` 全部图像：8 张有标注图 + 其余约 150 张无标签图 |
| 伪标签 | 教师预测 argmax；保留置信度 $\ge\tau$（0.9，或逐类 top 60%）的像素 |
| 实例优先 | **单点 5×5 窗口 + 三角形内部像素**强制覆盖为实例类别，不参与置信度过滤，作为每轮锚点 |
| 训练方式 | 用上一轮权重初始化，继续用 4.2 的损失训练；实例监督项全程保留 |
| 防漂移 | 每轮抽 3\~5 张预测图人工核验；若某类 IoU / 三角形一致率连续下降则回退上一轮权重 |

每一轮记录：伪标签覆盖率、各类像素占比、验证 mIoU、三角形内部一致率、伪标签与上一轮的一致率（过高说明没学到新东西，过低说明在震荡）。

***

## 6. 后处理与推理

### 6.1 后处理

1. **SLIC 投票**（主）：沿用第 3 节的超像素，对每个超像素取预测概率均值后 argmax，得到边界贴合的分割；
2. **CRF**（可选）：`pydensecrf` 5 次迭代（颜色 + 位置高斯核）；Windows 编译失败则跳过，不影响主线；
3. **去小连通域**：删除面积小于阈值的孤立连通域——**大图必须用小阈值**（默认 `min_ratio=0.0005`；用 0.5% 会把整只动物删掉）；
4. **保留实例所在连通域**：对每个单点和每个三角形的质心，在同类掩码中保留其所在连通域；只删「无实例覆盖且面积小」的同类孤立块；
5. **实例一致性回填**：最终掩码在单点处或三角形内部若类别不符，强制回填实例类别（三角形内部整块回填）。

### 6.2 推理（仅 test_photo）

1. 只遍历 `test_photo/` 下的 6 张图；`train_eval_photos/` 不参与推理（可另行抽样看效果，但不作为产出）；
2. letterbox：`s = 1024 / max(H, W)`，长边缩放到 1024，短边居中 padding 到 1024×1024，记录 `s` 与 `pad`；
3. 前向得 `[1, 5, 1024, 1024]` logits，argmax 得掩码；
4. 裁掉 padding 回到 `round(W·s) × round(H·s)`；
5. 最近邻上采样回原图 `(H, W)`；
6. 执行 6.1 后处理，按 6.3 保存。

**注意**：letterbox 的 `pad` 必须按「居中」还是「左上」二选一并全程统一，否则还原会整体偏移。

### 6.3 输出规范（仅 test_photo）

产物目录：`outputs/pred/test_photo/`，文件名与输入保持一致（`{stem}` = 去掉扩展名的原图名）。

| 文件                    | 内容                                                     |
| --------------------- | ------------------------------------------------------ |
| `{stem}_mask.png`     | 单通道 uint8 掩码，像素值 = 类别 ID（0\~4），尺寸与原图一致                   |
| `{stem}_color.png`    | 彩色分割图，每类固定颜色（0 = 黑）                                    |
| `{stem}_overlay.png`  | 原图与彩色掩码按 0.5 透明度叠加，用于报告附录                              |
| `{stem}.json`         | 元信息：原图尺寸、letterbox 的 `s` 与 `pad`、各类像素占比、推理耗时             |

调色板固定写在 `src/postprocess.py` 的 `PALETTE` 中，不随运行变化，保证附录图配色一致。

***

## 7. 评估

| 指标                      | 说明                        |
| ----------------------- | ------------------------- |
| mIoU                    | 每类 IoU 均值，核心指标            |
| mAcc（类别平均准确率，即 mPA）      | 各类 PA 均值                  |
| Pixel Accuracy (PA)     | 全局像素精度                    |
| Dice / F1               | 逐类计算后取均值                  |
| Boundary IoU            | 边界带内 IoU，检验边界质量           |
| 每类 IoU                  | 逐类列出，定位被吞并/混淆的类别          |
| 点准确率                    | 预测在点处与标注一致的比例（弱监督专属锚点指标）  |

`test_photo` 没有真值，只产出分割结果，不用于指标计算。像素级指标在 `train_eval_photos/` 内另建评估子集：

1. **推荐**：从 `train_eval_photos/` 中另标 10\~20 张像素掩码（不参与训练），在该子集上算 mIoU 等 5 个以上指标；
2. 暂无参考时，报告验证图（`5N0A1577_1.jpg`）的点准确率、伪标签一致率与定性结果，并明确说明 mIoU 待参考补齐。

指标一律按类别 ID（0\~4）报告，不做物种命名。同时报告初始伪标签的质量（与评估子集的 IoU），用于区分「伪标签错」和「模型欠拟合」。

***

## 8. 代码组织与运行

```
HW1/
├── outputs/label.json              # 人工点标注（唯一标注来源）
├── outputs/dataset.json            # data_preprocess.py 产出
├── outputs/pseudo/                 # 伪标签 + 置信度
├── outputs/pred/test_photo/        # 测试集分割结果（mask / color / overlay / json）
├── configs/
│   └── config.yaml                 # 超参、路径
├── src/
│   ├── weak_labels/
│   │   ├── validate_labels.py      # 校验 label.json（单点/三角形混排、越界、退化三角形、缺图）
│   │   ├── slic_features.py        # SLIC + 颜色/LBP/边缘/位置特征
│   │   ├── prototypes.py           # 实例种子（单点 + 三角形）与类别/背景原型
│   │   └── make_pseudo.py          # 图割 + 置信度 + 可视化（画三角形轮廓）
│   ├── data/dataset.py             # 图像 + 实例（点/三角形）+ 伪标签联合加载
│   ├── models/vq.py, vq_unet.py    # 码本（EMA/死码重启）与网络
│   ├── losses.py                   # 点窗口 CE + 三角形内部 CE/Dice + 一致性 + 重构
│   ├── train.py
│   ├── selftrain.py
│   ├── postprocess.py
│   ├── metrics.py
│   └── test.py
└── checkpoints/, results/
```

```
python src/weak_labels/validate_labels.py --labels outputs/label.json
python src/weak_labels/make_pseudo.py --labels outputs/label.json --out outputs/pseudo
python src/train.py --config configs/config.yaml --data train_eval_photos
python src/selftrain.py --init checkpoints/best.pth --rounds 3
python src/test.py --ckpt checkpoints/best.pth --data test_photo --out outputs/pred/test_photo
```

`test.py` 只处理 `test_photo/`，产出 6 张图的 mask / color / overlay / json，不计算指标；指标评估需按第 7 节另建评估子集。

旧路线的 `train_vqvae.py`、`kmeans_segment.py` 不再参与主线，仅作 E2（纯 VQ 重构预训练）反例实验时复用。

***

## 9. 关键决策与已知坑

**为什么废弃纯 VQ 重构预训练**（作为报告中的实验分析）：

| 证据      | 观测值                              | 含义                       |
| ------- | -------------------------------- | ------------------------ |
| 码本利用率   | 0.586%（6/1024）                   | 1024 个码只有 6 个在工作，语义容量被锁死 |
| 重建 PSNR | 11.74 dB                         | 重建收益主要来自背景平坦区域           |
| 训练曲线    | eval loss 自第 2 epoch 起上升，11 轮早停   | 样本少，模型在记忆背景              |
| 目标尺度    | 6354×4236 → 512 → ×16 → 32×32     | 动物主体仅占 1\~2 个潜变量格         |

结论：纯 VQ-VAE 重构预训练因背景像素主导、小目标令牌不足，初始码本偏背景，故不作为主路线；改为超像素伪标签初始化 + 在线 VQ 瓶颈。

**已知坑**：

1. **类 2/3/4 各只有 1\~2 张图、1\~3 个点**，单点原型极不稳定 → 图割前检查原型距离分布，必要时用该图多个同色块补充，或用 E4（增加外观点）验证；
2. **点落在边界超像素上** → 会同时触发「动物硬约束」和「背景硬约束」，必须以点优先，背景约束跳过含点超像素；
3. **图割把整片雪地判成动物** → 检查背景原型是否被污染（边界带含点）、$\lambda$ 是否过小、位置特征权重是否过高；
4. **增强后点越界** → 随机裁剪时丢弃越界点，但若一张图所有点都被丢，该样本退化为只有伪标签监督，需记录被丢弃的点数；
5. **自训练确认偏误** → 点窗口始终强制覆盖，置信度阈值不要低于 0.85，每轮必须看可视化；
6. **CRF 装不上** → 直接用 SLIC 投票，边界质量损失有限，且不影响跑通。

***

## 10. 代码实现与首次实验记录

> 本节记录按本文档落地后的实际代码、代码检查中修掉的问题，以及首次在 RTX 5060 Ti 上的实验结果（2026-10-26）。

### 10.1 代码落地情况

| 模块 | 文件 | 状态 |
| --- | --- | --- |
| 配置 | `configs/config.yaml` | 已完成（数据/伪标签/模型/训练/自训练/推理六段） |
| 标注校验 | `src/weak_labels/validate_labels.py` | 已完成，输出 `outputs/label_stats.json` |
| SLIC 特征 | `src/weak_labels/slic_features.py` | 已完成（Lab + LBP + Sobel + 位置，20 维） |
| 原型 | `src/weak_labels/prototypes.py` | 已完成（点硬约束 + 边界背景原型） |
| 图割 | `src/weak_labels/graphcut.py` | 已完成（alpha-expansion，PyMaxflow 缺失时退化为 ICM） |
| 伪标签 | `src/weak_labels/make_pseudo.py` | 已完成，输出 `{stem}_label.png` / `{stem}_conf.png` / `vis/` |
| 数据加载 | `src/data/dataset.py` | 已完成（点坐标同步变换、弱/强两路、点中心裁剪） |
| 码本 | `src/models/vq.py` | 已完成（EMA 在线更新 + 死码重启 + 直通估计） |
| 网络 | `src/models/vq_unet.py` | 已完成（GroupNorm + VQ 瓶颈 + 跳连解码 + 1×1 头） |
| 损失 | `src/losses.py` | 已完成（点 CE / 伪标签 CE+Dice / 一致性 / 重构） |
| 指标 | `src/metrics.py` | 已完成（mIoU / mAcc / PA / Dice / Boundary IoU / 每类） |
| 后处理 | `src/postprocess.py` | 已完成（调色板、去小连通域、点连通域保留、SLIC 投票、letterbox） |
| 推理 | `src/inference.py` | 已完成（letterbox + 滑窗 tiled 两种） |
| 训练 | `src/train.py` | 已完成（早停、AMP、CSV 日志、最佳权重） |
| 自训练 | `src/selftrain.py` | 已完成，**尚未正式运行** |
| 测试 | `src/test.py` | 已完成（仅 test_photo，输出 4 类文件） |
| 冒烟测试 | `scripts/smoke_test.py` | 已完成（合成图跑通伪标签→模型→损失→数据→输出） |

实际运行命令：

```
python src/weak_labels/validate_labels.py
python src/weak_labels/make_pseudo.py
python src/train.py --epochs 30 --tag final_
python src/test.py --ckpt checkpoints/best.pth
```

### 10.2 代码检查：发现并修复的 7 个问题

这些问题都会让实验在无报错的情况下失效，属于必查项：

| # | 问题 | 现象 | 修复 |
| --- | --- | --- | --- |
| 1 | VQ 的 EMA 分母用了接近 0 的 `cluster_size` | vq loss 冲到 **7.7e6**，训练崩溃 | 分母改 `cluster_size.clamp(min=1.0)`；死码判定改 `<0.5` |
| 2 | AMP 下死码重启 dtype 不匹配 | 第 20 步 `Index put requires...` 报错 | 重启前 `zf[pick].to(self.emb.dtype)` |
| 3 | 类别权重 `1/freq` 且背景被裁到 0.1 | 模型退化成**全前景** | 改 `1/sqrt(freq)`，裁剪 `[0.2, 5]` |
| 4 | 伪标签置信度用 softmax，普遍 <0.9 | 伪标签 CE 的 mask 为空，`pseudo=0, dice=0` | 初始伪标签 conf 恒为 1，自训练才用 softmax 阈值 |
| 5 | 训练（原图裁剪）与推理（整图缩小）尺度不一致 | 验证集预测几乎全背景 | 训练前长边统一 1024，与 letterbox 推理一致 |
| 6 | 点中心裁剪抖动仅 ±size/4 | 目标总在裁剪框中心，模型学成「中心即前景」的捷径 | 抖动改 ±size/2，点在框内均匀分布 |
| 7 | 去小连通域阈值 0.5% | 大图 0.5% ≈ 15 万像素，把动物全部删掉 → 输出全黑 | 默认改 0.0005，并开放 `--min-ratio` |

### 10.3 首次实验记录

**环境**

```
NVIDIA GeForce RTX 5060 Ti, driver 610.88, 16311 MiB
torch 2.11.0+cu128 | cuda available True
```

**阶段 1：伪标签（原图分辨率）**

| 项目 | 结果 |
| --- | --- |
| 标注 | 9 张图 / 19 点 / 4 类 |
| 超像素 | 5704 ~ 6131 / 图（`n_segments ≈ H*W/1200`） |
| 求解 | alpha-expansion 全部收敛 |
| 点一致性 | **9/9 = 100%**（硬约束生效） |
| 前景占比 | 0.4% ~ 12.9%（多数 1%~3%） |
| 耗时 | 13 ~ 36 s / 图（SLIC 占主要） |

**阶段 2：VQ-U-Net 训练**

| 项目 | 结果 |
| --- | --- |
| 训练样本 | 8 张点标注图 × 16 重复 = 128 / 轮 |
| 配置 | 512 输入、batch 8、AdamW lr 1e-3、AMP、K=64、VQ ×16 |
| 轮数 | 30 轮，早停于第 29 轮（best 第 14 轮） |
| loss | 2.58 → 0.36 |
| 点损失 | 0.59 → 0.001 |
| 码本使用 | 18 → 64 个码 |
| 单轮耗时 | 35 ~ 50 s |

**阶段 3：测试集推理（仅 test_photo）**

| 图像 | 前景占比 | 耗时 |
| --- | --- | --- |
| 5N0A5174_1 | 0.0% | 8.6 s |
| 5N0A5204_1 | 0.07%（类 2） | 7.3 s |
| 5N0A5965_1 | 0.97%（类 4） | 6.8 s |
| 5N0A7706_1 | 0.57%（类 1+2） | 8.2 s |
| 5N0A7727_1 | 2.58%（类 4） | 4.7 s |
| 5N0A9908_1 | 0.07%（类 1） | 6.1 s |

产物：`outputs/pred/test_photo/{stem}_mask.png / _color.png / _overlay.png / .json`。

### 10.4 当前结果与结论

| 指标 | 数值 | 参照 |
| --- | --- | --- |
| 验证图 mIoU（vs 图割伪标签） | **0.258** | 全背景基线 0.19 |
| 验证图前景 IoU | ≈ 0.07 | — |
| 验证点准确率 | 0 | — |
| 训练集伪标签 mIoU | 仍偏低（未拟合） | — |

结论：**流程正确、可复现，但模型仍处于欠拟合 + 背景偏置状态**，还达不到可用的语义分割。原因按影响排序：

1. 监督极稀疏：8 张图、18 个点，且编码器随机初始化；
2. 目标极小：原图 6000×4000 缩到 1024 后动物仅剩几十像素，伪标签前景只占 1%；
3. 自训练尚未运行——按 5.6 节，这一步才是把监督从 9 张扩散到 166 张的关键；
4. 尺度一致性敏感：训练裁剪尺度与推理尺度必须对齐，否则模型直接退化为全背景（本轮实测已确认）。

### 10.5 下一步（按性价比排序）

1. **运行自训练 1~2 轮**：`python src/selftrain.py --rounds 2 --epochs 20`，把伪标签扩散到 `train_eval_photos/` 全部图像；
2. **补点**：类 2/3/4 各只有 1~2 个点，建议每类补到 3~5 个点（对应消融 E4）；
3. **加 E0 对照**：普通小 U-Net（无 VQ）+ 同一伪标签，验证 VQ 瓶颈是否值得；
4. **提高目标像素占比**：训练/推理统一在更大分辨率（如 1536 / 2048）或按目标尺度滑窗，避免动物只剩几十像素；
5. **评估参考**：补 10~20 张像素级标注，才能算正式 mIoU / mAcc / Dice / Boundary IoU。

> **注意**：`outputs/pseudo/` 与 `checkpoints/best.pth` 都是按**旧格式**（把三角形的 3 个点当成 3 个独立动物）生成的。改成 1.1/3.3 的新语义后，必须重新生成伪标签并重训，旧权重不能继续用。

***

## 11. 实现步骤（供 agent 执行）

> 目标：让代码支持 1.1 的“单点 / 三角形混排”格式，并把三角形内部作为区域级监督接入伪标签、损失、自训练与后处理。
> 约束：**不得把三角形当成完整掩码**；**不得假设 `point` 元素全是 dict**；改动要保持旧行为可复现（单点图逻辑不变）。

### 11.1 新增实例解析模块 `src/data/instances.py`

```python
@dataclass
class Instance:
    cls: int                    # 图级 label
    kind: str                   # "point" | "triangle"
    points: list[tuple[float, float]]   # 1 个点 或 3 个顶点（原图坐标）
    area: float                 # 单点为 0；三角形为 |cross|/2
```

必须实现（函数名固定，供其他模块调用）：

| 函数 | 签名 | 行为 |
| --- | --- | --- |
| `parse_instances` | `(entry: dict, img_size: tuple[int,int]) -> list[Instance]` | 遍历 `entry["point"]`；`dict` → 单点实例；`list` 且长度==3 → 三角形实例；其它长度 → 跳过并 warn；顶点越界 → clip 并 warn；退化三角形（面积 < 100 px² 或三点共线）→ 跳过并 warn |
| `load_instances` | `(labels_path: str) -> tuple[list[dict], dict[str, list[Instance]]]` | 返回原始条目 + `{stem: [Instance]}`；`stem` 由 `image name` 去扩展名得到 |
| `triangle_mask` | `(points, shape) -> np.ndarray[bool]` | 用 `PIL.ImageDraw.polygon` 在 `shape` 网格上填充，返回布尔掩码（顶点可为浮点，四舍五入） |
| `instances_to_crop` | `(instances, x0, y0, size, flip=False) -> list[Instance]` | 坐标减 `(x0,y0)`，`flip` 时 `x = size-1-x`；单点出框 → 丢弃；三角形与裁剪框交集面积/自身面积 < 0.5 → 丢弃，否则保留（顶点 clip 到框内） |
| `instance_centers` | `(instances) -> list[tuple[float,float,int]]` | 单点返回自身，三角形返回质心，供“实例中心裁剪”使用 |

### 11.2 修改 `src/weak_labels/slic_features.py`

新增一个函数（其余不动）：

```python
def superpixel_triangle_overlap(seg, tri_mask):
    """返回每个超像素被三角形覆盖的比例 rho [S]。"""
    idx = seg.reshape(-1)
    hit = np.bincount(idx, weights=tri_mask.reshape(-1).astype(np.float64),
                      minlength=S)
    tot = np.bincount(idx, minlength=S)
    return (hit / np.maximum(tot, 1)).astype(np.float32)
```

### 11.3 修改 `src/weak_labels/prototypes.py`

- 用 `instances_to_crop` / `parse_instances` 替换 `seed_superpixels`（保留旧函数作为单点包装，避免破坏调用方）；
- 新增 `seed_superpixels_from_instances(seg, instances, rho_hard=0.9, rho_soft=0.5)`：
  - 返回 `seeds: list[(sup_id, cls)]`（$\rho \ge$ `rho_soft` 的超像素，含全部单点超像素）与 `hard: dict[sup_id, cls]`（单点超像素 + $\rho \ge$ `rho_hard` 的超像素）；
- `class_prototypes` 的均值改为**加权均值**：单点权重 1.0，三角形权重 `min(area / 1e4, 50.0)`（防止超大三角形独占原型）；
- `border_superpixels` 的 `exclude_sups` 需包含三角形覆盖 $\rho \ge 0.1$ 的超像素（背景不能从三角形里取）。

### 11.4 修改 `src/weak_labels/make_pseudo.py`

1. 读取用 `load_instances`，第一遍缓存里保存 `instances` 与每张图的 `tri_masks`（三角形布尔图，可只存 bbox 内的 mask 以省内存）；
2. 硬约束改用 3.3 的两档规则；
3. 求得 `sup_labels` 后，在**像素级**追加一次强制写入：
   ```python
   for inst in instances:
       if inst.kind == "triangle":
           mask[triangle_mask(inst.points, mask.shape)] = inst.cls
           conf[...] = 1.0
   ```
4. `vis/` 叠加图用 `ImageDraw.polygon` 画出三角形边框（颜色与类别一致）；
5. `summary.json` 每张图新增 `tri_consistency`：三角形内部与该类一致的像素占比（应 ≥ 0.95）；
6. 输出日志打印“实例数 / 单点数 / 三角形数”，便于核对 1.2 的统计。

### 11.5 修改 `src/data/dataset.py`

- `load_points` → `load_instances`（保留 `load_points` 作为兼容包装）；
- `__getitem__` 额外返回：

| 键 | 形状 | 含义 |
| --- | --- | --- |
| `tri` | `[size,size]` int64 | 三角形内部像素 = 实例类别，其余 = `ignore_index` |
| `tri_w` | `[size,size]` float32 | 三角形内部为 1，其余 0 |
| `n_tri` | 标量 | 本样本内有效三角形数 |

- 绘制顺序：先 `tri`（区域），再单点 5×5 窗口覆盖（点优先，二者重叠时以点为准）；
- 裁剪中心：用 `instance_centers` 随机选一个实例中心（三角形用质心），抖动仍为 ±size/2；
- 变换：`instances_to_crop(..., flip=flip)`，翻转后三角形顶点顺序无需保持。

### 11.6 修改 `src/losses.py` 与 `src/train.py`

- `losses.py` 新增：
  ```python
  def tri_ce(logits, target, weight, ignore=255):
      ce = F.cross_entropy(logits, target, ignore_index=ignore, reduction="none")
      return (ce * weight).sum() / (weight.sum() + 1e-6)
  ```
- `train.py`：
  - 取 `tri, tri_w = batch["tri"], batch["tri_w"]`；
  - `l_tri = tri_ce(logits_w, tri, tri_w)`；当 `tri_w.sum()==0` 时该项自动为 0（不能报错）；
  - 总损失改为 `λ_tri*l_tri + λ_point*l_point + ...`，`λ_tri` 读 `config.train.lambda_tri`（默认 2.0）；
  - CSV 增加 `tri` 列；验证函数增加 `val_tri_consistency`（三角形内部一致率）。

### 11.7 修改 `src/selftrain.py`

`generate_pseudo` 的“实例优先”段落改为：

```python
for inst in instances:
    if inst.kind == "point":
        mask = force_points(mask, [(x, y, inst.cls) for x, y in inst.points])
    else:  # triangle
        tmask = triangle_mask(inst.points, mask.shape)
        mask[tmask] = inst.cls
        conf[tmask] = 1.0
```

### 11.8 修改 `src/weak_labels/validate_labels.py`

输出改为“实例 / 单点 / 三角形 / 原始点”四列（对照 1.2），并新增错误类型：

- 三角形顶点数 ≠ 3；
- 退化三角形（面积 < 100 px²）；
- 同一三角形顶点越界；
- 实例数与原始点数不一致（解析丢点时必须显式报错，不能静默跳过）。

### 11.9 配置与验收

`configs/config.yaml` 新增：

```yaml
pseudo:
  rho_hard: 0.9
  rho_soft: 0.5
train:
  lambda_tri: 2.0
```

验收顺序（每步通过再进下一步）：

1. `python src/weak_labels/validate_labels.py` → 9 图 / 19 实例 / 15 单点 / 4 三角形 / 27 原始点，0 error；
2. `python -c "from src.data.instances import load_instances; ..."` 检查 0053 解析出 7 个实例（3 三角形 + 4 单点）；
3. `python src/weak_labels/make_pseudo.py` → 每张图 `tri_consistency ≥ 0.95`，点一致性 100%；
4. 小样本训练 3 epoch，确认日志里 `tri` 列 > 0 且下降；
5. `python src/test.py` → 仍输出 6 张图的 4 类文件。

**禁止事项**：

- 不得把三角形内部当 GT 去算 mIoU（它只是内点集）；
- 不得在 `parse_instances` 里把 3 点三角形拆成 3 个独立实例（这正是旧实现的错误）；
- 不得因某张图没有三角形就跳过该图（单点图仍要正常训练）；
- 不得改动 `label.json`（只读）。
