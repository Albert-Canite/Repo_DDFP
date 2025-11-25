# Repo_DDFP
Repository root (unchanged): `/workspace/Repo_DDFP`.
For reproducing the experiments of AnDi (https://www.nature.com/articles/s41928-024-01315-9)

## Archive usage
- The `archive/` folder is a static snapshot of the BCCD dataset (images and `annotations.csv`).
- Scripts reference it with paths relative to the repo root (e.g., `archive/images/...`).
- Keep the folder in place if you move or mount the repo; no external download is needed.

## RSNA 回归与 DDFP 对比流程概览
- `experiments/rsna_regression.py` 先用 PyTorch 训练 YOLO-tiny 风格的全精度（FP32）单通道 CNN（`RegressionNet`），用 stride-2 卷积完成下采样并回归 512×512 胸片的肺炎框坐标。
- 训练完成后，脚本用 FP32 模型在测试集上得到基准框（`run_network_fp32`），并把每层卷积核及其 stride/padding 导出到全局配置（`core.config.set_kernels` + `set_kernel_metadata`）。
- 接着对相同的卷积核做两条推理分支：
  - **DDFP**：`ddfp/calibrate_ddfp` 先用校准集搜索输入/权重的量化步长（默认 5bit 输入、5bit 权重、8bit ADC），再用 `run_network_ddfp` 逐层执行“整数量化 → stride/padding 感知的 3×3/1×1 MAC → ADC 线性量化 → 反量化并可选 ReLU”，生成特征图供回归头解码。
  - **Baseline**：`ddfp/run_network_baseline` 用同位宽的固定线性量化在软件中跑整数卷积链路，作为传统线性量化的对照。
- 最终以同一个回归头解码 DDFP 与 Baseline 的特征输出，对比 MAE/IoU 与可视化标注，检验 DDFP 与全精度、传统量化的精度差异。

## RSNA 回归模型结构与超参数

| 项目 | 配置 |
| --- | --- |
| 输入 | 单通道 512×512 归一化胸片，所有卷积核大小为 3×3（1×1 压缩层除外），BatchNorm + LeakyReLU(0.1) |
| YOLO-tiny 主干 | Stage1: 3×3 s1 16 → 3×3 s2 32；Stage2: 3×3 s1 32 → 3×3 s2 64；Stage3: 3×3 s1 64 → 3×3 s2 128；Stage4: 3×3 s1 128 → 3×3 s2 256；Stage5: 3×3 s1 256 → 3×3 s2 512；随后 1×1 conv 512→256、3×3 conv 256→256 |
| 下采样与特征 | 仅靠 stride-2 卷积逐级下采样到 16×16，末端全局平均池化得到 256 维特征向量 |
| 回归头 | MLP：256 → 128 → 64 → 4，ReLU 间隔；输出经 Sigmoid 约束到 [0,1]（x_center, y_center, width, height），末层偏置用数据先验初始化 |
| 训练损失 | L1(中心+尺寸) + 0.5×L1(角点) + 2.0×(1−IoU)，梯度裁剪 1.0 |
| 训练批大小 | 4（默认），pin_memory=True，num_workers=2 |
| 优化器 / 学习率 | Adam，初始学习率 2e-3，权重衰减 1e-4；8 epoch 线性 warmup 后余弦退火至 2e-5 |
| 轮次与日志 | 共 80 epoch，训练集 2000 样本、验证集 400 样本；校准 20 张、测试 2 张；每个 epoch 首批及每 20 个 step 打印进度 |
| 量化对比 | 使用同一 FP32 卷积核：FP32 推理 vs. DDFP（5bit 输入/权重，8bit ADC，自适应步长，支持 stride/padding）vs. Baseline（同位宽固定线性量化） |
