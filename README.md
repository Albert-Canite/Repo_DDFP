# Repo_DDFP
For reproducing the experiments of AnDi (https://www.nature.com/articles/s41928-024-01315-9)

## RSNA 回归与 DDFP 对比流程概览
- `experiments/rsna_regression.py` 先用 PyTorch 训练一个全精度（FP32）单通道 CNN（`RegressionNet`），用标准卷积和均值池化学习 512×512 胸片的肺炎框坐标。
- 训练完成后，脚本用 FP32 模型在测试集上得到基准框（`run_network_fp32`），并把卷积核导出到全局配置（`core.config.set_kernels`）。
- 接着对相同的卷积核做两条推理分支：
  - **DDFP**：`ddfp/calibrate_ddfp` 先用校准集搜索输入/权重的量化步长（默认 5bit 输入、5bit 权重、8bit ADC），再用 `run_network_ddfp` 逐层执行“整数量化 → 3×3/4×4 MAC → ADC 线性量化 → 反量化并可选 ReLU”，生成特征图供回归头解码。
  - **Baseline**：`ddfp/run_network_baseline` 用固定的线性量化（同 5bit 输入/权重、8bit ADC）在软件中跑等价的整数卷积链路，作为传统线性量化的对照。
- 最终以同一个回归头解码 DDFP 与 Baseline 的特征输出，对比 MAE/IoU 与可视化标注，检验 DDFP 与全精度、传统量化的精度差异。

## RSNA 回归模型结构与超参数

| 项目 | 配置 |
| --- | --- |
| 输入 | 单通道 512×512 归一化胸片，卷积核大小 4×4，padding=2，BatchNorm + ReLU 激活 | 
| 卷积主干 | 6 个卷积块，通道数依次为 32 → 32 → 64 → 64 → 96 → 96；每块为 Conv2d(4×4, bias=False) + BatchNorm2d + ReLU | 
| 下采样 | AvgPool2d(kernel=stride=32) 将特征汇聚到 16×16（池化后拼接坐标通道） | 
| 坐标编码 | 在 16×16 特征上拼接归一化的 x/y 网格坐标（共 2 个通道），随后展平成向量 | 
| 回归头 | 全连接 1024 输入 → 512 → 256 → 4，ReLU 间隔；末层偏置用数据先验（中心/尺寸）初始化 | 
| 训练损失 | L1(中心+尺寸) + 0.5×L1(角点) + 2.0×(1−IoU)，梯度裁剪 1.0 | 
| 训练批大小 | 8（默认），pin_memory=True，num_workers=2 | 
| 优化器 / 学习率 | Adam，初始学习率 3e-3，权重衰减 1e-4；10 epoch 线性 warmup 后余弦退火至 3e-5 | 
| 轮次与日志 | 共 120 epoch，训练集 4000 样本、验证集 800 样本；每个 epoch 首批及每 20 个 step 打印进度 | 
| 量化对比 | 使用同一 FP32 卷积核：FP32 推理 vs. DDFP（5bit 输入/权重，8bit ADC，自适应步长）vs. Baseline（同位宽固定线性量化） | 
