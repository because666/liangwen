# 高频量化预测模型 - 云GPU使用指南

## 📋 目录结构

```
submission/
├── train_cloud.py           # 最终训练脚本
├── Predictor_final.py       # 最终预测器
├── run_cloud.sh             # 一键运行脚本 (Linux)
├── run_cloud.bat            # 一键运行脚本 (Windows)
├── prepare_submission.py    # 提交准备脚本
├── requirements.txt         # Python依赖
└── README_CLOUD.md          # 本文件
```

## 🚀 快速开始

### 方法一：一键运行（推荐）

**Linux/云GPU:**
```bash
chmod +x run_cloud.sh
./run_cloud.sh
```

**Windows:**
```cmd
run_cloud.bat
```

### 方法二：手动运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行训练
python train_cloud.py \
    --data_dir ./2026train_set/2026train_set \
    --output_dir ./output_final \
    --epochs 60 \
    --batch_size 256 \
    --use_amp \
    --augment \
    --use_tta

# 3. 准备提交
python prepare_submission.py --model_dir ./output_final
```

## ⚙️ 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data` | 训练数据目录 |
| `--output_dir` | `./output_final` | 输出目录 |
| `--epochs` | `60` | 训练轮数 |
| `--batch_size` | `256` | 批次大小 |
| `--lr` | `2e-4` | 学习率 |
| `--hidden_dim` | `128` | 隐藏层维度 |
| `--max_files` | `1200` | 最大文件数 |

### 优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_amp` | 开启 | 混合精度训练 |
| `--augment` | 开启 | 数据增强 |
| `--use_tta` | 开启 | 测试时增强 |
| `--tta_rounds` | `5` | TTA轮数 |
| `--label_smoothing` | `0.1` | 标签平滑 |

## 💾 GPU显存需求

| 配置 | 显存需求 | A10适配 |
|------|----------|---------|
| 基础 (hidden_dim=128) | ~8GB | ✅ |
| 标准 (hidden_dim=160) | ~12GB | ✅ |
| 高配 (hidden_dim=192) | ~16GB | ✅ |
| 超高 (hidden_dim=256) | ~20GB | ✅ |

**A10 (24GB) 推荐配置：**
```bash
python train_cloud.py \
    --hidden_dim 160 \
    --batch_size 256 \
    --use_amp
```

## 📊 预期效果

| 优化项 | 预期收益 |
|--------|----------|
| OFI特征 (25个) | 提高对价格变化的预测能力 |
| 多尺度TCN | 同时捕捉局部和全局模式 |
| CBAM注意力 | 自适应关注重要特征 |
| 数据增强 | 提高泛化能力 |
| TTA (5轮) | 提高预测稳定性 |
| 置信度过滤 | 减少错误交易 |

## 🔧 常见问题

### Q1: 显存不足怎么办？

```bash
# 减小batch_size
python train_cloud.py --batch_size 128

# 或减小hidden_dim
python train_cloud.py --hidden_dim 96
```

### Q2: 训练时间太长？

```bash
# 减少文件数（快速测试）
python train_cloud.py --max_files 100 --epochs 10

# 减少TTA轮数
python train_cloud.py --tta_rounds 3
```

### Q3: 如何查看训练进度？

训练过程会实时显示：
- 每个epoch的损失和准确率
- 每个窗口的最优阈值
- 置信度过滤后的准确率

### Q4: 如何使用已训练的模型？

```python
from Predictor_final import Predictor

predictor = Predictor()
results = predictor.predict(test_data)
```

## 📁 输出文件

训练完成后，`output_final/` 目录包含：

```
output_final/
├── best_model.pt    # 最佳模型权重
├── config.json      # 配置文件
└── thresholds.json  # 最优阈值
```

## 🎯 提交流程

1. **训练模型**
   ```bash
   python train_cloud.py --use_amp --augment --use_tta
   ```

2. **准备提交**
   ```bash
   python prepare_submission.py --model_dir ./output_final
   ```

3. **检查提交**
   ```bash
   # 确保submission目录包含：
   # - config.json
   # - Predictor.py (或 Predictor_final.py)
   # - best_model.pt
   # - requirements.txt
   ```

4. **打包提交**
   ```bash
   zip -r submission.zip submission/
   ```

## 📈 性能优化建议

### A10 GPU优化

```bash
# 最佳配置（效果优先）
python train_cloud.py \
    --hidden_dim 160 \
    --batch_size 256 \
    --epochs 60 \
    --use_amp \
    --augment \
    --use_tta \
    --tta_rounds 5

# 快速配置（时间优先）
python train_cloud.py \
    --hidden_dim 128 \
    --batch_size 512 \
    --epochs 30 \
    --use_amp \
    --max_files 800
```

### A100/H100 GPU优化

```bash
# 更大模型
python train_cloud.py \
    --hidden_dim 256 \
    --batch_size 512 \
    --epochs 80 \
    --use_amp \
    --augment \
    --use_tta \
    --tta_rounds 7
```

## ⚠️ 注意事项

1. **数据路径**：确保 `--data_dir` 指向正确的训练数据目录

2. **Python版本**：建议使用 Python 3.10

3. **CUDA版本**：建议 CUDA 11.8+ 或 CUDA 12.0+

4. **依赖安装**：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install pandas numpy pyarrow tqdm scipy
   ```

5. **TTA开销**：TTA会增加推理时间，但能提高稳定性

---

祝训练顺利！如有问题，请检查日志输出。
