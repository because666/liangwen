# T-KAN + LightGBM 云GPU训练包

## 架构说明

本训练包采用 **T-KAN 编码器 + LightGBM 分类器** 的两阶段架构：

1. **T-KAN 编码器预训练**：使用 Huber Loss 预测 5 个窗口的中间价变化率
2. **LightGBM 分类器训练**：基于 T-KAN 提取的 128 维特征进行三分类

## 目录结构

```
TKAN_LGBM_Cloud_Training_Package/
├── data/                          # 训练数据 (约470MB)
│   └── 2026train_set/            # 1200个parquet文件
├── src/                          # 源代码
│   ├── model.py                  # T-KAN 模型定义
│   ├── train_tkan.py            # T-KAN 预训练脚本
│   ├── train_lgbm.py            # LightGBM 训练脚本
│   ├── Predictor.py             # 推理代码
│   ├── config.json              # 评测配置
│   └── requirements.txt         # 依赖列表
├── scripts/                      # 脚本
│   ├── setup_env.sh             # 环境安装
│   └── run_train.sh             # 训练启动
└── README.md                     # 本文件
```

## 快速开始

### 1. 安装环境

```bash
cd TKAN_LGBM_Cloud_Training_Package
bash scripts/setup_env.sh
```

### 2. 运行训练

```bash
bash scripts/run_train.sh
```

### 3. 自定义参数

```bash
# 修改batch size和epoch数
BATCH_SIZE=512 EPOCHS=50 bash scripts/run_train.sh

# 只使用部分数据快速测试
MAX_FILES=100 bash scripts/run_train.sh

# 修改模型结构
HIDDEN_DIM=256 NUM_LAYERS=4 bash scripts/run_train.sh
```

## 训练流程

### 阶段1: T-KAN 编码器预训练

- 输入: (batch, 100, 40) - 仅原始价量特征
- 输出: (batch, 128) - 特征向量
- 目标: 预测 5 个窗口的中间价变化率
- 损失: Huber Loss
- 输出文件: `output/tkan_encoder.pt`

### 阶段2: LightGBM 分类器训练

- 输入: 128 维 T-KAN 特征
- 输出: 5 个窗口的三分类 (0/1/2)
- 模型: 5 个独立的 LightGBM 分类器
- 输出文件: `output/submission/`

## 提交文件

训练完成后，提交文件位于 `output/submission/`：

- `tkan_encoder.pt` - 编码器权重 + LightGBM模型（内嵌）
- `Predictor.py` - 推理代码
- `config.json` - 评测配置
- `requirements.txt` - 依赖

打包命令：
```bash
cd output/submission
zip -r ../../submission.zip .
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| BATCH_SIZE | 256 | 训练批次大小 |
| EPOCHS | 30 | T-KAN预训练轮数 |
| LR | 1e-4 | 学习率 |
| HIDDEN_DIM | 128 | T-KAN隐藏维度 |
| NUM_LAYERS | 3 | T-KAN层数 |
| MAX_FILES | - | 最大文件数（用于测试） |

## 硬件要求

- **GPU**: NVIDIA GPU with CUDA 11.8+
- **内存**: 建议 16GB+
- **存储**: 约 2GB（含数据）

## 训练时间估算

- T-KAN预训练: 约 2-4 小时（V100/A100）
- LightGBM训练: 约 10-30 分钟
- 总计: 约 3-5 小时
