# T-KAN+OFI 架构 V2

## 版本信息

- **创建日期**: 2026-04-08
- **架构名称**: T-KAN+OFI (Temporal Kolmogorov-Arnold Network + Order Flow Imbalance)
- **状态**: 已归档

## 模型特点

### 核心组件

1. **T-KAN 编码器**
   - 使用 B-样条函数替代线性权重
   - 多层时序特征提取
   - 参数：hidden_dim=256, output_dim=128, num_layers=2, grid_size=8

2. **OFI 特征工程**
   - ofi_raw: 原始订单流不平衡
   - ofi_ewm: 指数加权移动平均
   - ofi_multilevel: 多层级加权
   - ofi_velocity: 速度特征
   - ofi_volatility: 波动率特征

3. **预测头**
   - 回归头：预测价格变化量
   - 动作门：决定出手时机

### 训练策略

- **两阶段训练**:
  - Stage 1: Huber Loss 预训练
  - Stage 2: Profit-Aware Loss 微调

- **优化器**: Adam (lr=1e-3)
- **批次大小**: 256
- **训练数据**: 1200个parquet文件

## 文件说明

| 文件名 | 说明 |
|--------|------|
| `best_model.pt` | 最终模型权重 (71.4 MB) |
| `best_model_stage1.pt` | Stage 1 模型权重 (71.4 MB) |
| `best_thresholds.json` | 最佳阈值配置 |
| `config.json` | 模型配置 |
| `Predictor.py` | 预测类实现 |
| `requirements.txt` | 依赖列表 |
| `submission.zip` | 提交包 (58.1 MB) |

## 经验教训

### 关键问题

1. **Polars DataFrame 兼容性**
   - 评测平台传入的是 Polars DataFrame
   - 解决方案：使用 `df_to_numpy()` 转换为 numpy 数组

2. **config.json 的 feature 列表陷阱**
   - config.json 中的 feature 会被评测平台用于验证
   - 只能包含原始列，衍生特征必须在 Predictor 中计算

3. **模型输入维度**
   - 原始列：46个
   - 衍生列：8个
   - 总计：54个特征

## 性能表现

- **训练时间**: ~2小时 (快速训练模式)
- **模型参数量**: ~120万
- **提交状态**: 多次尝试解决 Polars 兼容性问题

## 后续改进方向

1. 简化特征工程，减少 Polars 兼容性问题
2. 使用更简单的模型架构
3. 优化 config.json 的设计

## 相关链接

- 训练代码: `新版本_架构V2/training/train.py`
- 模型定义: `新版本_架构V2/models/model.py`
- 损失函数: `新版本_架构V2/training/losses.py`
