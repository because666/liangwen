# T-KAN + LightGBM 优化方案使用说明

## 概述

本项目实现了多种优化方案，旨在提高股票价格方向预测的收益率。当前基准模型准确率 0.589，但收益仅 0.06，说明需要从"预测准确"转向"收益优化"。

## 优化方案说明

### 第一梯队（快速见效）

#### 1. 动态阈值 + 后处理过滤
- **原理**：根据市场状态动态调整出手阈值，减少不利市场条件下的交易
- **文件**：`Predictor_optimized.py`
- **特点**：
  - 价差大时提高阈值（减少交易）
  - 波动率大时降低阈值（增加交易）
  - 后处理过滤：连续错误暂停、价差/波动率过滤

#### 2. 回归预测
- **原理**：直接预测价格变化率，根据预测值大小决定是否出手
- **文件**：`Predictor_regression.py`
- **特点**：
  - 预测值 > threshold: 上涨
  - 预测值 < -threshold: 下跌
  - 其他: 不变
  - 更直观地反映价格变化幅度

### 第二梯队（中期优化）

#### 3. 两阶段模型
- **原理**：先判断是否值得交易，再预测方向
- **文件**：`Predictor_two_stage.py`
- **特点**：
  - 阶段1：判断波动是否够大（是否值得交易）
  - 阶段2：只对值得交易的样本预测方向
  - 减少低价值交易

#### 4. 代价敏感学习
- **原理**：对不同错误类型设置不同惩罚
- **文件**：`Predictor_cost_sensitive.py`
- **特点**：
  - 预测涨但实际跌：高惩罚
  - 预测跌但实际涨：高惩罚
  - 大波动样本权重更高

## 使用流程

### 1. 训练模型

```bash
# 训练所有优化方案
python train_optimized.py --mode all

# 单独训练某个方案
python train_optimized.py --mode regression
python train_optimized.py --mode two_stage
python train_optimized.py --mode cost_sensitive
```

### 2. 本地测试

```bash
# 测试所有方案
python test_optimized.py --mode all

# 测试单个方案
python test_optimized.py --mode dynamic
python test_optimized.py --mode regression
python test_optimized.py --mode two_stage
python test_optimized.py --mode cost_sensitive
```

### 3. 打包提交

```bash
# 打包所有方案
python pack_submission.py --mode all

# 打包单个方案
python pack_submission.py --mode dynamic
python pack_submission.py --mode regression
python pack_submission.py --mode two_stage
python pack_submission.py --mode cost_sensitive
```

## 文件结构

```
tkan_lgbm_optimized/
├── config.json              # 配置文件
├── model.py                 # T-KAN 模型定义
├── requirements.txt         # 依赖文件
├── train_optimized.py       # 综合训练脚本
├── test_optimized.py        # 本地测试脚本
├── pack_submission.py       # 打包脚本
├── Predictor_optimized.py   # 动态阈值 + 后处理过滤
├── Predictor_regression.py  # 回归预测
├── Predictor_two_stage.py   # 两阶段模型
├── Predictor_cost_sensitive.py  # 代价敏感学习
└── output_optimized/        # 训练输出目录
    ├── submission/          # 动态阈值模型
    ├── regression/          # 回归模型
    ├── two_stage/           # 两阶段模型
    └── cost_sensitive/      # 代价敏感模型
```

## 预期效果

| 方案 | 预期改进 | 实现难度 |
|------|----------|----------|
| 动态阈值 | 减少不利交易，提高单次收益 | 低 |
| 后处理过滤 | 避免连续亏损，提高稳定性 | 低 |
| 回归预测 | 更精确的出手时机 | 中 |
| 两阶段模型 | 减少低价值交易 | 中 |
| 代价敏感学习 | 减少严重错误 | 中 |

## 注意事项

1. **训练前**：确保已有 T-KAN 编码器预训练权重（`output/tkan_encoder.pt`）
2. **数据路径**：默认使用 `../../2026train_set/2026train_set`，可通过参数修改
3. **GPU 支持**：自动检测 CUDA，无 GPU 时使用 CPU
4. **内存管理**：使用 `sample_interval` 参数控制数据加载量

## 下一步建议

1. 先测试动态阈值方案，观察收益变化
2. 如果收益提升明显，继续尝试回归预测
3. 如果收益仍不理想，尝试两阶段模型或代价敏感学习
4. 可以组合多种方案（如动态阈值 + 回归预测）
