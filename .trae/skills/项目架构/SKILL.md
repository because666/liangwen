---
name: "项目架构"
description: "良文杯高频量化预测项目架构说明。包含技术栈、模块设计、数据字段、训练策略和提交规范。在开发新功能或修改代码前应先了解此架构。"
---

# 良文杯 - T-KAN+OFI 高频方向预测项目架构

## 一、项目概述

### 竞赛目标
预测股票未来中间价的移动方向（下跌/不变/上涨），共5个预测窗口（5/10/20/40/60 tick）。

### 评价标准
- **累计收益率**（已扣除双边手续费0.02%）
- 单次收益率 = 累计收益率 / (预测上涨总数 + 预测下跌总数)

### 核心创新
1. **T-KAN编码器**：用可学习B样条替代线性权重，学习非线性时序特征
2. **OFI特征**：订单流不平衡指标，捕捉买卖压力方向
3. **收益感知损失**：直接优化交易收益，而非分类准确率
4. **出手门机制**：让模型自己学会"何时值得交易"

## 二、技术栈

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| 开发语言 | Python 3.10 | 平台指定 |
| 深度学习框架 | PyTorch 2.0+ | 兼容A6000/A10 GPU |
| 数据处理 | Pandas, NumPy | 数据加载与OFI特征计算 |
| 时序编码器 | TKAN / RKAN | B样条时序特征提取 |
| 损失函数 | Huber + Profit-Aware Loss | 直接优化交易收益 |

## 三、数据字段说明

### 原始数据字段（163列）

#### 基础行情（8列）
- `date`, `sym`, `time` - 日期、标的、时间戳
- `open`, `high`, `low`, `close` - OHLC价格（无量纲，相对昨收涨跌幅）
- `volume_delta`, `amount_delta` - 成交量/金额变化

#### 十档订单簿（40列）
- `bid1` ~ `bid10` - 买一价~买十价（无量纲）
- `ask1` ~ `ask10` - 卖一价~卖十价
- `bsize1` ~ `bsize10` - 买一量~买十量（无量纲，换手率%）
- `asize1` ~ `asize10` - 卖一量~卖十量

#### 订单流统计（18列）
- `lb_intst`, `la_intst` - 限价买单/卖单到达强度
- `mb_intst`, `ma_intst` - **市价买单/卖单到达强度**（OFI核心字段）
- `cb_intst`, `ca_intst` - 买单/卖单撤销强度
- `lb_ind`, `la_ind`, `mb_ind`, `ma_ind`, `cb_ind`, `ca_ind` - 强度指标标志
- `lb_acc`, `la_acc`, `mb_acc`, `ma_acc`, `cb_acc`, `ca_acc` - 强度平均变化率

#### 衍生特征（92列）
- `midprice`, `midprice1` ~ `midprice10` - 中间价
- `spread1` ~ `spread10` - 买卖价差
- `bid_diff1` ~ `bid_diff10`, `ask_diff1` ~ `ask_diff10` - 档位价差
- `bid_mean`, `ask_mean`, `bsize_mean`, `asize_mean` - 均价/均量
- `cumspread`, `imbalance` - 累计价差、不平衡
- `avgbid`, `avgask`, `totalbsize`, `totalasize` - 委买/委卖均价和总量
- `bid_rate1` ~ `bid_rate10`, `ask_rate1` ~ `ask_rate10` - 价格变化率
- `bsize_rate1` ~ `bsize_rate10`, `asize_rate1` ~ `asize_rate10` - 量变化率

#### 标签（5列）
- `label_5`, `label_10`, `label_20`, `label_40`, `label_60` - 5个窗口的方向标签（0=下跌, 1=不变, 2=上涨）

### OFI 特征计算（新增5列）

```python
# 1. 原始OFI（基于市价单强度）
ofi_raw = mb_intst - ma_intst

# 2. 指数加权移动OFI（捕捉衰减效应）
ofi_ewm = 0.9 * ofi_prev + 0.1 * ofi_raw

# 3. 多档OFI（Multi-Level OFI）
ofi_multilevel = sum([0.5, 0.3, 0.1, 0.05, 0.05][k] * (mb_intst - ma_intst) for k in range(5))

# 4. OFI变化率（加速度）
ofi_velocity = ofi_raw - ofi_raw.shift(1)

# 5. OFI波动率（不确定性）
ofi_volatility = ofi_raw.rolling(window=10).std()
```

## 四、模型架构

```
输入: (batch, 100, F)  # 100 ticks × F特征
    │
    ▼
┌─────────────────────────────────────┐
│  T-KAN/RKAN 编码器 (2-3层)          │
│  - B样条网格数: 8                    │
│  - B样条阶数: 3                      │
│  - 隐藏维度: 256 → 128               │
└─────────────────────────────────────┘
    │
    ▼
特征向量 h: (batch, 128)
    │
    ├─────────────────┬─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────┐     ┌─────────┐     ┌─────────────┐
│ 回归头×5│     │ 出手门  │     │             │
│ 预测ΔP  │     │ sigmoid │     │             │
│ Huber   │     │ BCE     │     │             │
└────┬────┘     └────┬────┘     └─────────────┘
     │               │
     └───────┬───────┘
             │
             ▼
┌─────────────────────────────────────┐
│  收益感知聚合层                      │
│  if p_action > τ:                    │
│    pred = 2 if ΔP > 0 else 0        │
│  else:                               │
│    pred = 1 (不变)                   │
└─────────────────────────────────────┘
    │
    ▼
输出: (batch, 5)  # 5个窗口的预测标签
```

## 五、损失函数

### Stage 1: Huber Loss（预训练）
```python
loss = F.smooth_l1_loss(pred_delta, true_delta)
```

### Stage 2: Profit-Aware Loss（微调）
```python
def profit_aware_loss(pred_delta, true_delta, action_prob, threshold=0.5, gamma=0.03):
    action = (action_prob > threshold).float()
    trade_dir = torch.sign(pred_delta)  # +1买, -1卖, 0不变
    
    # 实际收益 = 方向 × 真实变化 - 双边手续费0.02%
    actual_return = trade_dir * true_delta - 0.0002
    realized_return = action * actual_return
    
    # 损失 = 负收益 + 出手惩罚
    loss = -realized_return.mean() + gamma * action.mean()
    return loss
```

## 六、训练策略

### 阶段划分
1. **Stage 1（50 epochs）**：预训练回归头+编码器，Huber Loss
2. **Stage 2（20 epochs）**：联合微调回归头+出手门，Profit-Aware Loss
3. **阈值调优**：验证集搜索最优阈值（0.3~0.7）

### 超参数
- 优化器：AdamW, lr=1e-3, weight_decay=1e-5
- 学习率调度：CosineAnnealingLR
- Batch Size：256~512
- 早停：验证集收益连续5轮不提升

## 七、目录结构

```
d:\量化\良文杯\
├── 2026train_set/              # 训练数据
├── 新版本_架构V2/              # 新架构开发目录
│   ├── models/                 # 模型定义
│   ├── training/               # 训练脚本
│   ├── submission/             # 提交文件
│   ├── utils/                  # 工具函数
│   └── data_processing/        # 数据处理
├── 旧版本_备份/                # 旧版本备份
├── .trae/
│   ├── rules/project_rules.md  # 项目规则
│   └── skills/
│       ├── 良文杯评测提交/      # 提交规范skill
│       └── 项目架构/           # 本skill
└── 全新架构.md                 # 架构设计文档
```

## 八、提交规范

### 必需文件
```
submission.zip
├── Predictor.py        # 预测类（必须处理Polars DataFrame）
├── model.py           # 模型定义
├── best_model.pt      # 模型权重
├── best_thresholds.json  # 阈值配置
├── config.json        # 配置文件
└── requirements.txt   # 依赖（必须包含pyarrow）
```

### 关键注意事项
1. **Polars DataFrame**：评测平台传入Polars格式，必须先转换为Pandas
2. **特征计算**：OFI特征需要在推理时实时计算
3. **绝对路径**：使用 `os.path.dirname(__file__)` 加载模型
4. **torch.load**：添加 `weights_only=False` 参数

## 九、可行性分析

### ✅ 已确认可行
- [x] OFI相关字段存在（mb_intst, ma_intst等）
- [x] 数据包含163列，特征丰富
- [x] T-KAN有官方开源实现
- [x] PyTorch 2.0+ 支持B样条操作

### ⚠️ 需要注意
- T-KAN实现需要从官方仓库提取
- 收益感知损失需要梯度裁剪防止不稳定
- 阈值搜索需使用独立验证集防止过拟合

### 预期效果
- 模型参数量：~12M（可调至30M以内）
- 训练时间：约2-3小时（A10 GPU）
- 推理时间：<0.5秒/batch
- 预期收益提升：30%~80%
