<br />

***

## 最终方案：T-KAN Pro 架构

### 设计理念

| 目标       | 策略                     |
| :------- | :--------------------- |
| **扩大参数** | 深度 T-KAN + 宽隐藏层 + 残差连接 |
| **收益导向** | 收益加权 CE + 轻量级收益预测头     |
| **训练稳定** | 数值稳定的 B-spline + 梯度裁剪  |

***

### 架构设计

```
输入: (batch, 100, 54)  # 100 tick, 54 features
         ↓
┌────────────────────────────────────────────┐
│  特征嵌入层                   │
│  Linear(54, 256) + LayerNorm + GELU        │  参数: 54×256 = 13,824
│  Dropout(0.1)                              │
└────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────┐
│  T-KAN 深度块 × 4 (核心)                    │
│                                            │
│  每个块:                                    │
│  ├─ TKANLayer(256, 256)  # B-spline 非线性 │  参数: ~256×256×(8+3) ≈ 720K/块
│  ├─ LayerNorm + GELU                       │
│  ├─ Dropout(0.15)                          │
│  └─ 残差连接                               │
│                                            │
│  总参数: ~290 万                            │
└────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────┐
│  时序聚合层                 │
│                                            │
│  方案: 轻量级注意力 + Mean Pool             │
│  MultiheadAttention(256, 8)                │  参数: 256×256×4 ≈ 260K
│  LayerNorm + 残差                          │
│  Mean(dim=1) → (batch, 256)                │
└────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────┐
│  收益导向预测头 (关键创新)                   │
│                                            │
│  共享层:                                    │
│  Linear(256, 128) + LayerNorm + GELU       │  参数: 256×128 = 32K
│                                            │
│  双分支:                                    │
│  ├─ 分类分支: 5 × Linear(128, 3)           │  参数: 128×3×5 = 1.9K
│  │   → 预测方向 (0/1/2)                    │
│  │                                         │
│  └─ 收益分支: 5 × Linear(128, 1)           │  参数: 128×1×5 = 0.6K
│      → 预测期望收益 (用于损失计算)          │
└────────────────────────────────────────────┘
         ↓
输出: 
  - 分类 logits: (batch, 5, 3)
  - 收益预测: (batch, 5)
```

**总参数量: 约 320 万**（比 V3 的 50 万大 6 倍，但结构更简单）

***

### 收益导向损失函数（核心创新）

```python
class ProfitGuidedLoss(nn.Module):
    """
    收益导向损失函数
    
    核心思想：
    1. 分类损失：收益加权的交叉熵
    2. 收益损失：预测收益与真实收益的 MSE
    3. 交易惩罚：过度交易的惩罚项
    
    关键：避免 NaN，数值稳定
    """
    
    def __init__(self, fee=0.0002, lambda_return=0.5, lambda_trade=0.1):
        super().__init__()
        self.fee = fee
        self.lambda_return = lambda_return  # 收益损失权重
        self.lambda_trade = lambda_trade    # 交易惩罚权重
    
    def forward(self, logits, return_pred, labels, true_returns):
        """
        Args:
            logits: (batch, 5, 3) 分类预测
            return_pred: (batch, 5) 收益预测
            labels: (batch, 5) 真实标签
            true_returns: (batch, 5) 真实收益
        """
        batch_size, num_windows = labels.shape
        
        # 1. 收益加权交叉熵
        ce_loss = 0
        for w in range(num_windows):
            w_logits = logits[:, w, :]
            w_labels = labels[:, w]
            w_returns = true_returns[:, w].abs()
            
            # 收益越大，权重越高（但限制范围）
            weights = torch.clamp(w_returns / self.fee, 0.5, 3.0)
            
            # 对"不变"类别给予基础权重
            hold_mask = (w_labels == 1).float()
            weights = weights * (1 - hold_mask) + 1.0 * hold_mask
            
            ce = F.cross_entropy(w_logits, w_labels, reduction='none')
            ce_loss = ce_loss + (ce * weights).mean()
        
        ce_loss = ce_loss / num_windows
        
        # 2. 收益预测损失（关键：直接优化收益）
        # 只对预测交易（非"不变"）的样本计算收益损失
        preds = logits.argmax(dim=-1)  # (batch, 5)
        trade_mask = (preds != 1).float()  # 交易掩码
        
        # 方向：上涨为 +1，下跌为 -1
        direction = torch.where(preds == 2, 1.0, 
                               torch.where(preds == 0, -1.0, 0.0))
        
        # 预期收益 = 方向 × 真实收益 - 手续费
        expected_return = direction * true_returns - self.fee
        
        # 收益损失：预测收益与期望收益的差异
        return_loss = F.mse_loss(
            return_pred * trade_mask, 
            expected_return * trade_mask, 
            reduction='sum'
        ) / (trade_mask.sum() + 1e-8)  # 防止除零
        
        # 3. 交易惩罚（鼓励观望）
        trade_rate = trade_mask.mean()
        trade_penalty = torch.relu(trade_rate - 0.5)  # 交易率超过 50% 才惩罚
        
        # 总损失
        total_loss = ce_loss + self.lambda_return * return_loss + self.lambda_trade * trade_penalty
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss.item(),
            'return_loss': return_loss.item(),
            'trade_rate': trade_rate.item(),
        }
```

***

### 数值稳定性保障

#### 1. **B-spline 稳定化**

```python
class StableSplineLinear(nn.Module):
    """数值稳定的 B-spline 线性层"""
    
    def __init__(self, in_features, out_features, grid_size=8, spline_order=3):
        super().__init__()
        # ... 初始化代码 ...
        
        # 关键：使用较小的初始化
        nn.init.xavier_uniform_(self.base_weight, gain=0.5)
        nn.init.normal_(self.spline_weight, std=0.01)  # 很小的初始化
    
    def b_splines(self, x):
        # ... B-spline 计算 ...
        
        # 关键：对输入进行 clamp，防止极端值
        x = torch.clamp(x, -2.0, 2.0)
        
        # ... 后续计算 ...
```

#### 2. **梯度裁剪策略**

```python
# 更激进的梯度裁剪
max_grad_norm = 0.3  # 比 0.5 更严格

# 梯度检查
for param in model.parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            param.grad = torch.zeros_like(param.grad)
```

#### 3. **学习率调度**

```python
# 更保守的学习率
lr = 3e-5  # 比之前更小

# Warm-up + Cosine
warmup_epochs = 5
scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
```

***

### 训练策略

| 阶段      | Epochs | 学习率         | 目标    |
| :------ | :----- | :---------- | :---- |
| Warm-up | 5      | 1e-5 → 3e-5 | 稳定初始化 |
| 主训练     | 15     | 3e-5 → 1e-5 | 收敛    |
| 微调      | 5      | 1e-6        | 精调    |

***

### 与 V3 的对比

| 方面    | V3 架构                | T-KAN Pro      |
| :---- | :------------------- | :------------- |
| 参数量   | 50 万                 | 320 万          |
| 专家数量  | 4 个                  | 无（单模型）         |
| 损失函数  | CE + Sharpe + Sparse | 收益加权 CE + 收益预测 |
| 训练稳定性 | 差（经常 NaN）            | 好（数值稳定）        |
| 可解释性  | 差（多专家融合）             | 好（单一 T-KAN）    |
| 预期效果  | 不稳定                  | 稳定且有提升         |

***

### 为什么这个方案更好？

#### 1. **参数效率高**

- T-KAN 的 B-spline 参数效率比 Transformer 高
- 320 万参数主要集中在非线性建模上

#### 2. **收益导向明确**

- 直接预测收益，而不是间接通过分类
- 收益损失直接优化目标

#### 3. **训练稳定**

- 单一模型，没有多专家冲突
- 数值稳定的 B-spline 实现
- 保守的学习率和梯度裁剪

#### 4. **避免过拟合**

- Dropout 0.15
- Weight Decay 1e-4
- Early Stopping

***

## 预期效果

| 指标      | 预期范围     |
| :------ | :------- |
| 训练 Loss | 稳定下降，不震荡 |
| 交易率     | 30-60%   |
| 验证收益    | 正收益      |
| NaN 出现  | 0 次      |

***

这个方案如何？如果认可，我可以开始实现代码。
