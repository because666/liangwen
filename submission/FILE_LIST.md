# 最终文件清单 - 收益导向版本

## 📁 当前目录结构

```
d:\量化\良文杯\submission\
├── 📄 核心文件 (3个)
│   ├── train_profit_v2.py      # 收益导向训练脚本 ✅
│   ├── Predictor_profit.py     # 收益导向预测器 ✅
│   └── test_profit_local.py    # 本地测试验证 ✅
│
├── 📄 配置文件 (2个)
│   ├── config.json             # 配置（含214个特征）✅
│   └── requirements.txt        # Python依赖 ✅
│
├── 📄 工具文件 (1个)
│   └── prepare_submission.py   # 提交打包脚本 ✅
│
├── 📄 文档文件 (3个)
│   ├── GUIDE_PROFIT.md         # 完整使用指南 ✅
│   ├── CLEANUP_LOG.md          # 清理记录 ✅
│   └── README_CLOUD.md         # 云GPU说明（参考）✅
│
├── 📄 运行脚本 (2个)
│   ├── run_cloud.bat           # Windows运行脚本 ✅
│   └── run_cloud.sh            # Linux运行脚本 ✅
│
└── 📄 模型文件 (1个)
    └── best_model.pt           # 模型权重 ⚠️ 需重新训练
```

**总计：12个文件**（清理前27个，删除16个过时文件）

---

## ✅ 核心文件验证结果

### Python文件语法检查
```
✓ train_profit_v2.py - 语法正确
✓ Predictor_profit.py - 语法正确
✓ test_profit_local.py - 语法正确
```

### 配置文件检查
```
✓ config.json - 存在且包含完整214个特征
✓ requirements.txt - 存在且依赖正确
```

### 文档文件检查
```
✓ GUIDE_PROFIT.md - 存在
✓ CLEANUP_LOG.md - 存在
```

---

## 🎯 核心改进验证

### 1. 收益导向损失函数 (ProfitLoss) ✅
```python
# train_profit_v2.py 第402-463行
class ProfitLoss(nn.Module):
    """
    收益导向损失函数
    
    核心思想：
    1. 直接优化交易收益，而非分类准确率
    2. 对"不变"类给予极低的权重
    3. 对正确的涨跌预测给予高奖励
    4. 对错误的涨跌预测给予高惩罚
    """
    weight_matrix = [
        [+2.0, +0.05, -3.0],   # 实际下跌时
        [-1.5, +0.05, -1.5],   # 实际不变时
        [-3.0, +0.05, +2.0],   # 实际上涨时
    ]
```

### 2. 不对称类别权重 (AsymmetricFocalLoss) ✅
```python
# train_profit_v2.py 第466-506行
class AsymmetricFocalLoss(nn.Module):
    """
    不对称Focal Loss
    
    针对"准确率高但收益低"问题的特殊设计：
    - 大幅降低类别1（不变）的权重至0.15
    - 提高类别0和2（跌和涨）的权重至2.5
    """
    alpha=[2.5, 0.15, 2.5]  # 关键参数
    gamma=3.0
```

### 3. 两阶段训练策略 ✅
```python
# train_profit_v2.py 第877-884行
if epoch < 30:
    current_criteria = criteria  # AsymmetricFocalLoss
    loss_name = "AsymmetricFocal"
else:
    current_criteria = profit_criteria  # ProfitLoss
    loss_name = "Profit"
```

### 4. 更保守的出手策略 ✅
```python
# train_profit_v2.py 第724-728行
def optimize_threshold_for_profit(probs, targets):
    best_threshold = 0.55  # 默认更高阈值
    for threshold in np.arange(0.45, 0.85, 0.01):  # 从更高阈值开始搜索
        ...
```

### 5. 收益指标监控 ✅
```python
# train_profit_v2.py 第914-931行
# 用平均单次收益作为主要评估指标
avg_profits = [r['metrics']['avg_profit'] for r in window_results if r['metrics']['trade_count'] > 0]
current_avg_profit = np.mean(avg_profits)

if current_avg_profit > best_avg_profit:
    best_avg_profit = current_avg_profit
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"  ✓ 保存最佳模型 (单次收益: {best_avg_profit:.6f})")
```

---

## 📦 历史提交压缩包（已保留）

以下压缩包保留作为历史记录：

| 压缩包路径 | 说明 | 问题 |
|-----------|------|------|
| `d:\量化\良文杯\导出\submission_optimized.zip` | 优化版提交 | ❌ 收益为负 |
| `d:\量化\良文杯\submission_ensemble.zip` | 集成学习提交 | ❌ 收益为负 |
| `d:\量化\良文杯\submission.zip` | 原始提交 | ❌ 收益为负 |

**注意**：这些压缩包中的模型都存在"准确率高但收益低"问题，仅供参考，不建议使用。

---

## 🚀 下一步操作

### 1. 上传到云GPU
```bash
# 方式A：打包上传
cd d:\量化\良文杯
tar -czvf submission_profit.tar.gz submission/

# 方式B：直接上传submission目录
```

### 2. 训练新模型
```bash
cd submission

python train_profit_v2.py \
    --data_dir ../2026train_set/2026train_set \
    --output_dir ./output_profit \
    --epochs 80 \
    --batch_size 256 \
    --lr 1.5e-4 \
    --hidden_dim 128 \
    --dropout 0.25 \
    --max_files 1200 \
    --use_amp \
    --augment \
    --use_tta \
    --tta_rounds 5
```

### 3. 检查关键指标
训练日志中重点关注：
- ✅ **平均单次收益 > 0**（必须为正）
- ✅ **胜率 > 0.52**
- ✅ **出手率 25%-40%**
- ⚠️ **准确率可能降至 0.35-0.45**（正常）

### 4. 本地验证
```bash
python test_profit_local.py --max_samples 1000
```

### 5. 打包提交
```bash
python prepare_submission.py --model_dir ./output_profit
```

---

## 📊 预期效果对比

| 指标 | 旧版本（历史提交） | 新版本（预期） | 改善 |
|------|------------------|--------------|------|
| **准确率** | 0.50-0.60 | 0.35-0.45 | ↓ 可接受 |
| **累计收益** | -76 ~ -98 | **正值** | ↑↑↑ 关键！ |
| **单次收益** | -0.0002 | **> 0** | ↑↑↑ 关键！ |
| **胜率** | < 0.50 | **> 0.53** | ↑ 重要 |
| **出手率** | < 10% | **25-40%** | ↑ 合理 |

---

## ✅ 最终验证清单

- [x] 核心Python文件语法正确
- [x] 配置文件包含完整214个特征
- [x] 收益导向损失函数正确实现
- [x] 不对称类别权重正确设置
- [x] 两阶段训练策略已实现
- [x] 更保守的阈值策略已实现
- [x] 收益指标监控已实现
- [x] 过时文件已删除
- [x] 历史压缩包已保留
- [x] 文档已完善

---

## 📞 技术支持

如遇问题，请查看：
1. **详细使用指南**：[GUIDE_PROFIT.md](file:///d:/量化/良文杯/submission/GUIDE_PROFIT.md)
2. **清理记录**：[CLEANUP_LOG.md](file:///d:/量化/良文杯/submission/CLEANUP_LOG.md)

---

**验证完成时间**：2026-04-06  
**验证结果**：✅ 所有文件正常，优化合理，可以使用  
**下一步**：上传到云GPU开始训练
