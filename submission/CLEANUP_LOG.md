# 项目文件清理记录

## 清理时间
2026-04-06

## 清理目的
整理项目文件夹，删除过时的旧版本文件，只保留最新的收益导向版本，避免混淆。

---

## ✅ 保留的核心文件（submission/目录）

### 训练相关
| 文件名 | 说明 | 状态 |
|--------|------|------|
| `train_profit_v2.py` | **收益导向训练脚本**（最新版） | ✅ 主力文件 |
| `test_profit_local.py` | 本地测试验证脚本 | ✅ 配套工具 |

### 预测相关
| 文件名 | 说明 | 状态 |
|--------|------|------|
| `Predictor_profit.py` | **收益导向预测器**（最新版） | ✅ 主力文件 |

### 配置相关
| 文件名 | 说明 | 状态 |
|--------|------|------|
| `config.json` | 配置文件（含214个特征） | ✅ 已更新 |
| `requirements.txt` | Python依赖 | ✅ 保留 |
| `prepare_submission.py` | 提交打包脚本 | ✅ 已更新 |

### 文档相关
| 文件名 | 说明 | 状态 |
|--------|------|------|
| `GUIDE_PROFIT.md` | **收益导向完整指南** | ✅ 主力文档 |
| `README_CLOUD.md` | 云GPU使用说明（旧版） | ✅ 参考文档 |

### 运行脚本
| 文件名 | 说明 | 状态 |
|--------|------|------|
| `run_cloud.bat` | Windows运行脚本 | ✅ 保留 |
| `run_cloud.sh` | Linux运行脚本 | ✅ 保留 |

### 模型文件
| 文件名 | 说明 | 状态 |
|--------|------|------|
| `best_model.pt` | 模型权重（需要重新训练） | ⚠️ 待更新 |

---

## ❌ 已删除的过时文件

### 旧版训练脚本（已删除）
- ❌ `train.py` - 原始训练脚本
- ❌ `train_advanced.py` - 阶段3高级优化版
- ❌ `train_diversified.py` - 阶段2差异化架构版
- ❌ `train_final.py` - 阶段整合版
- ❌ `train_profit_optimized.py` - 阶段1收益优化版
- ❌ `train_cloud.py` - 云GPU版
- ❌ `train_cloud_large.py` - 云GPU大模型版
- ❌ `train_ensemble.py` - 集成学习版

**删除原因**：所有这些版本都存在"准确率高但收益低"的问题，已被 `train_profit_v2.py` 替代。

### 旧版预测器（已删除）
- ❌ `Predictor.py` - 原始预测器
- ❌ `Predictor_advanced.py` - 高级优化版预测器
- ❌ `Predictor_diversified.py` - 差异化架构预测器
- ❌ `Predictor_final.py` - 最终版预测器
- ❌ `Predictor_v2.py` - V2版预测器
- ❌ `Predictor_ensemble.py` - 集成学习预测器

**删除原因**：所有这些预测器都配合旧版模型使用，已被 `Predictor_profit.py` 替代。

### 旧版模型定义（已删除）
- ❌ `model.py` - 原始模型定义
- ❌ `models_diversified.py` - 差异化架构模型定义

**删除原因**：模型定义已整合到 `train_profit_v2.py` 和 `Predictor_profit.py` 中。

### 旧版测试脚本（已删除）
- ❌ `test_local.py` - 旧版本地测试
- ❌ `evaluate.py` - 旧版评估脚本
- ❌ `analyze_features.py` - 特征分析脚本

**删除原因**：已被 `test_profit_local.py` 替代，后者专门针对收益导向评估。

### 其他（已删除）
- ❌ `__pycache__/` - Python缓存目录

**删除原因**：缓存文件，可重新生成。

---

## 📦 保留的历史提交压缩包

以下压缩包保留作为历史记录，位于项目根目录：

| 压缩包路径 | 说明 | 保留原因 |
|-----------|------|---------|
| `d:\量化\良文杯\导出\submission_optimized.zip` | 优化版提交 | 历史记录 |
| `d:\量化\良文杯\submission_ensemble.zip` | 集成学习提交 | 历史记录 |
| `d:\量化\良文杯\submission.zip` | 原始提交 | 历史记录 |

**注意**：这些压缩包中的模型都存在"准确率高但收益低"的问题，仅供参考，不建议使用。

---

## 🎯 当前推荐使用流程

### 1. 训练新模型（云GPU）
```bash
cd submission
python train_profit_v2.py \
    --data_dir ../2026train_set/2026train_set \
    --output_dir ./output_profit \
    --epochs 80 \
    --batch_size 256 \
    --use_amp \
    --augment \
    --use_tta
```

### 2. 本地验证
```bash
python test_profit_local.py --max_samples 1000
```

### 3. 打包提交
```bash
python prepare_submission.py --model_dir ./output_profit
```

---

## 📊 文件数量统计

### 清理前
- Python文件：21个
- 配置文件：2个
- 文档文件：2个
- 其他：2个
- **总计：27个文件**

### 清理后
- Python文件：**3个**（核心）
- 配置文件：2个
- 文档文件：2个
- 运行脚本：2个
- 其他：2个
- **总计：11个文件**

**清理效果**：删除了16个过时文件，文件数量减少59%，目录结构更清晰。

---

## ⚠️ 重要提醒

1. **best_model.pt 需要重新训练**
   - 当前文件是旧版模型，存在收益问题
   - 使用 `train_profit_v2.py` 重新训练

2. **不要使用旧的压缩包**
   - 所有历史压缩包中的模型都有收益问题
   - 仅作为记录保留

3. **核心文件只有3个**
   - `train_profit_v2.py` - 训练
   - `Predictor_profit.py` - 预测
   - `test_profit_local.py` - 测试

4. **详细使用说明**
   - 查看 `GUIDE_PROFIT.md` 获取完整指南

---

## ✅ 验证清理结果

运行以下命令验证文件完整性：

```bash
# 检查核心文件是否存在
ls -la train_profit_v2.py
ls -la Predictor_profit.py
ls -la test_profit_local.py
ls -la config.json

# 验证Python语法
python -m py_compile train_profit_v2.py
python -m py_compile Predictor_profit.py
python -m py_compile test_profit_local.py
```

---

**清理完成！项目现在只保留最新的收益导向版本，结构清晰，易于使用。**
