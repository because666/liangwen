# 良文杯 - 统计建模与AI预测挑战赛（金融大数据主题）

## 项目概述

这是一个股票高频数据预测竞赛项目，目标是利用股票过往及当前数据预测未来中间价的移动方向。

## 竞赛核心规则

### 预测目标
- **任务**：预测未来中间价(midprice)的移动方向
- **中间价定义**：
  - 当 ask1 和 bid1 都不为零时：midprice = (ask1 + bid1) / 2
  - 当 bid1 为零时（跌停）：midprice = ask1
  - 当 ask1 为零时（涨停）：midprice = bid1

### 预测时间窗口
- 5个预测任务：t+5tick, t+10tick, t+20tick, t+40tick, t+60tick
- 对应标签：label_5, label_10, label_20, label_40, label_60

### 移动方向定义（三分类）
- x = 待预测时刻midprice - 当前时刻midprice
- x < -α 时，ϕ(x) = 0（下跌）
- -α ≤ x ≤ α 时，ϕ(x) = 1（不变）
- x > α 时，ϕ(x) = 2（上涨）
- **α取值**：
  - 5tick, 10tick：α = 0.05%
  - 20tick, 40tick, 60tick：α = 0.1%

### 评价标准
- **累计收益率**：所有预测上涨/下跌的收益率之和（已扣除手续费0.01%）
- **单次收益率** = 累计收益率 / (预测的上涨总数 + 预测的下跌总数)
- 取5个预测任务中的最高得分参与排行

## 数据说明

### 行情频率
- 3秒一个数据点（1个tick的snapshot）
- 允许利用过去100tick（包含当前tick）的数据进行预测

### 数据列说明

#### 基础行情数据
| 字段 | 说明 |
|------|------|
| date | 日期（0-119，保留跨日标的可比性） |
| time | 时间戳（保留实际时间间隔，3s一档行情） |
| sym | 股票标的（0-4，5只股票） |
| open | 开盘价（无量纲，相对昨收的涨跌幅） |
| high | 最高价 |
| low | 最低价 |
| close | 最新成交价 |
| amount_delta | 成交金额变化（从上tick到当前tick的成交金额，单位：元） |
| volume_delta | 成交量变化（无量纲，以换手率%表示） |

#### 十档订单簿
| 字段 | 说明 |
|------|------|
| bid1 ~ bid10 | 买一价~买十价（无量纲，以涨跌幅表示） |
| ask1 ~ ask10 | 卖一价~卖十价（无量纲，以涨跌幅表示） |
| bsize1 ~ bsize10 | 买一量~买十量（无量纲，以换手率%表示） |
| asize1 ~ asize10 | 卖一量~卖十量（无量纲，以换手率%表示） |

#### 订单簿衍生特征
| 字段 | 说明 |
|------|------|
| midprice | 中间价（无量纲，以涨跌幅表示） |
| midprice1 ~ midprice10 | 1档~10档中间价 |
| spread1 ~ spread10 | 十档买卖价差 |
| bid_diff1 ~ bid_diff10 | 档位价差 |
| ask_diff1 ~ ask_diff10 | 档位价差 |
| bid_mean / ask_mean | 十档均价 |
| bsize_mean / asize_mean | 十档均量 |
| cumspread | 十档累计买卖价差 |
| imbalance | 十档累计买卖量差 |
| avgbid / avgask | 委买/委卖均价（十档） |
| totalbsize / totalasize | 委买/委卖总量（十档） |

#### 订单流统计特征
| 字段 | 说明 |
|------|------|
| lb_intst / la_intst | 限价买单/卖单到达强度 |
| mb_intst / ma_intst | 市价买单/卖单到达强度 |
| cb_intst / ca_intst | 买单/卖单撤销强度 |
| lb_ind / la_ind / mb_ind / ma_ind / cb_ind / ca_ind | 上述六类订单强度指标（1为True，0为False） |
| lb_acc / la_acc / mb_acc / ma_acc / cb_acc / ca_acc | 上述六类订单强度平均变化率 |

#### 变化率特征
| 字段 | 说明 |
|------|------|
| bid_rate1 ~ bid_rate10 | 十档价格平均变化率 |
| ask_rate1 ~ ask_rate10 | 十档价格平均变化率 |
| bsize_rate1 ~ bsize_rate10 | 十档量平均变化率 |
| asize_rate1 ~ asize_rate10 | 十档量平均变化率 |

#### 标签列
| 字段 | 说明 |
|------|------|
| label_5 | 5tick后价格移动方向（0=下跌，1=不变，2=上涨） |
| label_10 | 10tick后价格移动方向 |
| label_20 | 20tick后价格移动方向 |
| label_40 | 40tick后价格移动方向 |
| label_60 | 60tick后价格移动方向 |

## 评测提交规范

### 文件结构
```
submission.zip
├── config.json          # 配置文件
├── Predictor.py         # 预测类
├── model.py            # 模型定义（可选）
├── best_model.pt       # 模型权重
└── requirements.txt    # 依赖文件
```

### config.json 格式
```json
{
  "python_version": "3.10",
  "batch": 256,
  "feature": ["bid1", "ask1", "bsize1", "asize1", ...],
  "label": ["label_5", "label_10", "label_20", "label_40", "label_60"]
}
```

### Predictor 类规范
```python
from typing import List
import pandas as pd

class Predictor:
    def __init__(self):
        # 加载模型
        pass
    
    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        """
        输入: List[pd.DataFrame]，长度为batch
              每个DataFrame为100个tick的数据，列名为feature
        输出: List[List[int]]，长度为batch
              每个内层List长度为label个数，值为0/1/2
        """
        pass
```

### 评测环境
- **CPU模式**：16个CPU核
- **GPU模式**：1块NVIDIA RTX A6000（CUDA 13.0）
- **限制**：
  - 每12小时限提交1次
  - 单次评测时长≤3小时
  - 模型文件≤2GB（FP32精度）

## 项目目录结构

```
d:\量化\良文杯\
├── 2026train_set/              # 训练数据
│   └── 2026train_set/
│       └── snapshot_sym*_date*.parquet
├── example_本地测试代码/        # 本地测试示例
│   └── example/
│       ├── mmpc/
│       │   ├── config.json
│       │   ├── Predictor.py
│       │   ├── model.py
│       │   └── best_model.pt
│       ├── main.py
│       └── snapshot_sym0_date0_am.parquet
├── mmpc_模型demo/              # 模型demo
│   └── mmpc/
│       ├── config.json
│       ├── Predictor.py
│       ├── model.py
│       └── best_model.pt
├── 竞赛规则.md
├── 评测指南.md
└── .trae/rules/project_rules.md  # 本文件
```

## 开发注意事项

1. **数据预处理**：
   - 输入数据已归一化（无量纲）
   - 日期(date)在评测时会置为0
   - sym范围为0-4（可能包含非训练集股票数据）

2. **模型要求**：
   - 必须使用过去100tick数据
   - 预测5个时间窗口的方向
   - 输出格式必须为List[List[int]]

3. **性能优化**：
   - 建议使用batch推理
   - GPU评测需将模型和数据移至GPU
   - 预测结果需移回CPU

4. **提交前检查**：
   - 使用pipreqs生成纯净依赖
   - 在空白环境测试依赖安装
   - 确保所有文件在同一目录（无子文件夹）
