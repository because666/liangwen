---
name: "良文杯评测提交"
description: "良文杯竞赛评测提交专用技能。在创建或修改 Predictor.py、打包提交评测前必须调用，确保代码符合评测平台规范。"
---

# 良文杯评测提交技能

## 核心问题

### 问题1：Polars DataFrame 转换

**评测平台传入的数据格式是 Polars DataFrame，不是 Pandas DataFrame！**

评测平台的 `predict` 方法签名虽然标注为 `List[pd.DataFrame]`，但实际传入的是 `List[polars.DataFrame]`。

### 问题2：config.json 中的 feature 列表陷阱（重要！）

**config.json 中的 `feature` 列表会被评测平台用于验证输入数据！**

如果 config.json 中包含评测数据不存在的列（如衍生特征 `midprice`, `ofi_volatility` 等），评测平台会在调用 `predict` 之前尝试访问这些列，导致报错：

```
No match for FieldRef.Name(ofi_volatility) in date: string, sym: string, ...
```

**解决方案**：
- **config.json 中的 `feature` 只能包含评测平台实际提供的原始列**
- **衍生特征必须在 Predictor.py 内部计算，不写入 config.json**

---

## 正确的 config.json 格式

```json
{
  "python_version": "3.10",
  "batch": 256,
  "feature": [
    "bid1", "bid2", ..., "bid10",
    "ask1", "ask2", ..., "ask10",
    "bsize1", "bsize2", ..., "bsize10",
    "asize1", "asize2", ..., "asize10",
    "mb_intst", "ma_intst", "lb_intst", "la_intst", "cb_intst", "ca_intst"
  ],
  "label": ["label_5", "label_10", "label_20", "label_40", "label_60"]
}
```

**注意**：不要包含 `midprice`, `imbalance`, `cumspread`, `ofi_*` 等衍生特征！

---

## 正确的 Predictor.py 实现模板

```python
from typing import List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import json


def df_to_numpy(df):
    """将任意 DataFrame 转换为 numpy 数组，返回 (数据数组, 列名列表)"""
    if isinstance(df, pd.DataFrame):
        cols = list(df.columns)
        arr = df.to_numpy(dtype=np.float32, copy=False)
        return arr, cols

    # Polars 或其他类型：先转 pandas 再提取
    if hasattr(df, 'to_pandas'):
        try:
            pdf = df.to_pandas()
            cols = list(pdf.columns)
            arr = pdf.to_numpy(dtype=np.float32, copy=False)
            return arr, cols
        except Exception:
            pass

    # 通过 pyarrow 中转
    if hasattr(df, 'to_arrow') or hasattr(df, '__arrow_table__'):
        try:
            import pyarrow as pa
            if hasattr(df, 'to_arrow'):
                table = df.to_arrow()
            else:
                table = df.__arrow_table__()
            pdf = table.to_pandas()
            cols = list(pdf.columns)
            arr = pdf.to_numpy(dtype=np.float32, copy=False)
            return arr, cols
        except Exception:
            pass

    # 逐列提取
    if hasattr(df, 'columns') and hasattr(df, '__len__'):
        cols = list(df.columns)
        data = []
        for col in df.columns:
            col_data = df[col]
            if hasattr(col_data, 'to_numpy'):
                data.append(col_data.to_numpy().astype(np.float32))
            elif hasattr(col_data, 'to_list'):
                data.append(np.array(col_data.to_list(), dtype=np.float32))
            else:
                data.append(np.zeros(len(df), dtype=np.float32))
        arr = np.column_stack(data)
        return arr, cols

    return np.zeros((0, 0), dtype=np.float32), []


def compute_derived_features(arr, cols):
    """
    从 numpy 数组计算衍生特征
    关键：不操作 DataFrame，只用 numpy 计算
    """
    col_idx = {c: i for i, c in enumerate(cols)}
    n = arr.shape[0]

    def get_col(name):
        if name in col_idx:
            return arr[:, col_idx[name]].copy()
        return np.zeros(n, dtype=np.float32)

    # 计算 midprice
    bid1 = get_col('bid1')
    ask1 = get_col('ask1')
    midprice = np.zeros(n, dtype=np.float32)
    both_nonzero = (bid1 != 0) & (ask1 != 0)
    bid_zero = (bid1 == 0) & (ask1 != 0)
    ask_zero = (ask1 == 0) & (bid1 != 0)
    midprice[both_nonzero] = (bid1[both_nonzero] + ask1[both_nonzero]) / 2
    midprice[bid_zero] = ask1[bid_zero]
    midprice[ask_zero] = bid1[ask_zero]

    # 计算 imbalance
    total_bsize = np.zeros(n, dtype=np.float32)
    total_asize = np.zeros(n, dtype=np.float32)
    for i in range(1, 11):
        total_bsize += get_col(f'bsize{i}')
        total_asize += get_col(f'asize{i}')
    total_size = total_bsize + total_asize
    imbalance = np.zeros(n, dtype=np.float32)
    mask = total_size > 0
    imbalance[mask] = (total_bsize[mask] - total_asize[mask]) / total_size[mask]

    # 计算 cumspread
    cumspread = np.zeros(n, dtype=np.float32)
    for i in range(1, 11):
        cumspread += get_col(f'ask{i}') - get_col(f'bid{i}')

    # 计算 OFI 特征
    mb_intst = get_col('mb_intst')
    ma_intst = get_col('ma_intst')
    ofi_raw = mb_intst - ma_intst

    ofi_ewm = np.zeros(n, dtype=np.float32)
    alpha = 0.1
    for i in range(n):
        if i == 0:
            ofi_ewm[i] = ofi_raw[i]
        else:
            ofi_ewm[i] = (1 - alpha) * ofi_ewm[i - 1] + alpha * ofi_raw[i]

    ofi_velocity = np.zeros(n, dtype=np.float32)
    ofi_velocity[1:] = ofi_raw[1:] - ofi_raw[:-1]

    ofi_volatility = np.zeros(n, dtype=np.float32)
    window = 10
    for i in range(window, n):
        ofi_volatility[i] = np.std(ofi_raw[i - window:i])

    return {
        'midprice': midprice,
        'imbalance': imbalance,
        'cumspread': cumspread,
        'ofi_raw': ofi_raw,
        'ofi_ewm': ofi_ewm,
        'ofi_velocity': ofi_velocity,
        'ofi_volatility': ofi_volatility,
    }


class Predictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'best_model.pt')
        config_path = os.path.join(current_dir, 'config.json')

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 关键：config.json 只包含原始列，但模型需要完整特征
        self.raw_feature_cols = self.config['feature']  # 来自 config.json
        self.derived_feature_cols = [  # 内部计算
            'midprice', 'imbalance', 'cumspread',
            'ofi_raw', 'ofi_ewm', 'ofi_velocity', 'ofi_volatility'
        ]
        self.feature_cols = self.raw_feature_cols + self.derived_feature_cols

        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = YourModelClass(len(self.feature_cols))
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        batch_size = len(x)

        features = []
        for df in x:
            # 关键：只读不写！先用 numpy 提取，再计算特征
            arr, cols = df_to_numpy(df)
            col_idx = {c: i for i, c in enumerate(cols)}

            # 计算衍生特征（纯 numpy，不碰 DataFrame）
            derived = compute_derived_features(arr, cols)

            # 按顺序组装特征数组：先原始列，再衍生列
            feature_arrays = []
            # 1. 原始列（来自 config.json）
            for col_name in self.raw_feature_cols:
                if col_name in col_idx:
                    feature_arrays.append(arr[:, col_idx[col_name]])
                else:
                    feature_arrays.append(np.zeros(arr.shape[0], dtype=np.float32))
            # 2. 衍生列（计算得到）
            for col_name in self.derived_feature_cols:
                if col_name in derived:
                    feature_arrays.append(derived[col_name])
                else:
                    feature_arrays.append(np.zeros(arr.shape[0], dtype=np.float32))

            feature_data = np.column_stack(feature_arrays).astype(np.float32)
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(feature_data)

        features = np.array(features, dtype=np.float32)
        features_tensor = torch.from_numpy(features).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(features_tensor)
            # 处理输出...

        return results
```

---

## requirements.txt 格式

```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=10.0.0
```

---

## 打包提交

```powershell
# Windows PowerShell
Compress-Archive -Path Predictor.py,config.json,best_model.pt,requirements.txt -DestinationPath submission.zip -Force
```

---

## 常见错误及解决方案

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `No match for FieldRef.Name(xxx)` | config.json 包含评测数据没有的列 | **config.json 只放原始列，衍生列在 Predictor 中计算** |
| `torch.load weights_only` 错误 | PyTorch 2.6+ 默认值变更 | 添加 `weights_only=False` 参数 |
| 解压缩失败 | 使用 tar 格式 | 使用 PowerShell `Compress-Archive` |
| 模型输入维度不匹配 | 特征数与训练时不一致 | 确保 `len(raw_feature_cols) + len(derived_feature_cols)` 等于训练时的特征数 |

---

## 评测平台数据列

评测平台提供的原始数据列：
- `date`, `sym`, `time`, `open`, `high`, `low`, `close`
- `volume_delta`, `amount_delta`
- `bid1` ~ `bid10`, `ask1` ~ `ask10`
- `bsize1` ~ `bsize10`, `asize1` ~ `asize10`
- `mb_intst`, `ma_intst`, `lb_intst`, `la_intst`, `cb_intst`, `ca_intst`

**重要**：训练数据中的衍生特征（如 `midprice`, `imbalance`, `cumspread`, `ofi_*` 等）在评测数据中不存在，**必须在 Predictor 中自行计算，且不能写入 config.json**。

---

## 检查清单（提交前必读）

1. [ ] **config.json 中的 feature 只包含评测平台提供的原始列**
2. [ ] **衍生特征（midprice, imbalance, ofi_* 等）在 Predictor.py 中计算**
3. [ ] **Predictor.py 绝不往 DataFrame 写入新列**
4. [ ] **使用 `df_to_numpy()` 提取数据，用 numpy 计算特征**
5. [ ] **使用绝对路径加载模型文件**
6. [ ] **requirements.txt 包含 pyarrow**
7. [ ] **使用正确的 ZIP 格式打包**
8. [ ] **torch.load 添加 `weights_only=False` 参数**

---

## 经验教训总结

### 本次问题的根本原因

1. **config.json 包含了衍生特征列**（如 `ofi_volatility`）
2. **评测平台在调用 predict 之前**，使用 config.json 的 feature 列表验证输入数据
3. **评测平台尝试访问 Polars DataFrame 的 `ofi_volatility` 列**，但该列不存在，报错

### 正确的设计模式

```
config.json (原始列)  →  评测平台验证  →  Predictor.predict()  →  numpy计算衍生特征  →  模型推理
```

**绝不能**：
```
config.json (原始列+衍生列)  →  评测平台验证(访问不存在的列)  →  报错
```

