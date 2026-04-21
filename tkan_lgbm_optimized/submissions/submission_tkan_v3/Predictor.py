"""
T-KAN 端到端分类器预测器

架构：
- T-KAN 编码器 + 分类头
- 直接预测 5 个窗口的标签（三分类：0/1/2）

评测规范：
1. 使用 os.path.dirname(__file__) 加载模型（绝对路径）
2. 处理 Polars DataFrame（兼容 Pandas/Polars/PyArrow）
3. 只使用 config.json 中列出的 40 维原始价量特征
4. 返回 List[List[int]] 格式
"""

import os
import json
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pyarrow as pa
except ImportError:
    pa = None


FEATURE_COLS = [
    'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'bid6', 'bid7', 'bid8', 'bid9', 'bid10',
    'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask6', 'ask7', 'ask8', 'ask9', 'ask10',
    'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'bsize6', 'bsize7', 'bsize8', 'bsize9', 'bsize10',
    'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'asize6', 'asize7', 'asize8', 'asize9', 'asize10',
]

WINDOW_SIZES = [5, 10, 20, 40, 60]


class StableSplineLinear(nn.Module):
    """数值稳定的 B-spline 线性层"""

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 8, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        grid = torch.linspace(-1, 1, grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.base_weight, gain=0.5)
        nn.init.normal_(self.spline_weight, std=0.01)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -2.0, 2.0)
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            denom1 = grid[k:-1] - grid[:-(k + 1)]
            denom1 = torch.where(denom1.abs() < 1e-8, torch.ones_like(denom1), denom1)
            denom2 = grid[k + 1:] - grid[1:-k]
            denom2 = torch.where(denom2.abs() < 1e-8, torch.ones_like(denom2), denom2)
            bases = (
                (x - grid[:-(k + 1)]) / denom1 * bases[..., :-1]
                + (grid[k + 1:] - x) / denom2 * bases[..., 1:]
            )
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        base_output = F.linear(x, self.base_weight)
        spline_basis = self.b_splines(x)
        spline_output = torch.einsum('bij,oij->bo', spline_basis, self.spline_weight)
        output = base_output + spline_output
        return output.view(*original_shape[:-1], self.out_features)


class TKANLayer(nn.Module):
    """T-KAN 单层（带残差连接）"""

    def __init__(self, features: int, grid_size: int = 8,
                 spline_order: int = 3, dropout: float = 0.0):
        super().__init__()
        self.features = features
        self.spline_linear = StableSplineLinear(features, features, grid_size, spline_order)
        self.layer_norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.spline_linear(x)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = x + residual
        return x


class TKANClassifier(nn.Module):
    """T-KAN 端到端分类器"""

    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, num_layers: int = 3,
                 grid_size: int = 8, spline_order: int = 3, dropout: float = 0.0,
                 num_windows: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_windows = num_windows

        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.tkan_layers = nn.ModuleList([
            TKANLayer(hidden_dim, grid_size, spline_order, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_windows * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_embedding(x)
        for layer in self.tkan_layers:
            x = layer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits.view(-1, self.num_windows, 3)


def df_to_numpy(df) -> tuple:
    """将 DataFrame 转换为 numpy 数组"""
    if isinstance(df, pd.DataFrame):
        cols = list(df.columns)
        arr = df.to_numpy(dtype=np.float32, copy=False)
        return arr, cols

    if hasattr(df, 'to_pandas'):
        try:
            pdf = df.to_pandas()
            cols = list(pdf.columns)
            arr = pdf.to_numpy(dtype=np.float32, copy=False)
            return arr, cols
        except Exception:
            pass

    if pa is not None and (hasattr(df, 'to_arrow') or hasattr(df, '__arrow_table__')):
        try:
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


def extract_features_from_df(arr: np.ndarray, cols: list,
                             feature_cols: list) -> np.ndarray:
    """从 numpy 数组提取指定特征列"""
    col_idx = {c: i for i, c in enumerate(cols)}
    n = arr.shape[0]
    result = np.zeros((n, len(feature_cols)), dtype=np.float32)

    for i, col in enumerate(feature_cols):
        if col in col_idx:
            result[:, i] = arr[:, col_idx[col]]

    return result


def clean_features(features: np.ndarray) -> np.ndarray:
    """特征清洗"""
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    for i, col in enumerate(FEATURE_COLS):
        if col.startswith('bid') or col.startswith('ask'):
            features[:, i] = np.clip(features[:, i], -0.3, 0.3)
        elif col.startswith('bsize') or col.startswith('asize'):
            features[:, i] = np.clip(features[:, i], 0, 100)
    return features


class Predictor:
    """T-KAN 端到端分类器预测器"""

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        model_path = os.path.join(current_dir, 'best_model.pt')
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.config = checkpoint.get('config', {})
        self.mean = checkpoint.get('mean', None)
        self.std = checkpoint.get('std', None)

        input_dim = self.config.get('input_dim', len(FEATURE_COLS))
        hidden_dim = self.config.get('hidden_dim', 128)
        num_layers = self.config.get('num_layers', 3)
        num_windows = self.config.get('num_windows', len(WINDOW_SIZES))

        self.model = TKANClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
            num_windows=num_windows,
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'], strict=False)
        self.model.eval()
        print(f"T-KAN 分类器加载成功: hidden_dim={hidden_dim}, num_layers={num_layers}")

        config_path = os.path.join(current_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                eval_config = json.load(f)
            self.feature_cols = eval_config.get('feature', FEATURE_COLS)
        else:
            self.feature_cols = FEATURE_COLS

    def preprocess(self, x: List) -> torch.Tensor:
        """预处理输入数据"""
        batch_size = len(x)
        seq_len = 100
        feature_dim = len(self.feature_cols)

        features = np.zeros((batch_size, seq_len, feature_dim), dtype=np.float32)

        for i, df in enumerate(x):
            arr, cols = df_to_numpy(df)
            df_features = extract_features_from_df(arr, cols, self.feature_cols)
            df_features = clean_features(df_features)

            actual_len = min(len(df_features), seq_len)
            features[i, :actual_len, :] = df_features[:actual_len]

        if self.mean is not None and self.std is not None:
            mean = self.mean[:feature_dim]
            std = self.std[:feature_dim]
            features = (features - mean) / std
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(features).to(self.device, dtype=torch.float32)

    def predict(self, x: List) -> List[List[int]]:
        """预测接口（支持阈值调整）
        
        阈值说明：
        - 阈值越低 → 交易次数越多 → 召回率越高 → 但准确率可能下降
        - 阈值越高 → 交易次数越少 → 召回率越低 → 但准确率可能提高
        
        调整建议：
        - label_40/60 历史表现好，可以降低阈值（0.35-0.45）
        - label_5/10 噪声大，可以提高阈值（0.55-0.65）
        """
        features_tensor = self.preprocess(x)

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    logits = self.model(features_tensor)
            else:
                logits = self.model(features_tensor)

        # ============================================================
        # 可调参数：阈值设置
        # 修改这里的值来调整交易频率
        # ============================================================
        THRESHOLDS = {
            5: 0.42,    # label_5 阈值（短窗口，建议 0.55-0.65）
            10: 0.42,   # label_10 阈值（短窗口，建议 0.50-0.60）
            20: 0.40,   # label_20 阈值（中窗口，建议 0.45-0.55）
            40: 0.41,   # label_40 阈值（长窗口，建议 0.40-0.50）← 重点调整
            60: 0.43,   # label_60 阈值（长窗口，建议 0.40-0.50）← 重点调整
        }
        # ============================================================

        # 获取概率分布
        probs = F.softmax(logits, dim=-1)  # (batch, num_windows, 3)
        
        batch_size = probs.size(0)
        preds = []
        
        for i in range(batch_size):
            window_preds = []
            for w_idx, w in enumerate([5, 10, 20, 40, 60]):
                p = probs[i, w_idx]  # [p_down, p_unchanged, p_up]
                tau = THRESHOLDS[w]
                
                # 预测逻辑
                if p[2] > tau:       # 涨的概率 > 阈值
                    pred = 2
                elif p[0] > tau:     # 跌的概率 > 阈值
                    pred = 0
                else:
                    pred = 1         # 不变
                
                window_preds.append(pred)
            preds.append(window_preds)

        return preds


if __name__ == '__main__':
    predictor = Predictor()

    test_data = []
    for _ in range(4):
        df_dict = {}
        for col in FEATURE_COLS:
            if col.startswith('bid') or col.startswith('ask'):
                df_dict[col] = np.random.randn(100) * 0.01
            else:
                df_dict[col] = np.random.rand(100) * 10
        test_data.append(pd.DataFrame(df_dict))

    predictions = predictor.predict(test_data)
    print(f"预测结果: {predictions}")
    print(f"预测形状: {len(predictions)} x {len(predictions[0])}")
