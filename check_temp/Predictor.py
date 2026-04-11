"""
T-KAN Pro Predictor - 评测提交版本

关键原则：
1. 绝不往 DataFrame 写入新列！
2. config.json 只包含原始列
3. 衍生特征在内部用 numpy 计算
4. 直接三分类输出
"""

from typing import List
import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def df_to_numpy(df):
    """将任意 DataFrame 转换为 numpy 数组"""
    if isinstance(df, pd.DataFrame):
        return df.to_numpy(dtype=np.float32, copy=False), list(df.columns)

    if hasattr(df, 'to_pandas'):
        try:
            pdf = df.to_pandas()
            return pdf.to_numpy(dtype=np.float32, copy=False), list(pdf.columns)
        except Exception:
            pass

    if hasattr(df, 'to_arrow') or hasattr(df, '__arrow_table__'):
        try:
            import pyarrow as pa
            table = df.to_arrow() if hasattr(df, 'to_arrow') else df.__arrow_table__()
            pdf = table.to_pandas()
            return pdf.to_numpy(dtype=np.float32, copy=False), list(pdf.columns)
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
        return np.column_stack(data), cols

    return np.zeros((0, 0), dtype=np.float32), []


def compute_derived_numpy(arr, cols):
    """纯 numpy 计算衍生特征"""
    col_idx = {c: i for i, c in enumerate(cols)}
    n = arr.shape[0]

    def get_col(name):
        return arr[:, col_idx[name]].copy() if name in col_idx else np.zeros(n, dtype=np.float32)

    bid1 = get_col('bid1')
    ask1 = get_col('ask1')
    midprice = np.zeros(n, dtype=np.float32)
    both = (bid1 != 0) & (ask1 != 0)
    midprice[both] = (bid1[both] + ask1[both]) / 2
    midprice[(bid1 == 0) & (ask1 != 0)] = ask1[(bid1 == 0) & (ask1 != 0)]
    midprice[(ask1 == 0) & (bid1 != 0)] = bid1[(ask1 == 0) & (bid1 != 0)]

    total_b = sum(get_col(f'bsize{i}') for i in range(1, 11))
    total_a = sum(get_col(f'asize{i}') for i in range(1, 11))
    total = total_b + total_a
    imbalance = np.zeros(n, dtype=np.float32)
    mask = total > 0
    imbalance[mask] = (total_b[mask] - total_a[mask]) / total[mask]

    cumspread = sum(get_col(f'ask{i}') - get_col(f'bid{i}') for i in range(1, 11))

    mb = get_col('mb_intst')
    ma = get_col('ma_intst')
    ofi_raw = mb - ma

    ofi_ewm = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ofi_ewm[i] = ofi_raw[i] if i == 0 else 0.9 * ofi_ewm[i - 1] + 0.1 * ofi_raw[i]

    ofi_velocity = np.zeros(n, dtype=np.float32)
    ofi_velocity[1:] = ofi_raw[1:] - ofi_raw[:-1]

    ofi_volatility = np.zeros(n, dtype=np.float32)
    for i in range(10, n):
        ofi_volatility[i] = np.std(ofi_raw[i - 10:i])

    return {
        'midprice': midprice, 'imbalance': imbalance, 'cumspread': cumspread,
        'ofi_raw': ofi_raw, 'ofi_ewm': ofi_ewm,
        'ofi_velocity': ofi_velocity, 'ofi_volatility': ofi_volatility,
    }


def clean_features(features: np.ndarray, feature_cols: list) -> np.ndarray:
    """清洗特征数据"""
    features = features.copy()
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    for i, col in enumerate(feature_cols):
        if col.startswith('bid') or col.startswith('ask'):
            features[:, i] = np.clip(features[:, i], -0.3, 0.3)
        elif col.startswith('bsize') or col.startswith('asize'):
            features[:, i] = np.clip(features[:, i], 0, 100)
        elif 'intst' in col:
            features[:, i] = np.clip(features[:, i], -100, 100)
        elif col == 'midprice':
            features[:, i] = np.clip(features[:, i], -0.3, 0.3)
        elif col == 'imbalance':
            features[:, i] = np.clip(features[:, i], -1, 1)
        elif col.startswith('cumspread'):
            features[:, i] = np.clip(features[:, i], -1, 1)
        elif col.startswith('ofi'):
            features[:, i] = np.clip(features[:, i], -100, 100)
    
    return features


# ========== T-KAN Pro 模型定义（必须与训练时完全一致） ==========

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
                 spline_order: int = 3, dropout: float = 0.15):
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


class FeatureEmbedding(nn.Module):
    """特征嵌入层"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class TemporalAggregation(nn.Module):
    """时序聚合层"""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)
        x = x.mean(dim=1)
        return x


class ProfitGuidedHead(nn.Module):
    """收益导向预测头"""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, 
                 num_windows: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_windows = num_windows
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifiers = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(num_windows)])
        self.return_predictors = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_windows)])
    
    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        logits = torch.stack([cls(h) for cls in self.classifiers], dim=1)
        return_pred = torch.stack([pred(h).squeeze(-1) for pred in self.return_predictors], dim=1)
        return logits, return_pred


class TKANPro(nn.Module):
    """T-KAN Pro 模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_tkan_layers: int = 4, num_windows: int = 5,
                 grid_size: int = 8, spline_order: int = 3,
                 num_heads: int = 8, dropout: float = 0.15):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_windows = num_windows
        
        self.feature_embedding = FeatureEmbedding(input_dim, hidden_dim, dropout=0.1)
        self.tkan_layers = nn.ModuleList([
            TKANLayer(hidden_dim, grid_size, spline_order, dropout)
            for _ in range(num_tkan_layers)
        ])
        self.temporal_agg = TemporalAggregation(hidden_dim, num_heads, dropout)
        self.prediction_head = ProfitGuidedHead(hidden_dim, 128, num_windows, dropout)
    
    def forward(self, x: torch.Tensor):
        x = self.feature_embedding(x)
        for layer in self.tkan_layers:
            x = layer(x)
        x = self.temporal_agg(x)
        logits, return_pred = self.prediction_head(x)
        return logits, return_pred
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return logits.argmax(dim=-1)


class Predictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'best_model.pt')
        config_path = os.path.join(current_dir, 'config.json')

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.raw_feature_cols = self.config['feature']
        self.derived_feature_cols = [
            'midprice', 'imbalance', 'cumspread',
            'ofi_raw', 'ofi_ewm', 'ofi_velocity', 'ofi_volatility',
        ]
        self.feature_cols = self.raw_feature_cols + self.derived_feature_cols

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.mean = checkpoint.get('mean', None)
        self.std = checkpoint.get('std', None)

        self.model = TKANPro(
            input_dim=len(self.feature_cols),
            hidden_dim=256,
            num_tkan_layers=4,
            num_windows=5,
            grid_size=8,
            spline_order=3,
            num_heads=8,
            dropout=0.15,
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        print(f"模型加载完成，设备: {self.device}")

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        batch_size = len(x)
        features = []

        for df in x:
            arr, cols = df_to_numpy(df)
            col_idx = {c: i for i, c in enumerate(cols)}
            derived = compute_derived_numpy(arr, cols)

            feature_arrays = []
            for col_name in self.raw_feature_cols:
                if col_name in col_idx:
                    feature_arrays.append(arr[:, col_idx[col_name]])
                else:
                    feature_arrays.append(np.zeros(arr.shape[0], dtype=np.float32))
            for col_name in self.derived_feature_cols:
                if col_name in derived:
                    feature_arrays.append(derived[col_name])
                else:
                    feature_arrays.append(np.zeros(arr.shape[0], dtype=np.float32))

            feature_data = np.column_stack(feature_arrays).astype(np.float32)
            feature_data = clean_features(feature_data, self.feature_cols)

            if self.mean is not None and self.std is not None:
                feature_data = (feature_data - self.mean) / self.std
                feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

            features.append(feature_data)

        features = np.array(features, dtype=np.float32)
        features_tensor = torch.from_numpy(features).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.model.predict(features_tensor)

        results = []
        for i in range(batch_size):
            results.append([int(p) for p in predictions[i].cpu().tolist()])

        return results
