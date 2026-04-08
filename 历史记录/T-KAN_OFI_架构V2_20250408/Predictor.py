"""
T-KAN+OFI 高频方向预测 - Predictor 推理封装

关键原则：绝不往 DataFrame 写入新列！
只从 DataFrame 读取数据，用 numpy 计算特征，最后组装成 numpy 数组。
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


def compute_features_numpy(arr, cols):
    """
    从 numpy 数组计算所有特征，返回完整的特征数组
    
    关键：不操作 DataFrame，只用 numpy 计算
    """
    col_idx = {c: i for i, c in enumerate(cols)}
    n = arr.shape[0]

    def get_col(name):
        if name in col_idx:
            return arr[:, col_idx[name]].copy()
        return np.zeros(n, dtype=np.float32)

    # 计算衍生特征
    bid1 = get_col('bid1')
    ask1 = get_col('ask1')

    # midprice
    midprice = np.zeros(n, dtype=np.float32)
    both_nonzero = (bid1 != 0) & (ask1 != 0)
    bid_zero = (bid1 == 0) & (ask1 != 0)
    ask_zero = (ask1 == 0) & (bid1 != 0)
    midprice[both_nonzero] = (bid1[both_nonzero] + ask1[both_nonzero]) / 2
    midprice[bid_zero] = ask1[bid_zero]
    midprice[ask_zero] = bid1[ask_zero]

    # imbalance
    total_bsize = np.zeros(n, dtype=np.float32)
    total_asize = np.zeros(n, dtype=np.float32)
    for i in range(1, 11):
        total_bsize += get_col(f'bsize{i}')
        total_asize += get_col(f'asize{i}')
    total_size = total_bsize + total_asize
    imbalance = np.zeros(n, dtype=np.float32)
    mask = total_size > 0
    imbalance[mask] = (total_bsize[mask] - total_asize[mask]) / total_size[mask]

    # cumspread
    cumspread = np.zeros(n, dtype=np.float32)
    for i in range(1, 11):
        cumspread += get_col(f'ask{i}') - get_col(f'bid{i}')

    # OFI 特征
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

    weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05], dtype=np.float32)
    ofi_multilevel = np.sum(weights[:, None] * (mb_intst[None, :] - ma_intst[None, :]), axis=0)

    ofi_velocity = np.zeros(n, dtype=np.float32)
    ofi_velocity[1:] = ofi_raw[1:] - ofi_raw[:-1]

    ofi_volatility = np.zeros(n, dtype=np.float32)
    window = 10
    for i in range(window, n):
        ofi_volatility[i] = np.std(ofi_raw[i - window:i])

    # 组装所有衍生特征列
    derived_cols = ['midprice', 'imbalance', 'cumspread',
                    'mb_intst', 'ma_intst', 'lb_intst', 'la_intst', 'cb_intst', 'ca_intst',
                    'ofi_raw', 'ofi_ewm', 'ofi_multilevel', 'ofi_velocity', 'ofi_volatility']
    derived_data = {
        'midprice': midprice,
        'imbalance': imbalance,
        'cumspread': cumspread,
        'mb_intst': mb_intst,
        'ma_intst': ma_intst,
        'lb_intst': get_col('lb_intst'),
        'la_intst': get_col('la_intst'),
        'cb_intst': get_col('cb_intst'),
        'ca_intst': get_col('ca_intst'),
        'ofi_raw': ofi_raw,
        'ofi_ewm': ofi_ewm,
        'ofi_multilevel': ofi_multilevel,
        'ofi_velocity': ofi_velocity,
        'ofi_volatility': ofi_volatility,
    }

    return derived_data


class SplineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 8,
                 spline_order: int = 3):
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

        nn.init.kaiming_uniform_(self.base_weight)
        nn.init.zeros_(self.spline_weight)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)]) * bases[..., :-1]
                + (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:-k]) * bases[..., 1:]
            )
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        base_output = F.linear(x, self.base_weight)
        spline_basis = self.b_splines(x)
        spline_output = torch.einsum('bij,oij->bo', spline_basis, self.spline_weight)
        output = base_output + spline_output
        output = output.view(*original_shape[:-1], self.out_features)
        return output


class TKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 8,
                 spline_order: int = 3, dropout: float = 0.1):
        super().__init__()
        self.spline_linear = SplineLinear(in_features, out_features, grid_size, spline_order)
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spline_linear(x)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class TKANEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128,
                 num_layers: int = 2, grid_size: int = 8, spline_order: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layers = []
        dims = [hidden_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            layers.append(TKANLayer(dims[i], dims[i + 1], grid_size, spline_order, dropout))
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.final_norm(x)
        return x


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_windows: int = 5, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_windows)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ActionGate(nn.Module):
    def __init__(self, hidden_dim: int, num_windows: int = 5, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_windows),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


class TKANOFIModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128,
                 num_windows: int = 5, num_layers: int = 2, grid_size: int = 8,
                 spline_order: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_windows = num_windows
        self.encoder = TKANEncoder(input_dim, hidden_dim, output_dim, num_layers,
                                   grid_size, spline_order, dropout)
        self.regression_head = RegressionHead(output_dim, num_windows, dropout)
        self.action_gate = ActionGate(output_dim, num_windows, dropout)
        self.thresholds = nn.Parameter(torch.ones(num_windows) * 0.5, requires_grad=False)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        pred_delta = self.regression_head(h)
        action_prob = self.action_gate(h)
        return pred_delta, action_prob

    def predict(self, x: torch.Tensor, thresholds: torch.Tensor = None):
        pred_delta, action_prob = self.forward(x)
        if thresholds is None:
            thresholds = torch.ones(self.num_windows, device=x.device) * 0.5
        batch_size = x.size(0)
        predictions = torch.ones(batch_size, self.num_windows, dtype=torch.long, device=x.device)
        for w in range(self.num_windows):
            should_act = action_prob[:, w] > thresholds[w]
            predictions[should_act, w] = torch.where(
                pred_delta[should_act, w] > 0,
                torch.tensor(2, device=x.device),
                torch.tensor(0, device=x.device)
            )
        return predictions


class Predictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'best_model.pt')
        config_path = os.path.join(current_dir, 'config.json')
        thresholds_path = os.path.join(current_dir, 'best_thresholds.json')

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # config.json 只包含原始列，但我们需要计算衍生特征
        # 定义完整的特征列表（原始列 + 衍生列）
        self.raw_feature_cols = self.config['feature']
        self.derived_feature_cols = [
            'midprice', 'imbalance', 'cumspread',
            'ofi_raw', 'ofi_ewm', 'ofi_multilevel', 'ofi_velocity', 'ofi_volatility'
        ]
        self.feature_cols = self.raw_feature_cols + self.derived_feature_cols

        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                thresholds_data = json.load(f)
                self.thresholds = thresholds_data.get('thresholds', [0.5, 0.5, 0.5, 0.5, 0.5])
        else:
            self.thresholds = [0.5, 0.5, 0.5, 0.5, 0.5]

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        model_config = checkpoint.get('config', {})
        self.mean = checkpoint.get('mean', None)
        self.std = checkpoint.get('std', None)

        self.model = TKANOFIModel(
            input_dim=len(self.feature_cols),
            hidden_dim=model_config.get('hidden_dim', 256),
            output_dim=model_config.get('output_dim', 128),
            num_windows=5,
            num_layers=model_config.get('num_encoder_layers', 2),
            grid_size=model_config.get('grid_size', 8),
            spline_order=model_config.get('spline_order', 3),
            dropout=0.0
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        print(f"模型加载完成，设备: {self.device}")

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        """
        预测函数

        Args:
            x: List[pd.DataFrame]，长度为batch
               每个DataFrame为100个tick的数据

        Returns:
            List[List[int]]，长度为batch
            每个内层List长度为5，值为0/1/2
        """
        batch_size = len(x)

        features = []
        for df in x:
            # 关键：只读不写！先用 numpy 提取，再计算特征
            arr, cols = df_to_numpy(df)
            col_idx = {c: i for i, c in enumerate(cols)}

            # 计算衍生特征（纯 numpy，不碰 DataFrame）
            derived = compute_features_numpy(arr, cols)

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

            # 归一化
            if self.mean is not None and self.std is not None:
                feature_data = (feature_data - self.mean) / self.std

            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

            features.append(feature_data)

        features = np.array(features, dtype=np.float32)
        features_tensor = torch.from_numpy(features).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            thresholds_tensor = torch.tensor(self.thresholds, device=self.device)
            predictions = self.model.predict(features_tensor, thresholds_tensor)

        results = []
        for i in range(batch_size):
            sample_preds = predictions[i].cpu().tolist()
            results.append([int(p) for p in sample_preds])

        return results
