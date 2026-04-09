"""
深度收益导向分层集成网络 - Predictor 推理封装

关键原则：
1. 绝不往 DataFrame 写入新列！
2. config.json 只包含原始列
3. 衍生特征在内部用 numpy 计算
4. 直接三分类输出，不再使用动作门
5. 数据清洗：处理 NaN、Inf 和异常值
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
        'mb_intst': mb, 'ma_intst': ma,
        'lb_intst': get_col('lb_intst'), 'la_intst': get_col('la_intst'),
        'cb_intst': get_col('cb_intst'), 'ca_intst': get_col('ca_intst'),
        'ofi_raw': ofi_raw, 'ofi_ewm': ofi_ewm,
        'ofi_velocity': ofi_velocity, 'ofi_volatility': ofi_volatility,
    }


def clean_features(features: np.ndarray, feature_cols: list) -> np.ndarray:
    """清洗特征数据：处理 NaN、Inf 和异常值"""
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


# ========== 模型定义（必须与训练时完全一致） ==========

class SplineLinear(nn.Module):
    """T-KAN 的 B-spline 线性层"""
    
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
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        
        nn.init.kaiming_uniform_(self.base_weight)
        nn.init.zeros_(self.spline_weight)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
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
        shape = x.shape
        x = x.view(-1, self.in_features)
        base = F.linear(x, self.base_weight)
        spline = torch.einsum('bij,oij->bo', self.b_splines(x), self.spline_weight)
        return (base + spline).view(*shape[:-1], self.out_features)


class TKANLayer(nn.Module):
    """T-KAN 层"""
    
    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 8, spline_order: int = 3, dropout: float = 0.1):
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


class ExpertA_TKAN(nn.Module):
    """专家A：T-KAN 时序专家"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64,
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
        return self.final_norm(x)


class ExpertB_LevelAttention(nn.Module):
    """专家B：档位注意力专家"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        attn_out, _ = self.attention(h, h, h)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        h = h.mean(dim=1)
        return self.final_norm(self.output_proj(h))


class ExpertC_PatchTransformer(nn.Module):
    """专家C：Patch Transformer专家"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64,
                 patch_size: int = 5, num_heads: int = 4, num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(input_dim * patch_size, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        num_patches = T // self.patch_size
        if num_patches == 0:
            num_patches = 1
        x_trimmed = x[:, :num_patches * self.patch_size, :]
        x_patches = x_trimmed.reshape(B, num_patches, -1)
        h = self.patch_proj(x_patches)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.transformer(h)
        h = h[:, 0]
        return self.final_norm(self.output_proj(h))


class ExpertD_CrossAssetOFI(nn.Module):
    """专家D：跨资产OFI专家"""

    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)
        h = h.mean(dim=1)
        return self.final_norm(self.output_proj(h))


class DynamicGatingFusion(nn.Module):
    """动态门控融合"""

    def __init__(self, num_experts: int, expert_dim: int, market_state_dim: int,
                 hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(market_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.expert_proj = nn.ModuleList([
            nn.Linear(expert_dim, expert_dim) for _ in range(num_experts)
        ])

    def forward(self, expert_outputs: list, market_state: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate_network(market_state)
        gate_weights = F.softmax(gate_logits, dim=-1)
        projected = [proj(out) for proj, out in zip(self.expert_proj, expert_outputs)]
        stacked = torch.stack(projected, dim=1)
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)
        return fused


class ProfitGuidedHead(nn.Module):
    """收益导向预测头"""

    def __init__(self, input_dim: int, num_windows: int = 5, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 3) for _ in range(num_windows)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        logits = torch.stack([cls(h) for cls in self.classifiers], dim=1)
        return logits


class MarketStateEncoder(nn.Module):
    """市场状态编码器"""

    def __init__(self, input_dim: int, output_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 3, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = torch.cat([
            x.std(dim=1),
            x.max(dim=1).values - x.min(dim=1).values,
            x.mean(dim=1),
        ], dim=-1)
        return self.encoder(stats)


class DeepProfitNet(nn.Module):
    """深度收益导向分层集成网络"""

    def __init__(self, input_dim: int, num_windows: int = 5,
                 expert_a_dim: int = 64, expert_b_dim: int = 64,
                 expert_c_dim: int = 64, expert_d_dim: int = 32,
                 tkan_hidden: int = 128, tkan_layers: int = 2,
                 grid_size: int = 8, spline_order: int = 3,
                 num_heads: int = 4, patch_size: int = 5,
                 head_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_windows = num_windows

        self.expert_a = ExpertA_TKAN(
            input_dim, tkan_hidden, expert_a_dim, tkan_layers,
            grid_size, spline_order, dropout
        )
        self.expert_b = ExpertB_LevelAttention(
            input_dim, 64, expert_b_dim, num_heads, dropout
        )
        self.expert_c = ExpertC_PatchTransformer(
            input_dim, 64, expert_c_dim, patch_size, num_heads, 1, dropout
        )
        self.expert_d = ExpertD_CrossAssetOFI(
            input_dim, 32, expert_d_dim, dropout
        )

        total_expert_dim = expert_a_dim + expert_b_dim + expert_c_dim + expert_d_dim
        max_expert_dim = max(expert_a_dim, expert_b_dim, expert_c_dim, expert_d_dim)

        self.market_encoder = MarketStateEncoder(input_dim, 32)
        self.gating = DynamicGatingFusion(
            4, max_expert_dim, 32, 64, dropout
        )

        self.fusion_proj = nn.Linear(total_expert_dim, total_expert_dim)
        self.fusion_norm = nn.LayerNorm(total_expert_dim)

        self.prediction_head = ProfitGuidedHead(
            total_expert_dim, num_windows, head_hidden, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_a = self.expert_a(x)
        h_b = self.expert_b(x)
        h_c = self.expert_c(x)
        h_d = self.expert_d(x)

        market_state = self.market_encoder(x)

        expert_outputs = [h_a, h_b, h_c, h_d]
        expert_dims = [h.size(-1) for h in expert_outputs]
        max_dim = max(expert_dims)
        padded = []
        for h in expert_outputs:
            if h.size(-1) < max_dim:
                pad_size = max_dim - h.size(-1)
                h = F.pad(h, (0, pad_size))
            padded.append(h)

        fused = self.gating(padded, market_state)
        fused = torch.cat(expert_outputs, dim=-1)
        fused = self.fusion_norm(self.fusion_proj(fused))

        logits = self.prediction_head(fused)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)


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

        self.model = DeepProfitNet(
            input_dim=len(self.feature_cols),
            num_windows=5,
            dropout=0.2,
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        print(f"模型加载完成，设备: {self.device}")

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        """
        Args:
            x: List[pd.DataFrame]，长度为batch，每个DataFrame为100个tick的数据
        Returns:
            List[List[int]]，长度为batch，每个内层List长度为5，值为0/1/2
        """
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
