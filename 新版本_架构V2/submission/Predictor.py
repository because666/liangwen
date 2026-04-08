"""
T-KAN+OFI 高频方向预测 - Predictor 推理封装

严格按照评测平台规范实现：
1. 处理 Polars DataFrame
2. 实时计算 OFI 特征
3. 使用绝对路径加载模型
"""

from typing import List
import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_to_pandas(df):
    """将 Polars DataFrame 转换为纯 Pandas DataFrame"""
    if isinstance(df, pd.DataFrame):
        return df.copy()
    
    if hasattr(df, 'to_pandas'):
        try:
            result = df.to_pandas()
            return pd.DataFrame({col: np.array(result[col].values, dtype=float) for col in result.columns})
        except Exception:
            pass
    
    if hasattr(df, 'columns') and hasattr(df, '__len__'):
        n_rows = len(df)
        data = {}
        for col in df.columns:
            col_data = df[col]
            if hasattr(col_data, 'to_numpy'):
                data[col] = np.array(col_data.to_numpy(), dtype=float)
            elif hasattr(col_data, 'to_list'):
                data[col] = np.array(col_data.to_list(), dtype=float)
            else:
                data[col] = np.zeros(n_rows)
        return pd.DataFrame(data)
    
    return pd.DataFrame(df)


def compute_ofi_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算 OFI 特征"""
    df = df.copy()
    n = len(df)
    
    mb_intst = df['mb_intst'].values if 'mb_intst' in df.columns else np.zeros(n)
    ma_intst = df['ma_intst'].values if 'ma_intst' in df.columns else np.zeros(n)
    
    ofi_raw = mb_intst - ma_intst
    df['ofi_raw'] = ofi_raw
    
    ofi_ewm = np.zeros(n)
    alpha = 0.1
    for i in range(n):
        if i == 0:
            ofi_ewm[i] = ofi_raw[i]
        else:
            ofi_ewm[i] = (1 - alpha) * ofi_ewm[i-1] + alpha * ofi_raw[i]
    df['ofi_ewm'] = ofi_ewm
    
    weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    ofi_multilevel = np.zeros(n)
    for k, w in enumerate(weights):
        mb_col = 'mb_intst'
        ma_col = 'ma_intst'
        if mb_col in df.columns and ma_col in df.columns:
            ofi_multilevel += w * (df[mb_col].values - df[ma_col].values)
    df['ofi_multilevel'] = ofi_multilevel
    
    ofi_velocity = np.zeros(n)
    ofi_velocity[1:] = ofi_raw[1:] - ofi_raw[:-1]
    df['ofi_velocity'] = ofi_velocity
    
    ofi_volatility = np.zeros(n)
    window = 10
    for i in range(window, n):
        ofi_volatility[i] = np.std(ofi_raw[i-window:i])
    df['ofi_volatility'] = ofi_volatility
    
    return df


class SplineLinear(nn.Module):
    """B样条线性层"""
    
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
    """T-KAN 单层"""
    
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
    """T-KAN 编码器"""
    
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
    """回归头"""
    
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
    """出手门"""
    
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
    """T-KAN+OFI 模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128,
                 num_windows: int = 5, num_layers: int = 2, grid_size: int = 8,
                 spline_order: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.num_windows = num_windows
        
        self.encoder = TKANEncoder(input_dim, hidden_dim, output_dim, num_layers, 
                                   grid_size, spline_order, dropout)
        self.regression_head = RegressionHead(output_dim, num_windows, dropout)
        self.action_gate = ActionGate(output_dim, num_windows, dropout)
        
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
    """预测类"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'best_model.pt')
        config_path = os.path.join(current_dir, 'config.json')
        thresholds_path = os.path.join(current_dir, 'best_thresholds.json')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.feature_cols = self.config['feature']
        
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
            df = convert_to_pandas(df)
            
            df = compute_ofi_features(df)
            
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            
            feature_data = df[self.feature_cols].values.astype(np.float32)
            
            if self.mean is not None and self.std is not None:
                feature_data = (feature_data - self.mean) / self.std
            
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            features.append(feature_data)
        
        features = np.array(features, dtype=np.float32)
        features_tensor = torch.from_numpy(features).to(self.device)
        
        with torch.no_grad():
            thresholds_tensor = torch.tensor(self.thresholds, device=self.device)
            predictions = self.model.predict(features_tensor, thresholds_tensor)
        
        results = []
        for i in range(batch_size):
            sample_preds = predictions[i].cpu().tolist()
            results.append([int(p) for p in sample_preds])
        
        return results
