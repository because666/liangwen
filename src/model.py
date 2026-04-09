"""
T-KAN Pro 架构 - 深度收益导向模型

核心设计：
1. 深度 T-KAN 编码器（4层）+ 残差连接
2. 收益导向预测头（分类 + 收益预测双分支）
3. 数值稳定的 B-spline 实现
4. 参数量约 320 万

架构流程：
输入: (batch, 100, 54) -> 特征嵌入 -> T-KAN深度块x4 -> 时序聚合 -> 收益导向预测头 -> 输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class StableSplineLinear(nn.Module):
    """数值稳定的 B-spline 线性层
    
    核心改进：
    1. 较小的初始化（xavier gain=0.5, spline std=0.01）
    2. 输入 clamp 防止极端值
    3. 数值稳定的 B-spline 计算
    """
    
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
        """计算 B-spline 基函数（数值稳定版本）"""
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
        
        self.spline_linear = StableSplineLinear(
            features, features, grid_size, spline_order
        )
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
    """特征嵌入层
    
    将原始特征映射到隐藏空间
    """
    
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
    """时序聚合层
    
    使用轻量级注意力 + Mean Pool 聚合时序信息
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)
        x = x.mean(dim=1)
        return x


class ProfitGuidedHead(nn.Module):
    """收益导向预测头
    
    双分支设计：
    1. 分类分支：预测方向 (0/1/2)
    2. 收益分支：预测期望收益
    """
    
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
        
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 3) for _ in range(num_windows)
        ])
        
        self.return_predictors = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_windows)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (batch, num_windows, 3) 分类预测
            return_pred: (batch, num_windows) 收益预测
        """
        h = self.shared(x)
        
        logits = torch.stack([cls(h) for cls in self.classifiers], dim=1)
        return_pred = torch.stack([pred(h).squeeze(-1) for pred in self.return_predictors], dim=1)
        
        return logits, return_pred


class TKANPro(nn.Module):
    """T-KAN Pro 模型
    
    架构：
    1. 特征嵌入层: Linear(54, 256) + LayerNorm + GELU + Dropout
    2. T-KAN 深度块 × 4: 带残差连接
    3. 时序聚合层: MultiheadAttention + Mean Pool
    4. 收益导向预测头: 分类 + 收益预测双分支
    
    参数量约 320 万
    """
    
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            logits: (batch, num_windows, 3)
            return_pred: (batch, num_windows)
        """
        x = self.feature_embedding(x)
        
        for layer in self.tkan_layers:
            x = layer(x)
        
        x = self.temporal_agg(x)
        
        logits, return_pred = self.prediction_head(x)
        
        return logits, return_pred
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推理时只返回分类结果"""
        logits, _ = self.forward(x)
        return logits.argmax(dim=-1)


def create_model(input_dim: int, num_windows: int = 5, **kwargs) -> TKANPro:
    """创建 T-KAN Pro 模型"""
    defaults = {
        'hidden_dim': 256,
        'num_tkan_layers': 4,
        'grid_size': 8,
        'spline_order': 3,
        'num_heads': 8,
        'dropout': 0.15,
    }
    defaults.update(kwargs)
    return TKANPro(input_dim, num_windows=num_windows, **defaults)


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = create_model(input_dim=54, num_windows=5)
    print(f"模型参数量: {count_parameters(model):,}")
    
    x = torch.randn(2, 100, 54)
    logits, return_pred = model(x)
    print(f"输入形状: {x.shape}")
    print(f"分类输出形状: {logits.shape}")
    print(f"收益预测形状: {return_pred.shape}")
