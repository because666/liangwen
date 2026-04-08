"""
T-KAN (Temporal Kolmogorov-Arnold Network) 编码器

基于B样条的时序特征提取器，用于学习订单簿的非线性动力学
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SplineLinear(nn.Module):
    """B样条线性层"""
    
    def __init__(self, in_features: int, out_features: int, grid_size: int = 8, 
                 spline_order: int = 3, scale_noise: float = 0.1, 
                 scale_base: float = 1.0, scale_spline: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        grid = torch.linspace(-1, 1, grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)
        
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features) * scale_base
        )
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        
        with torch.no_grad():
            noise = (torch.rand(self.out_features, self.in_features, self.grid_size + self.spline_order) - 0.5) * self.scale_noise
            self.spline_weight.data.copy_(noise + self.scale_spline * torch.ones_like(self.spline_weight))
    
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """计算B样条基函数"""
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        grid = self.grid
        x = x.unsqueeze(-1)
        
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)]) * bases[..., :-1]
                + (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:-k]) * bases[..., 1:]
            )
        
        assert bases.size(-1) == self.grid_size + self.spline_order
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
    """T-KAN单层"""
    
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
    """T-KAN编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128,
                 num_layers: int = 2, grid_size: int = 8, spline_order: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        layers = []
        dims = [hidden_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            layers.append(TKANLayer(
                dims[i], dims[i + 1], grid_size, spline_order, dropout
            ))
        
        self.layers = nn.ModuleList(layers)
        
        self.final_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, output_dim)
        """
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = x.mean(dim=1)
        
        x = self.final_norm(x)
        
        return x


class RKANCell(nn.Module):
    """递归KAN单元"""
    
    def __init__(self, input_dim: int, hidden_dim: int, grid_size: int = 8,
                 spline_order: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.spline_ih = SplineLinear(input_dim + hidden_dim, hidden_dim, grid_size, spline_order)
        self.spline_hh = SplineLinear(hidden_dim, hidden_dim, grid_size, spline_order)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
            h: (batch, hidden_dim) 隐状态
        Returns:
            h_new: (batch, hidden_dim)
        """
        batch_size = x.size(0)
        
        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        combined = torch.cat([x, h], dim=-1)
        
        i = torch.sigmoid(self.spline_ih(combined))
        f = torch.sigmoid(self.spline_hh(h))
        
        h_new = i * torch.tanh(self.spline_ih(combined)) + f * h
        h_new = self.layer_norm(h_new)
        h_new = self.dropout(h_new)
        
        return h_new


class RKANEncoder(nn.Module):
    """递归KAN编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128,
                 num_layers: int = 2, grid_size: int = 8, spline_order: int = 3,
                 dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.cells = nn.ModuleList([
            RKANCell(hidden_dim if i > 0 else hidden_dim, hidden_dim, grid_size, spline_order, dropout)
            for i in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.final_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        x = self.input_proj(x)
        
        h = [None] * self.num_layers
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx] = cell(x_t, h[layer_idx])
                x_t = h[layer_idx]
        
        final_h = h[-1]
        
        if self.bidirectional:
            h_rev = [None] * self.num_layers
            for t in range(seq_len - 1, -1, -1):
                x_t = x[:, t, :]
                for layer_idx, cell in enumerate(self.cells):
                    h_rev[layer_idx] = cell(x_t, h_rev[layer_idx])
                    x_t = h_rev[layer_idx]
            final_h = torch.cat([final_h, h_rev[-1]], dim=-1)
        
        output = self.output_proj(final_h)
        output = self.final_norm(output)
        
        return output


class HybridEncoder(nn.Module):
    """混合编码器：TCN + T-KAN"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128,
                 num_tcn_layers: int = 2, num_kan_layers: int = 2,
                 grid_size: int = 8, spline_order: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        tcn_layers = []
        for i in range(num_tcn_layers):
            dilation = 2 ** i
            padding = dilation
            tcn_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, 
                                        padding=padding, dilation=dilation))
            tcn_layers.append(nn.LayerNorm(hidden_dim))
            tcn_layers.append(nn.GELU())
            tcn_layers.append(nn.Dropout(dropout))
        self.tcn = nn.Sequential(*tcn_layers)
        
        kan_layers = []
        for i in range(num_kan_layers):
            kan_layers.append(TKANLayer(hidden_dim, hidden_dim, grid_size, spline_order, dropout))
        self.kan_layers = nn.ModuleList(kan_layers)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, output_dim)
        """
        x = self.input_proj(x)
        
        x_tcn = x.transpose(1, 2)
        for i in range(0, len(self.tcn), 4):
            residual = x_tcn
            x_tcn = self.tcn[i](x_tcn)
            x_tcn = x_tcn[:, :, :residual.size(2)]
            x_tcn = self.tcn[i+1](x_tcn.transpose(1, 2)).transpose(1, 2)
            x_tcn = self.tcn[i+2](x_tcn)
            x_tcn = self.tcn[i+3](x_tcn)
            x_tcn = x_tcn + residual
        x = x_tcn.transpose(1, 2)
        
        for kan_layer in self.kan_layers:
            x = kan_layer(x)
        
        x = x.mean(dim=1)
        
        x = self.output_proj(x)
        x = self.final_norm(x)
        
        return x
