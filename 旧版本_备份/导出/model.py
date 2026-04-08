"""
高频量化预测模型 - 模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class TemporalBlock(nn.Module):
    """扩张因果卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.conv1(x)[:, :, :x.size(2)]
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.dropout(out)
        
        out = self.se(out)
        
        return F.gelu(out + residual)


class LargeTCN(nn.Module):
    """大规模TCN"""
    
    def __init__(self, in_channels: int, hidden_channels: int = 384, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (i % 4)
            self.blocks.append(TemporalBlock(hidden_channels, hidden_channels, 3, dilation, dropout))
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dim_feedforward: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class MultiScalePooling(nn.Module):
    """多尺度池化"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1),
        ])
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        pooled = [pool(x).squeeze(-1) for pool in self.pools]
        out = torch.cat(pooled, dim=-1)
        return self.fc(out)


class HFTModelLarge(nn.Module):
    """大规模高频交易模型"""
    
    def __init__(self, num_features: int, hidden_dim: int = 384, num_tcn_layers: int = 6,
                 num_transformer_layers: int = 4, num_heads: int = 8, dropout: float = 0.12):
        super().__init__()
        
        self.input_embed = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.tcn = LargeTCN(hidden_dim, hidden_dim, num_tcn_layers, dropout)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        self.multi_scale_pool = MultiScalePooling(hidden_dim)
        
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2)
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout / 3),
                nn.Linear(64, 3)
            ) for _ in range(5)
        ])
    
    def forward(self, x):
        x = self.input_embed(x)
        
        x = x.transpose(1, 2)
        x = self.tcn(x)
        
        x = x.transpose(1, 2)
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = x.transpose(1, 2)
        x = self.multi_scale_pool(x)
        
        x = self.shared_fc(x)
        
        return tuple(head(x) for head in self.heads)
