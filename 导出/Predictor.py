"""
高频量化预测类

按照竞赛规范实现 Predictor 类
"""

from typing import List
import pandas as pd
import numpy as np
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


class Predictor:
    """预测类"""
    
    def __init__(self):
        import json
        import os
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.json')
        model_path = os.path.join(current_dir, 'best_model.pt')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.feature_cols = checkpoint.get('feature_cols', self.config['feature'])
        self.hidden_dim = checkpoint.get('hidden_dim', 384)
        self.thresholds = checkpoint.get('thresholds', {})
        
        self.model = HFTModelLarge(
            num_features=len(self.feature_cols),
            hidden_dim=self.hidden_dim,
            num_tcn_layers=6,
            num_transformer_layers=4,
            num_heads=8,
            dropout=0.12
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
    
    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        """
        预测函数
        
        输入: List[pd.DataFrame]，长度为batch
              每个DataFrame为100个tick的数据，列名为feature
        输出: List[List[int]]，长度为batch
              每个内层List长度为label个数，值为0/1/2
        """
        batch_size = len(x)
        
        features = []
        for df in x:
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0
            
            feature_data = df[self.feature_cols].values.astype(np.float32)
            
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            features.append(feature_data)
        
        features = np.array(features, dtype=np.float32)
        
        features_tensor = torch.from_numpy(features).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
            
            predictions = []
            for i, output in enumerate(outputs):
                probs = F.softmax(output, dim=1)
                max_probs, preds = probs.max(dim=1)
                
                threshold = self.thresholds.get(str(i), 0.55)
                
                for j in range(batch_size):
                    if max_probs[j].item() < threshold:
                        preds[j] = 1
                
                predictions.append(preds.cpu().numpy())
        
        results = []
        for i in range(batch_size):
            sample_preds = [int(predictions[j][i]) for j in range(5)]
            results.append(sample_preds)
        
        return results
