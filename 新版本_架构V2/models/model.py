"""
T-KAN+OFI 高频方向预测模型

包含：
- 回归头：预测5个窗口的价格变化率
- 出手门：决定是否交易
- 收益感知聚合层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tkan_encoder import TKANEncoder, RKANEncoder, HybridEncoder


class RegressionHead(nn.Module):
    """回归头：预测价格变化率"""
    
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
        """
        Args:
            x: (batch, hidden_dim)
        Returns:
            pred_delta: (batch, num_windows) 价格变化率预测
        """
        return self.head(x)


class ActionGate(nn.Module):
    """出手门：决定是否交易"""
    
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
        """
        Args:
            x: (batch, hidden_dim)
        Returns:
            action_prob: (batch, num_windows) 出手概率
        """
        return self.gate(x)


class TKANOFIModel(nn.Module):
    """T-KAN+OFI 高频方向预测模型"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_windows: int = 5,
                 encoder_type: str = 'tkan',
                 num_encoder_layers: int = 2,
                 grid_size: int = 8,
                 spline_order: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_windows = num_windows
        self.hidden_dim = hidden_dim
        
        if encoder_type == 'tkan':
            self.encoder = TKANEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_encoder_layers,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout
            )
        elif encoder_type == 'rkan':
            self.encoder = RKANEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_encoder_layers,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout
            )
        else:
            self.encoder = HybridEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout
            )
        
        self.regression_head = RegressionHead(output_dim, num_windows, dropout)
        self.action_gate = ActionGate(output_dim, num_windows, dropout)
        
        self.thresholds = nn.Parameter(torch.ones(num_windows) * 0.5, requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            pred_delta: (batch, num_windows) 价格变化率预测
            action_prob: (batch, num_windows) 出手概率
        """
        h = self.encoder(x)
        pred_delta = self.regression_head(h)
        action_prob = self.action_gate(h)
        
        return pred_delta, action_prob
    
    def predict(self, x: torch.Tensor, thresholds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        推理时的预测
        
        Args:
            x: (batch, seq_len, input_dim)
            thresholds: (num_windows,) 各窗口的出手阈值
            
        Returns:
            predictions: (batch, num_windows) 预测标签 (0=下跌, 1=不变, 2=上涨)
        """
        pred_delta, action_prob = self.forward(x)
        
        if thresholds is None:
            thresholds = self.thresholds
            
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
    
    def set_thresholds(self, thresholds: List[float]):
        """设置出手阈值"""
        with torch.no_grad():
            self.thresholds.copy_(torch.tensor(thresholds, dtype=torch.float32))


class TKANOFIModelWithAttention(nn.Module):
    """带注意力机制的T-KAN+OFI模型"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_windows: int = 5,
                 encoder_type: str = 'tkan',
                 num_encoder_layers: int = 2,
                 num_attention_heads: int = 4,
                 grid_size: int = 8,
                 spline_order: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_windows = num_windows
        self.hidden_dim = hidden_dim
        
        if encoder_type == 'tkan':
            self.encoder = TKANEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_encoder_layers,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout
            )
        elif encoder_type == 'rkan':
            self.encoder = RKANEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_encoder_layers,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout
            )
        else:
            self.encoder = HybridEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                dropout=dropout
            )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        self.regression_head = RegressionHead(output_dim, num_windows, dropout)
        self.action_gate = ActionGate(output_dim, num_windows, dropout)
        
        self.thresholds = nn.Parameter(torch.ones(num_windows) * 0.5, requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            pred_delta: (batch, num_windows) 价格变化率预测
            action_prob: (batch, num_windows) 出手概率
        """
        h = self.encoder.encoder(x) if hasattr(self.encoder, 'encoder') else self.encoder.input_proj(x)
        
        h_attn, _ = self.attention(h, h, h)
        h = self.attention_norm(h + h_attn)
        
        h = h.mean(dim=1)
        
        h = self.output_proj(h)
        h = self.output_norm(h)
        
        pred_delta = self.regression_head(h)
        action_prob = self.action_gate(h)
        
        return pred_delta, action_prob
    
    def predict(self, x: torch.Tensor, thresholds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """推理时的预测"""
        pred_delta, action_prob = self.forward(x)
        
        if thresholds is None:
            thresholds = self.thresholds
            
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


def create_model(config: Dict) -> nn.Module:
    """根据配置创建模型"""
    model = TKANOFIModel(
        input_dim=config.get('input_dim', 150),
        hidden_dim=config.get('hidden_dim', 256),
        output_dim=config.get('output_dim', 128),
        num_windows=config.get('num_windows', 5),
        encoder_type=config.get('encoder_type', 'tkan'),
        num_encoder_layers=config.get('num_encoder_layers', 2),
        grid_size=config.get('grid_size', 8),
        spline_order=config.get('spline_order', 3),
        dropout=config.get('dropout', 0.1)
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    config = {
        'input_dim': 150,
        'hidden_dim': 256,
        'output_dim': 128,
        'num_windows': 5,
        'encoder_type': 'tkan',
        'num_encoder_layers': 2,
        'grid_size': 8,
        'spline_order': 3,
        'dropout': 0.1
    }
    
    model = create_model(config)
    print(f"模型参数量: {count_parameters(model):,}")
    
    x = torch.randn(32, 100, 150)
    pred_delta, action_prob = model(x)
    print(f"pred_delta shape: {pred_delta.shape}")
    print(f"action_prob shape: {action_prob.shape}")
    
    predictions = model.predict(x)
    print(f"predictions shape: {predictions.shape}")
