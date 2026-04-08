"""
高频量化预测模型 - 优化版

模型架构：
1. TCN (Temporal Convolutional Network) - 捕捉时序特征
2. Transformer Encoder - 捕捉长距离依赖
3. Multi-Head Attention - 关注重要时间步
4. Multi-Task Learning - 同时预测5个时间窗口

特点：
- 处理标签不平衡（Focal Loss + 类别权重）
- 残差连接和层归一化
- Dropout正则化
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class PositionalEncoding(nn.Module):
    """
    位置编码 - 为Transformer添加位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalBlock(nn.Module):
    """
    TCN时序块 - 因果卷积 + 残差连接
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out = out + residual
        return self.relu(out)


class TCN(nn.Module):
    """
    时序卷积网络
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        layers = []
        for i in range(len(hidden_channels)):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_channels[i - 1]
            out_ch = hidden_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=dilation, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        return self.transformer_encoder(x)


class MultiHeadAttentionPooling(nn.Module):
    """
    多头注意力池化 - 聚合时序特征
    """
    
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        attn_output, _ = self.attention(query, x, x)
        return attn_output.squeeze(1)


class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: List[float] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class HFTPredictor(nn.Module):
    """
    高频交易预测模型
    
    架构：
    1. 特征嵌入层
    2. TCN提取局部时序特征
    3. Transformer捕捉全局依赖
    4. 注意力池化聚合
    5. 多任务预测头
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes_per_head: List[int] = [3, 3, 3, 3, 3],
        hidden_dim: int = 128,
        tcn_channels: List[int] = [64, 128, 128],
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes_per_head = num_classes_per_head
        
        self.feature_embed = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.tcn = TCN(hidden_dim, tcn_channels, kernel_size=3, dropout=dropout)
        
        tcn_output_dim = tcn_channels[-1]
        
        self.transformer = TransformerEncoder(
            d_model=tcn_output_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        self.attention_pool = MultiHeadAttentionPooling(tcn_output_dim, num_heads=transformer_heads)
        
        self.shared_fc = nn.Sequential(
            nn.Linear(tcn_output_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, num_classes)
            ) for num_classes in num_classes_per_head
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            x: (batch, seq_len, num_features) 或 (batch, 1, seq_len, num_features)
        
        Returns:
            每个预测头的输出
        """
        if x.dim() == 4:
            x = x.squeeze(1)
        
        batch_size, seq_len, num_features = x.shape
        
        x = self.feature_embed(x)
        x = x.transpose(1, 2)
        
        x = self.tcn(x)
        
        x = x.transpose(1, 2)
        
        x = self.transformer(x)
        
        x = self.attention_pool(x)
        
        x = self.shared_fc(x)
        
        outputs = tuple(head(x) for head in self.heads)
        
        return outputs


class HFTPredictorLite(nn.Module):
    """
    轻量版高频交易预测模型
    
    适合快速训练和推理
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes_per_head: List[int] = [3, 3, 3, 3, 3],
        hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.feature_embed = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim * 2,
            num_layers=2, batch_first=True,
            dropout=dropout, bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.heads = nn.ModuleList([
            nn.Linear(128, num_classes) for num_classes in num_classes_per_head
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if x.dim() == 4:
            x = x.squeeze(1)
        
        x = self.feature_embed(x)
        
        lstm_out, _ = self.lstm(x)
        
        attn_weights = self.attention(lstm_out)
        x = (lstm_out * attn_weights).sum(dim=1)
        
        x = self.shared_fc(x)
        
        outputs = tuple(head(x) for head in self.heads)
        
        return outputs


def compute_class_weights(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """
    计算类别权重（用于处理类别不平衡）
    
    Args:
        labels: 标签张量
        num_classes: 类别数量
    
    Returns:
        类别权重张量
    """
    class_counts = torch.bincount(labels.long(), minlength=num_classes).float()
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-8)
    weights = weights / weights.sum()
    return weights


def create_model(
    model_type: str = "full",
    num_features: int = 200,
    num_classes_per_head: List[int] = None,
    **kwargs
) -> nn.Module:
    """
    模型工厂函数
    
    Args:
        model_type: "full" (TCN+Transformer) 或 "lite" (LSTM)
        num_features: 特征数量
        num_classes_per_head: 每个预测头的类别数
        **kwargs: 其他模型参数
    
    Returns:
        模型实例
    """
    if num_classes_per_head is None:
        num_classes_per_head = [3, 3, 3, 3, 3]
    
    if model_type == "full":
        return HFTPredictor(
            num_features=num_features,
            num_classes_per_head=num_classes_per_head,
            **kwargs
        )
    elif model_type == "lite":
        return HFTPredictorLite(
            num_features=num_features,
            num_classes_per_head=num_classes_per_head,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    batch_size = 4
    seq_len = 100
    num_features = 200
    
    x = torch.randn(batch_size, seq_len, num_features)
    
    print("测试完整模型...")
    model_full = HFTPredictor(num_features=num_features)
    outputs_full = model_full(x)
    print(f"输入形状: {x.shape}")
    for i, out in enumerate(outputs_full):
        print(f"输出{i+1}形状: {out.shape}")
    
    print("\n测试轻量模型...")
    model_lite = HFTPredictorLite(num_features=num_features)
    outputs_lite = model_lite(x)
    for i, out in enumerate(outputs_lite):
        print(f"输出{i+1}形状: {out.shape}")
    
    print("\n测试Focal Loss...")
    focal_loss = FocalLoss(alpha=[0.3, 0.4, 0.3], gamma=2.0)
    targets = torch.randint(0, 3, (batch_size,))
    loss = focal_loss(outputs_full[0], targets)
    print(f"Focal Loss: {loss.item():.4f}")
