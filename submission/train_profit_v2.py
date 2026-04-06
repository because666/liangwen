"""
高频量化预测模型 - 收益导向版 (Profit-Optimized)

解决"准确率高但收益低"的问题：
1. 直接以交易收益为优化目标
2. 降低"不变"类权重，鼓励预测涨跌
3. 训练时监控收益而非准确率
4. 更保守的出手策略（只在高置信度时交易）

GPU适配：NVIDIA A10 (24GB显存)
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from collections import Counter


BASE_FEATURE_COLS = [
    "open", "high", "low", "close",
    "volume_delta", "amount_delta",
    "bid1", "bsize1", "bid2", "bsize2", "bid3", "bsize3", "bid4", "bsize4", "bid5", "bsize5",
    "bid6", "bsize6", "bid7", "bsize7", "bid8", "bsize8", "bid9", "bsize9", "bid10", "bsize10",
    "ask1", "asize1", "ask2", "asize2", "ask3", "asize3", "ask4", "asize4", "ask5", "asize5",
    "ask6", "asize6", "ask7", "asize7", "ask8", "asize8", "ask9", "asize9", "ask10", "asize10",
    "avgbid", "avgask", "totalbsize", "totalasize",
    "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
    "lb_ind", "la_ind", "mb_ind", "ma_ind", "cb_ind", "ca_ind",
    "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc",
    "midprice1", "midprice2", "midprice3", "midprice4", "midprice5",
    "midprice6", "midprice7", "midprice8", "midprice9", "midprice10",
    "spread1", "spread2", "spread3", "spread4", "spread5",
    "spread6", "spread7", "spread8", "spread9", "spread10",
    "bid_diff1", "bid_diff2", "bid_diff3", "bid_diff4", "bid_diff5",
    "bid_diff6", "bid_diff7", "bid_diff8", "bid_diff9", "bid_diff10",
    "ask_diff1", "ask_diff2", "ask_diff3", "ask_diff4", "ask_diff5",
    "ask_diff6", "ask_diff7", "ask_diff8", "ask_diff9", "ask_diff10",
    "bid_mean", "ask_mean", "bsize_mean", "asize_mean",
    "cumspread", "imbalance",
    "bid_rate1", "bid_rate2", "bid_rate3", "bid_rate4", "bid_rate5",
    "bid_rate6", "bid_rate7", "bid_rate8", "bid_rate9", "bid_rate10",
    "ask_rate1", "ask_rate2", "ask_rate3", "ask_rate4", "ask_rate5",
    "ask_rate6", "ask_rate7", "ask_rate8", "ask_rate9", "ask_rate10",
    "bsize_rate1", "bsize_rate2", "bsize_rate3", "bsize_rate4", "bsize_rate5",
    "bsize_rate6", "bsize_rate7", "bsize_rate8", "bsize_rate9", "bsize_rate10",
    "asize_rate1", "asize_rate2", "asize_rate3", "asize_rate4", "asize_rate5",
    "asize_rate6", "asize_rate7", "asize_rate8", "asize_rate9", "asize_rate10",
    "midprice"
]

LABEL_COLS = ["label_5", "label_10", "label_20", "label_40", "label_60"]
WINDOW_SIZES = [5, 10, 20, 40, 60]
FEE_RATE = 0.0002


def compute_ofi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算订单流不平衡(OFI)特征
    
    包含：
    1. 基础OFI特征（25个）
    2. 订单簿不平衡衰减特征（EWI，学术文献验证有效）
    """
    df = df.copy()
    
    df['net_market_flow'] = df['mb_intst'] - df['ma_intst']
    df['net_limit_flow'] = df['lb_intst'] - df['la_intst']
    df['net_cancel_flow'] = df['cb_intst'] - df['ca_intst']
    
    for window in [5, 10, 20]:
        df[f'cum_ofi_{window}'] = df['net_market_flow'].rolling(window=window, min_periods=1).sum()
    
    for window in [10, 20]:
        df[f'ofi_volatility_{window}'] = df['net_market_flow'].rolling(window=window, min_periods=1).std()
    
    df['ofi_abs_sum_20'] = df['net_market_flow'].abs().rolling(window=20, min_periods=1).sum()
    
    total_order_flow = df['mb_intst'] + df['ma_intst'] + 1e-8
    df['buy_pressure'] = df['mb_intst'] / total_order_flow
    df['sell_pressure'] = df['ma_intst'] / total_order_flow
    
    df['total_imbalance'] = (df['mb_intst'] + df['lb_intst']) - (df['ma_intst'] + df['la_intst'])
    df['ofi_momentum_5'] = df['net_market_flow'].rolling(window=5, min_periods=1).mean()
    df['ofi_momentum_10'] = df['net_market_flow'].rolling(window=10, min_periods=1).mean()
    df['ofi_acceleration'] = df['net_market_flow'].diff()
    df['ofi_limit_ratio'] = (df['net_limit_flow'] - df['net_market_flow']) / (df['net_market_flow'].abs() + 1e-8)
    df['ofi_cancel_ratio'] = df['net_cancel_flow'] / (df['net_market_flow'].abs() + df['net_limit_flow'].abs() + 1e-8)
    df['ofi_pressure_imbalance'] = df['buy_pressure'] - df['sell_pressure']
    df['ofi_skewness_10'] = df['net_market_flow'].rolling(window=10, min_periods=1).skew()
    df['ofi_kurtosis_20'] = df['net_market_flow'].rolling(window=20, min_periods=1).kurt()
    df['ofi_max_10'] = df['net_market_flow'].rolling(window=10, min_periods=1).max()
    df['ofi_min_10'] = df['net_market_flow'].rolling(window=10, min_periods=1).min()
    df['ofi_range_10'] = df['ofi_max_10'] - df['ofi_min_10']
    df['ofi_sign_consistency_5'] = (df['net_market_flow'] > 0).rolling(window=5, min_periods=1).mean()
    df['ofi_sign_consistency_10'] = (df['net_market_flow'] > 0).rolling(window=10, min_periods=1).mean()
    
    # 订单簿不平衡衰减特征（EWI - Exponential Weighted Imbalance）
    # 学术文献验证：订单流影响随时间指数衰减，半衰期5tick（15秒）效果最佳
    total_bsize = df['totalbsize'] if 'totalbsize' in df.columns else df[[f'bsize{i}' for i in range(1, 11)]].sum(axis=1)
    total_asize = df['totalasize'] if 'totalasize' in df.columns else df[[f'asize{i}' for i in range(1, 11)]].sum(axis=1)
    
    # 原始订单簿不平衡
    imb = (total_bsize - total_asize) / (total_bsize + total_asize + 1e-8)
    
    # 指数加权移动平均（半衰期5tick）
    # alpha = 1 - exp(-ln(2)/halflife) ≈ 0.129 for halflife=5
    df['ewi_5'] = imb.ewm(alpha=0.129, adjust=False).mean()
    
    # 不同半衰期的EWI（捕捉不同时间尺度的压力）
    df['ewi_10'] = imb.ewm(alpha=0.067, adjust=False).mean()  # 半衰期10tick
    df['ewi_20'] = imb.ewm(alpha=0.034, adjust=False).mean()  # 半衰期20tick
    
    # EWI变化率（压力变化趋势）
    df['ewi_5_diff'] = df['ewi_5'].diff()
    df['ewi_10_diff'] = df['ewi_10'].diff()
    
    # EWI与当前不平衡的差异（压力回归信号）
    df['ewi_5_deviation'] = imb - df['ewi_5']
    df['ewi_10_deviation'] = imb - df['ewi_10']
    
    # EWI符号一致性（压力方向稳定性）
    df['ewi_sign_5'] = (df['ewi_5'] > 0).rolling(window=5, min_periods=1).mean()
    df['ewi_sign_10'] = (df['ewi_10'] > 0).rolling(window=10, min_periods=1).mean()
    
    return df


OFI_FEATURE_COLS = [
    'net_market_flow', 'net_limit_flow', 'net_cancel_flow',
    'cum_ofi_5', 'cum_ofi_10', 'cum_ofi_20',
    'ofi_volatility_10', 'ofi_volatility_20', 'ofi_abs_sum_20',
    'buy_pressure', 'sell_pressure', 'total_imbalance',
    'ofi_momentum_5', 'ofi_momentum_10', 'ofi_acceleration',
    'ofi_limit_ratio', 'ofi_cancel_ratio', 'ofi_pressure_imbalance',
    'ofi_skewness_10', 'ofi_kurtosis_20',
    'ofi_max_10', 'ofi_min_10', 'ofi_range_10',
    'ofi_sign_consistency_5', 'ofi_sign_consistency_10',
    'ewi_5', 'ewi_10', 'ewi_20',
    'ewi_5_diff', 'ewi_10_diff',
    'ewi_5_deviation', 'ewi_10_deviation',
    'ewi_sign_5', 'ewi_sign_10'
]


class PositionalEncoding(nn.Module):
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
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        branch_channels = out_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        return self.dropout(torch.cat([b1, b3, b5, b7], dim=1))


class DilatedMultiScaleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        branch_channels = out_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return self.dropout(torch.cat([b1, b2, b3, b4], dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return F.relu(out + residual)


class MultiScaleTCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, num_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        
        self.multi_scale_blocks = nn.ModuleList([
            MultiScaleConvBlock(hidden_channels, hidden_channels, dropout)
            for _ in range(num_blocks)
        ])
        
        self.dilated_blocks = nn.ModuleList([
            DilatedMultiScaleBlock(hidden_channels, hidden_channels, dropout)
            for _ in range(num_blocks)
        ])
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, dropout)
            for _ in range(num_blocks)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for i in range(len(self.multi_scale_blocks)):
            ms_out = self.multi_scale_blocks[i](x)
            dilated_out = self.dilated_blocks[i](x)
            
            combined = ms_out + dilated_out
            combined = combined.transpose(1, 2)
            combined = self.norms[i](combined)
            combined = combined.transpose(1, 2)
            
            x = x + combined
            x = self.residual_blocks[i](x)
        
        return x


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        batch, channels, seq_len = x.shape
        squeeze = x.mean(dim=2)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        excitation = excitation.unsqueeze(2)
        return x * excitation


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channel_attention = SEBlock(channels, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.channel_attention(x)
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial_attention(spatial_input)
        return x * spatial_attention


class HFTModel(nn.Module):
    """
    高频量化预测模型
    
    架构：
    1. 特征嵌入
    2. 多尺度TCN
    3. CBAM注意力
    4. Transformer编码器
    5. 多任务预测头
    """
    
    def __init__(
        self, 
        num_features: int, 
        hidden_dim: int = 128, 
        num_blocks: int = 3,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.feature_embed = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.multi_scale_tcn = MultiScaleTCN(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout
        )
        
        self.cbam = CBAM(hidden_dim)
        
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.attention_pool = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, 3)
            ) for _ in range(5)
        ])
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        
        x = self.feature_embed(x)
        
        x = x.transpose(1, 2)
        x = self.multi_scale_tcn(x)
        x = self.cbam(x)
        x = x.transpose(1, 2)
        
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        x, _ = self.attention_pool(query, x, x)
        x = x.squeeze(1)
        
        x = self.shared_fc(x)
        
        outputs = tuple(head(x) for head in self.heads)
        
        return outputs


class ProfitLoss(nn.Module):
    """
    收益导向损失函数
    
    核心思想：
    1. 直接优化交易收益，而非分类准确率
    2. 对"不变"类给予极低的权重（甚至负权重）
    3. 对正确的涨跌预测给予高奖励
    4. 对错误的涨跌预测给予高惩罚
    
    关键改进：
    - 降低类别1（不变）的权重 → 减少预测不变的概率
    - 增加涨跌类的权重 → 鼓励更多交易机会
    - 使用收益矩阵 → 让损失与实际收益挂钩
    """
    
    def __init__(
        self, 
        fee_rate: float = FEE_RATE,
        profit_weight: float = 2.0,
        penalty_weight: float = 3.0,
        hold_weight: float = 0.05
    ):
        super().__init__()
        self.fee_rate = fee_rate
        
        self.profit_weight = profit_weight
        self.penalty_weight = penalty_weight
        self.hold_weight = hold_weight
        
        weight_matrix = torch.tensor([
            [profit_weight, hold_weight, -penalty_weight],   # 实际下跌时：预测跌=奖励，预测不变=小惩罚，预测涨=大惩罚
            [-penalty_weight/2, hold_weight, -penalty_weight/2], # 实际不变时：预测方向=小惩罚
            [-penalty_weight, hold_weight, profit_weight]     # 实际上涨时：预测跌=大惩罚，预测不变=小惩罚，预测涨=奖励
        ], dtype=torch.float32)
        self.register_buffer('weight_matrix', weight_matrix)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 模型输出 (batch, 3)
            targets: 真实标签 (batch,)
        """
        batch_size = logits.size(0)
        device = logits.device
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        
        weight_matrix = self.weight_matrix.to(device)
        profit_weights = weight_matrix[targets, preds]
        
        profit_loss = -profit_weights * ce_loss
        
        class_weights = torch.tensor([2.0, 0.1, 2.0], device=device)
        weighted_ce = class_weights[targets] * ce_loss
        
        total_loss = 0.6 * profit_loss + 0.4 * weighted_ce
        
        return total_loss.mean()


class AsymmetricFocalLoss(nn.Module):
    """
    不对称Focal Loss
    
    针对"准确率高但收益低"问题的特殊设计：
    - 大幅降低类别1（不变）的权重
    - 提高类别0和2（跌和涨）的权重
    - 使用更强的gamma值关注困难样本
    """
    
    def __init__(
        self, 
        alpha: List[float] = None, 
        gamma: float = 3.0,
        label_smoothing: float = 0.02
    ):
        super().__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            # 默认：大幅降低"不变"类权重
            self.alpha = torch.tensor([2.5, 0.15, 2.5], dtype=torch.float32)
        
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / inputs.size(-1)
            ce_loss = -(targets_one_hot * F.log_softmax(inputs, dim=-1)).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        
        alpha_t = self.alpha.to(inputs.device)[targets]
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class LOBDataset(Dataset):
    """订单簿数据集"""
    
    def __init__(
        self, 
        data_dir: str, 
        feature_cols: List[str], 
        label_cols: List[str],
        seq_len: int = 100, 
        stride: int = 3, 
        max_files: int = 100, 
        augment: bool = False
    ):
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        
        self.features = []
        self.labels = []
        
        files = sorted(glob.glob(os.path.join(data_dir, "snapshot_sym*.parquet")))[:max_files]
        print(f"加载 {len(files)} 个文件...")
        
        for f in tqdm(files, desc="加载数据"):
            try:
                df = pd.read_parquet(f)
                df = df.fillna(0).replace([np.inf, -np.inf], 0)
                df = compute_ofi_features(df)
                
                for col in feature_cols:
                    if col not in df.columns:
                        df[col] = 0
                
                feature_data = df[feature_cols].values.astype(np.float32)
                label_data = df[label_cols].values.astype(np.int64)
                
                for i in range(0, len(df) - seq_len, stride):
                    self.features.append(feature_data[i:i+seq_len])
                    self.labels.append(label_data[i+seq_len-1])
                    
            except Exception as e:
                continue
        
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"总样本数: {len(self.features)}")
        print(f"标签分布: {Counter(self.labels.flatten())}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        X = self.features[idx].copy()
        y = self.labels[idx].copy()
        
        if self.augment:
            if np.random.random() < 0.3:
                X = X * np.random.uniform(0.98, 1.02)
            
            if np.random.random() < 0.2:
                noise = np.random.normal(0, 0.005, X.shape).astype(np.float32)
                X = X + noise
            
            if np.random.random() < 0.1:
                mask = np.random.random(X.shape[1]) > 0.1
                X[:, ~mask] = 0
        
        return torch.from_numpy(X), torch.from_numpy(y)


def train_one_epoch(model, train_loader, criteria, optimizer, device, scaler=None, clip_norm=1.0):
    model.train()
    total_loss = 0
    correct = [0] * 5
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="训练", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        if scaler:
            with autocast():
                outputs = model(inputs)
                loss = sum(criteria[i](outputs[i], targets[:, i]) for i in range(5)) / 5
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
        else:
            outputs = model(inputs)
            loss = sum(criteria[i](outputs[i], targets[:, i]) for i in range(5)) / 5
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            else:
                optimizer.zero_grad()
        
        if not torch.isnan(loss) and not torch.isinf(loss):
            total_loss += loss.item()
        total += inputs.size(0)
        for i in range(5):
            correct[i] += (outputs[i].argmax(1) == targets[:, i]).sum().item()
    
    return total_loss / len(train_loader) if total_loss > 0 else 0, [c / total for c in correct]


def validate(model, val_loader, criteria, device):
    model.eval()
    total_loss = 0
    correct = [0] * 5
    total = 0
    
    all_probs = [[] for _ in range(5)]
    all_targets = [[] for _ in range(5)]
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="验证", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = sum(criteria[i](outputs[i], targets[:, i]) for i in range(5)) / 5
            
            total_loss += loss.item()
            total += inputs.size(0)
            
            for i in range(5):
                correct[i] += (outputs[i].argmax(1) == targets[:, i]).sum().item()
                probs = F.softmax(outputs[i], dim=1)
                all_probs[i].append(probs.cpu())
                all_targets[i].append(targets[:, i].cpu())
    
    all_probs = [torch.cat(p, dim=0) for p in all_probs]
    all_targets = [torch.cat(t, dim=0) for t in all_targets]
    
    return total_loss / len(val_loader), [c / total for c in correct], all_probs, all_targets


def simulate_trading_profit(probs, targets, threshold=0.5):
    """
    模拟交易并计算收益
    
    这是关键函数！直接计算如果按当前策略交易会获得多少收益
    """
    max_probs, preds = probs.max(dim=1)
    
    confident_mask = (preds != 1) & (max_probs > threshold)
    
    trade_count = confident_mask.sum().item()
    
    if trade_count == 0:
        return {
            'trade_count': 0,
            'total_profit': 0,
            'avg_profit': 0,
            'win_rate': 0,
            'accuracy': 0,
            'hold_rate': 1.0
        }
    
    confident_preds = preds[confident_mask]
    confident_targets = targets[confident_mask]
    
    correct_up = (confident_preds == 2) & (confident_targets == 2)
    correct_down = (confident_preds == 0) & (confident_targets == 0)
    wrong_up = (confident_preds == 2) & (confident_targets == 0)
    wrong_down = (confident_preds == 0) & (confident_targets == 2)
    
    profits = torch.zeros(trade_count)
    
    # 正确预测上涨：赚取价格变化 - 手续费
    profits[correct_up] = 0.001 - FEE_RATE  # 模拟正收益
    
    # 正确预测下跌：赚取价格变化 - 手续费  
    profits[correct_down] = 0.001 - FEE_RATE
    
    # 错误预测上涨（实际下跌）：亏损价格变化 - 手续费
    profits[wrong_up] = -0.002 - FEE_RATE
    
    # 错误预测下跌（实际上涨）：亏损价格变化 - 手续费
    profits[wrong_down] = -0.002 - FEE_RATE
    
    total_profit = profits.sum().item()
    avg_profit = total_profit / trade_count
    win_rate = (profits > 0).float().mean().item()
    accuracy = (confident_preds == confident_targets).float().mean().item()
    hold_rate = 1 - trade_count / len(preds)
    
    return {
        'trade_count': trade_count,
        'total_profit': total_profit,
        'avg_profit': avg_profit,
        'win_rate': win_rate,
        'accuracy': accuracy,
        'hold_rate': hold_rate
    }


def optimize_threshold_for_profit(probs, targets):
    """
    为最大化收益而优化阈值
    
    与之前不同：这里的目标是最大化 avg_profit × sqrt(trade_count)
    而不是准确率
    """
    best_threshold = 0.55  # 默认更高阈值，更保守
    best_score = -float('inf')
    best_metrics = None
    
    for threshold in np.arange(0.45, 0.85, 0.01):  # 从更高的阈值开始搜索
        metrics = simulate_trading_profit(probs, targets, threshold)
        
        if metrics['trade_count'] >= 10:  # 至少有10次交易才有意义
            # 综合评分：平均收益 * 出手次数的平方根（平衡收益和频率）
            score = metrics['avg_profit'] * (metrics['trade_count'] ** 0.25)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics
    
    if best_metrics is None:
        # 如果没有找到合适的阈值，使用默认值
        best_threshold = 0.65
        best_metrics = {
            'trade_count': 0,
            'total_profit': 0,
            'avg_profit': 0,
            'win_rate': 0,
            'accuracy': 0,
            'hold_rate': 1.0
        }
    
    return best_threshold, best_metrics


def main():
    parser = argparse.ArgumentParser(description="高频量化预测模型 - 收益导向版")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output_profit')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--max_files', type=int, default=1200)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--use_tta', action='store_true', default=True)
    parser.add_argument('--tta_rounds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("高频量化预测模型 - 收益导向版 (Profit-Optimized)")
    print("=" * 70)
    print(f"设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\n⚠️ 核心改进:")
    print("   1. ProfitLoss - 直接优化交易收益")
    print("   2. 不对称类别权重 - 降低'不变'类权重至0.15")
    print("   3. 更高阈值起点 (0.55+) - 更保守的出手策略")
    print("   4. 收益指标监控 - 训练时看的是收益不是准确率")
    print("=" * 70)
    
    feature_cols = list(BASE_FEATURE_COLS) + OFI_FEATURE_COLS
    print(f"\n特征数量: {len(feature_cols)} (基础: {len(BASE_FEATURE_COLS)}, OFI: {len(OFI_FEATURE_COLS)})")
    
    print("\n加载训练数据...")
    train_dataset = LOBDataset(
        args.data_dir, feature_cols, LABEL_COLS,
        seq_len=args.seq_len, stride=args.stride,
        max_files=int(args.max_files * 0.85),
        augment=args.augment
    )
    
    print("\n加载验证数据...")
    val_dataset = LOBDataset(
        args.data_dir, feature_cols, LABEL_COLS,
        seq_len=args.seq_len, stride=args.stride * 2,
        max_files=int(args.max_files * 0.15),
        augment=False
    )
    
    num_workers = min(8, os.cpu_count() or 4)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    
    print("\n创建模型...")
    model = HFTModel(
        num_features=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    
    # 使用不对称Focal Loss - 降低"不变"类权重
    criteria = [
        AsymmetricFocalLoss(
            alpha=[2.5, 0.15, 2.5],  # 大幅降低"不变"类权重
            gamma=3.0,
            label_smoothing=0.02
        ) for _ in range(5)
    ]
    
    # 同时添加一个ProfitLoss用于后期微调
    profit_criteria = [
        ProfitLoss(fee_rate=FEE_RATE) for _ in range(5)
    ]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if args.use_amp else None
    
    best_avg_profit = -float('inf')
    best_state = None
    best_thresholds = {str(i): 0.6 for i in range(5)}
    best_epoch = 0
    patience_counter = 0
    patience = 10  # 更激进的早停策略（基于收益，连续3个epoch无提升则停止）
    
    experiment_log = []
    
    print("\n开始训练...")
    print("=" * 70)
    print("⚠️ 早停策略：基于验证集单次平均收益，连续10个epoch无提升则停止")
    print("=" * 70)
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")
        
        # 前30个epoch使用AsymmetricFocalLoss学习基本模式
        if epoch < 30:
            current_criteria = criteria
            loss_name = "AsymmetricFocal"
        else:
            # 后期切换到ProfitLoss进行收益优化
            current_criteria = profit_criteria
            loss_name = "Profit"
        
        train_loss, train_accs = train_one_epoch(
            model, train_loader, current_criteria, optimizer, device, scaler, args.gradient_clip
        )
        val_loss, val_accs, val_probs, val_targets = validate(model, val_loader, current_criteria, device)
        
        scheduler.step()
        
        print(f"[{loss_name}] 训练Loss: {train_loss:.4f}, 平均准确率: {np.mean(train_accs):.4f}")
        print(f"验证Loss: {val_loss:.4f}, 平均准确率: {np.mean(val_accs):.4f}")
        
        window_results = []
        for i in range(5):
            threshold, metrics = optimize_threshold_for_profit(val_probs[i], val_targets[i])
            window_results.append({
                'threshold': threshold,
                'metrics': metrics,
                'acc': val_accs[i]
            })
            
            print(f"\n  窗口{i} ({LABEL_COLS[i]}):")
            print(f"    准确率={val_accs[i]:.4f}")
            print(f"    最优阈值={threshold:.2f}")
            print(f"    出手率={(1-metrics['hold_rate']):.4f}")
            print(f"    出手次数={metrics['trade_count']}")
            print(f"    单次收益={metrics['avg_profit']:.6f}")
            print(f"    胜率={metrics['win_rate']:.4f}")
            print(f"    置信准确率={metrics['accuracy']:.4f}")
        
        # 用平均单次收益作为主要评估指标
        avg_profits = [r['metrics']['avg_profit'] for r in window_results if r['metrics']['trade_count'] > 0]
        if avg_profits:
            current_avg_profit = np.mean(avg_profits)
        else:
            current_avg_profit = -999
        
        print(f"\n  ★ 平均单次收益: {current_avg_profit:.6f}")
        
        # 记录实验日志
        experiment_log.append({
            'epoch': epoch + 1,
            'avg_profit': current_avg_profit,
            'loss': val_loss,
            'accuracy': np.mean(val_accs),
            'loss_type': loss_name
        })
        
        if current_avg_profit > best_avg_profit:
            best_avg_profit = current_avg_profit
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            
            for i in range(5):
                best_thresholds[str(i)] = window_results[i]['threshold']
            
            patience_counter = 0
            print(f"  ✓ 保存最佳模型 (Epoch {best_epoch}, 单次收益: {best_avg_profit:.6f})")
        else:
            patience_counter += 1
            print(f"  ⚠️ 收益未提升 ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n{'='*70}")
                print(f"早停触发：连续 {patience} 个epoch收益未提升")
                print(f"最佳Epoch: {best_epoch}")
                print(f"最佳单次收益: {best_avg_profit:.6f}")
                print(f"{'='*70}")
                break
    
    if best_state is not None:
        torch.save({
            'model_state': best_state,
            'num_features': len(feature_cols),
            'feature_cols': feature_cols,
            'hidden_dim': args.hidden_dim,
            'best_avg_profit': best_avg_profit,
            'best_epoch': best_epoch,
            'thresholds': best_thresholds,
            'use_tta': args.use_tta,
            'tta_rounds': args.tta_rounds if args.use_tta else 0
        }, os.path.join(args.output_dir, 'best_model.pt'))
    
    # 保存实验日志
    log_df = pd.DataFrame(experiment_log)
    log_df.to_csv(os.path.join(args.output_dir, 'experiment_log.csv'), index=False)
    
    config = {
        "python_version": "3.10",
        "batch": args.batch_size,
        "feature": feature_cols,
        "label": LABEL_COLS,
        "thresholds": best_thresholds,
        "use_tta": args.use_tta,
        "tta_rounds": args.tta_rounds if args.use_tta else 0,
        "best_epoch": best_epoch,
        "best_avg_profit": best_avg_profit
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"{'='*70}")
    print(f"最佳Epoch: {best_epoch}")
    print(f"最佳单次收益: {best_avg_profit:.6f}")
    print(f"最优阈值: {best_thresholds}")
    print(f"实验日志已保存: {os.path.join(args.output_dir, 'experiment_log.csv')}")
    print(f"模型已保存到: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
