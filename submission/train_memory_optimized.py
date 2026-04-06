"""
高频量化预测模型 - 内存优化版 (Memory-Optimized)

解决内存不足问题：
1. 惰性加载：只在需要时加载数据
2. 流式处理：不一次性加载所有文件
3. 内存映射：使用numpy memmap
4. 智能缓存：只缓存当前需要的batch

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
from typing import List, Dict, Tuple, Optional
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume_delta', 'amount_delta',
    'bid1', 'bsize1', 'bid2', 'bsize2', 'bid3', 'bsize3', 'bid4', 'bsize4', 'bid5', 'bsize5',
    'bid6', 'bsize6', 'bid7', 'bsize7', 'bid8', 'bsize8', 'bid9', 'bsize9', 'bid10', 'bsize10',
    'ask1', 'asize1', 'ask2', 'asize2', 'ask3', 'asize3', 'ask4', 'asize4', 'ask5', 'asize5',
    'ask6', 'asize6', 'ask7', 'asize7', 'ask8', 'asize8', 'ask9', 'asize9', 'ask10', 'asize10',
    'avgbid', 'avgask', 'totalbsize', 'totalasize',
    'lb_intst', 'la_intst', 'mb_intst', 'ma_intst', 'cb_intst', 'ca_intst',
    'lb_ind', 'la_ind', 'mb_ind', 'ma_ind', 'cb_ind', 'ca_ind',
    'lb_acc', 'la_acc', 'mb_acc', 'ma_acc', 'cb_acc', 'ca_acc',
    'midprice1', 'midprice2', 'midprice3', 'midprice4', 'midprice5',
    'midprice6', 'midprice7', 'midprice8', 'midprice9', 'midprice10',
    'spread1', 'spread2', 'spread3', 'spread4', 'spread5',
    'spread6', 'spread7', 'spread8', 'spread9', 'spread10',
    'bid_diff1', 'bid_diff2', 'bid_diff3', 'bid_diff4', 'bid_diff5',
    'bid_diff6', 'bid_diff7', 'bid_diff8', 'bid_diff9', 'bid_diff10',
    'ask_diff1', 'ask_diff2', 'ask_diff3', 'ask_diff4', 'ask_diff5',
    'ask_diff6', 'ask_diff7', 'ask_diff8', 'ask_diff9', 'ask_diff10',
    'bid_mean', 'ask_mean', 'bsize_mean', 'asize_mean', 'cumspread', 'imbalance',
    'bid_rate1', 'bid_rate2', 'bid_rate3', 'bid_rate4', 'bid_rate5',
    'bid_rate6', 'bid_rate7', 'bid_rate8', 'bid_rate9', 'bid_rate10',
    'ask_rate1', 'ask_rate2', 'ask_rate3', 'ask_rate4', 'ask_rate5',
    'ask_rate6', 'ask_rate7', 'ask_rate8', 'ask_rate9', 'ask_rate10',
    'bsize_rate1', 'bsize_rate2', 'bsize_rate3', 'bsize_rate4', 'bsize_rate5',
    'bsize_rate6', 'bsize_rate7', 'bsize_rate8', 'bsize_rate9', 'bsize_rate10',
    'asize_rate1', 'asize_rate2', 'asize_rate3', 'asize_rate4', 'asize_rate5',
    'asize_rate6', 'asize_rate7', 'asize_rate8', 'asize_rate9', 'asize_rate10',
    'midprice'
]

LABEL_COLS = ["label_5", "label_10", "label_20", "label_40", "label_60"]
FEE_RATE = 0.0002


def compute_ofi_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算订单流不平衡(OFI)特征"""
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
    
    # EWI特征
    total_bsize = df['totalbsize'] if 'totalbsize' in df.columns else df[[f'bsize{i}' for i in range(1, 11)]].sum(axis=1)
    total_asize = df['totalasize'] if 'totalasize' in df.columns else df[[f'asize{i}' for i in range(1, 11)]].sum(axis=1)
    
    imb = (total_bsize - total_asize) / (total_bsize + total_asize + 1e-8)
    
    df['ewi_5'] = imb.ewm(alpha=0.129, adjust=False).mean()
    df['ewi_10'] = imb.ewm(alpha=0.067, adjust=False).mean()
    df['ewi_20'] = imb.ewm(alpha=0.034, adjust=False).mean()
    
    df['ewi_5_diff'] = df['ewi_5'].diff()
    df['ewi_10_diff'] = df['ewi_10'].diff()
    
    df['ewi_5_deviation'] = imb - df['ewi_5']
    df['ewi_10_deviation'] = imb - df['ewi_10']
    
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


class LazyLOBDataset(Dataset):
    """
    惰性加载的订单簿数据集
    
    核心优化：
    1. 不一次性加载所有文件到内存
    2. 只记录文件路径和索引
    3. 在__getitem__时才加载对应的数据
    4. 使用LRU缓存最近访问的数据
    """
    
    def __init__(
        self, 
        data_dir: str, 
        feature_cols: List[str], 
        label_cols: List[str],
        seq_len: int = 100, 
        stride: int = 5,  # 增加stride减少样本数
        max_files: int = 300,  # 减少文件数
        augment: bool = False,
        cache_size: int = 50  # 缓存最近访问的50个文件
    ):
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        
        # 获取文件列表
        files = sorted(glob.glob(os.path.join(data_dir, "snapshot_sym*.parquet")))[:max_files]
        print(f"找到 {len(files)} 个文件，开始建立索引...")
        
        # 建立索引：记录每个样本对应的文件和起始位置
        self.sample_index = []  # [(file_idx, start_pos), ...]
        self.files = files
        
        # 预扫描文件，建立索引（不加载实际数据）
        for file_idx, f in enumerate(tqdm(files, desc="建立索引")):
            try:
                # 只读取行数，不加载全部数据
                df = pd.read_parquet(f, columns=['close'])  # 只读一列获取长度
                num_rows = len(df)
                
                # 记录该文件中的所有样本索引
                for start_pos in range(0, num_rows - seq_len, stride):
                    self.sample_index.append((file_idx, start_pos))
                    
            except Exception as e:
                continue
        
        print(f"总样本数: {len(self.sample_index)}")
        
        # LRU缓存
        self.cache = {}
        self.cache_size = cache_size
        self.cache_order = []
    
    def _load_file(self, file_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """加载文件（带缓存）"""
        if file_idx in self.cache:
            return self.cache[file_idx]
        
        # 如果缓存满了，删除最旧的
        if len(self.cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        # 加载文件
        f = self.files[file_idx]
        df = pd.read_parquet(f)
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        df = compute_ofi_features(df)
        
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        feature_data = df[self.feature_cols].values.astype(np.float32)
        label_data = df[self.label_cols].values.astype(np.int64)
        
        # 存入缓存
        self.cache[file_idx] = (feature_data, label_data)
        self.cache_order.append(file_idx)
        
        return feature_data, label_data
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        file_idx, start_pos = self.sample_index[idx]
        
        # 惰性加载
        feature_data, label_data = self._load_file(file_idx)
        
        # 提取样本
        X = feature_data[start_pos:start_pos+self.seq_len].copy()
        y = label_data[start_pos+self.seq_len-1].copy()
        
        # 数据增强
        if self.augment:
            if np.random.random() < 0.3:
                X = X * np.random.uniform(0.98, 1.02)
            
            if np.random.random() < 0.2:
                noise = np.random.normal(0, 0.005, X.shape).astype(np.float32)
                X = X + noise
        
        return torch.from_numpy(X), torch.from_numpy(y)


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


class AsymmetricFocalLoss(nn.Module):
    """不对称Focal Loss - 降低'不变'类权重"""
    
    def __init__(self, alpha=[2.5, 0.15, 2.5], gamma=3.0, label_smoothing=0.02):
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        
        alpha = self.alpha.to(inputs.device)
        at = alpha[targets]
        
        focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class ProfitLoss(nn.Module):
    """收益导向损失函数"""
    
    def __init__(self, fee_rate=FEE_RATE, profit_weight=2.0, penalty_weight=3.0, hold_weight=0.05):
        super().__init__()
        self.fee_rate = fee_rate
        self.profit_weight = profit_weight
        self.penalty_weight = penalty_weight
        self.hold_weight = hold_weight
        
        self.weight_matrix = torch.tensor([
            [profit_weight, hold_weight, -penalty_weight],
            [-penalty_weight/2, hold_weight, -penalty_weight/2],
            [-penalty_weight, hold_weight, profit_weight]
        ])
    
    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        
        weight_matrix = self.weight_matrix.to(inputs.device)
        target_weights = weight_matrix[targets]
        
        expected_profit = (probs * target_weights).sum(dim=1)
        
        loss = -expected_profit.mean()
        
        return loss


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
        
        total_loss += loss.item()
        
        for i in range(5):
            _, preds = outputs[i].max(1)
            correct[i] += (preds == targets[:, i]).sum().item()
        
        total += targets.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracies = [c / total for c in correct]
    
    return avg_loss, accuracies


def validate(model, val_loader, device, scaler=None):
    model.eval()
    all_probs = [[] for _ in range(5)]
    all_targets = [[] for _ in range(5)]
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="验证", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if scaler:
                with autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            for i in range(5):
                probs = F.softmax(outputs[i], dim=1)
                all_probs[i].append(probs.cpu())
                all_targets[i].append(targets[:, i].cpu())
    
    probs = [torch.cat(p, dim=0) for p in all_probs]
    targets = [torch.cat(t, dim=0) for t in all_targets]
    
    return probs, targets


def simulate_trading_profit(probs, targets, threshold=0.5):
    max_probs, preds = probs.max(dim=1)
    
    trade_mask = (preds != 1) & (max_probs > threshold)
    trade_preds = preds[trade_mask]
    trade_targets = targets[trade_mask]
    trade_probs = max_probs[trade_mask]
    
    trade_count = len(trade_preds)
    hold_count = len(preds) - trade_count
    
    if trade_count == 0:
        return {
            'trade_count': 0,
            'total_profit': 0,
            'avg_profit': 0,
            'win_rate': 0,
            'accuracy': (preds == targets).float().mean().item(),
            'hold_rate': 1.0
        }
    
    profits = []
    for i in range(len(trade_preds)):
        if trade_preds[i] == 2 and trade_targets[i] == 2:
            profits.append(0.001 - FEE_RATE)
        elif trade_preds[i] == 0 and trade_targets[i] == 0:
            profits.append(0.001 - FEE_RATE)
        elif trade_preds[i] == 2 and trade_targets[i] == 0:
            profits.append(-0.002 - FEE_RATE)
        elif trade_preds[i] == 0 and trade_targets[i] == 2:
            profits.append(-0.002 - FEE_RATE)
        else:
            profits.append(-FEE_RATE)
    
    total_profit = sum(profits)
    avg_profit = total_profit / trade_count
    win_rate = sum(1 for p in profits if p > 0) / len(profits)
    
    return {
        'trade_count': trade_count,
        'total_profit': total_profit,
        'avg_profit': avg_profit,
        'win_rate': win_rate,
        'accuracy': (preds == targets).float().mean().item(),
        'hold_rate': hold_count / len(preds)
    }


def optimize_threshold_for_profit(probs, targets):
    best_threshold = 0.55
    best_score = -float('inf')
    best_metrics = None
    
    for threshold in np.arange(0.45, 0.85, 0.01):
        metrics = simulate_trading_profit(probs, targets, threshold)
        
        if metrics['trade_count'] >= 10:
            score = metrics['avg_profit'] * (metrics['trade_count'] ** 0.25)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics
    
    if best_metrics is None:
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
    parser = argparse.ArgumentParser(description="高频量化预测模型 - 内存优化版")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output_profit')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--max_files', type=int, default=300)  # 减少到300
    parser.add_argument('--stride', type=int, default=5)  # 增加到5
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--use_tta', action='store_true', default=True)
    parser.add_argument('--tta_rounds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)  # 新增：数据加载线程数
    parser.add_argument('--cache_size', type=int, default=50)  # 新增：缓存大小
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("高频量化预测模型 - 内存优化版 (Memory-Optimized)")
    print("=" * 70)
    print(f"设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\n⚠️ 内存优化策略:")
    print(f"   1. 惰性加载 - 只在需要时加载数据")
    print(f"   2. 减少文件数 - 从1200降至{args.max_files}")
    print(f"   3. 增加步长 - 从3增至{args.stride}")
    print(f"   4. 智能缓存 - 缓存最近{args.cache_size}个文件")
    print(f"   5. 多线程加载 - {args.num_workers}个worker")
    print("=" * 70)
    
    feature_cols = list(BASE_FEATURE_COLS) + OFI_FEATURE_COLS
    print(f"\n特征数量: {len(feature_cols)}")
    
    print("\n建立数据索引...")
    full_dataset = LazyLOBDataset(
        args.data_dir, feature_cols, LABEL_COLS,
        seq_len=args.seq_len, stride=args.stride,
        max_files=args.max_files, augment=args.augment,
        cache_size=args.cache_size
    )
    
    # 划分训练集和验证集
    total_size = len(full_dataset)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, total_size))
    
    print(f"\n训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    model = HFTModel(
        num_features=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout
    ).to(device)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    criteria = [AsymmetricFocalLoss() for _ in range(5)]
    profit_criteria = ProfitLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if args.use_amp else None
    
    best_avg_profit = -float('inf')
    best_state = None
    best_thresholds = {str(i): 0.6 for i in range(5)}
    best_epoch = 0
    patience_counter = 0
    patience = 10
    
    print("\n开始训练...")
    print("=" * 70)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if epoch < 30:
            current_criteria = criteria
            loss_name = "AsymmetricFocal"
        else:
            current_criteria = [profit_criteria] * 5
            loss_name = "Profit"
        
        train_loss, train_accs = train_one_epoch(
            model, train_loader, current_criteria, optimizer, device, scaler, args.gradient_clip
        )
        
        print(f"[{loss_name}] 训练Loss: {train_loss:.4f}, 平均准确率: {np.mean(train_accs):.4f}")
        
        probs, targets = validate(model, val_loader, device, scaler)
        
        window_results = []
        for i in range(5):
            threshold, metrics = optimize_threshold_for_profit(probs[i], targets[i])
            window_results.append({
                'threshold': threshold,
                'metrics': metrics
            })
            
            print(f"\n  窗口{i} (label_{WINDOW_SIZES[i]}):")
            print(f"    准确率={metrics['accuracy']:.4f}")
            print(f"    最优阈值={threshold:.2f}")
            print(f"    出手率={1-metrics['hold_rate']:.4f}")
            print(f"    出手次数={metrics['trade_count']}")
            print(f"    单次收益={metrics['avg_profit']:.6f}")
            print(f"    胜率={metrics['win_rate']:.4f}")
        
        avg_profits = [r['metrics']['avg_profit'] for r in window_results if r['metrics']['trade_count'] > 0]
        current_avg_profit = np.mean(avg_profits) if avg_profits else 0
        
        print(f"\n  ★ 平均单次收益: {current_avg_profit:.6f}")
        
        if current_avg_profit > best_avg_profit:
            best_avg_profit = current_avg_profit
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            
            for i in range(5):
                best_thresholds[str(i)] = window_results[i]['threshold']
            
            patience_counter = 0
            print(f"  ✓ 保存最佳模型 (Epoch {best_epoch})")
        else:
            patience_counter += 1
            print(f"  ⚠️ 收益未提升 ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n早停: 连续 {patience} 个epoch收益未提升")
                break
        
        scheduler.step()
    
    if best_state is not None:
        torch.save({
            'model_state': best_state,
            'num_features': len(feature_cols),
            'feature_cols': feature_cols,
            'hidden_dim': args.hidden_dim,
            'best_avg_profit': best_avg_profit,
            'best_epoch': best_epoch,
            'thresholds': best_thresholds,
        }, os.path.join(args.output_dir, 'best_model.pt'))
    
    config = {
        "python_version": "3.10",
        "batch": args.batch_size,
        "feature": feature_cols,
        "label": LABEL_COLS,
        "thresholds": best_thresholds,
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"最佳Epoch: {best_epoch}")
    print(f"最佳单次收益: {best_avg_profit:.6f}")
    print(f"模型已保存到: {args.output_dir}")
    print(f"{'='*70}")


WINDOW_SIZES = [5, 10, 20, 40, 60]


if __name__ == '__main__':
    main()
