"""
高频量化预测模型 - 高性能版 (High-Performance)

解决：
1. GPU利用率低的问题（I/O瓶颈）
2. NaN问题（数据预处理）
3. 训练速度慢的问题

核心优化：
1. 预处理+缓存：一次性预处理并保存为numpy格式
2. 内存映射加载：使用np.memmap避免OOM
3. 更好的NaN处理
4. 多进程数据加载
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
from typing import List, Dict, Tuple
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


def compute_ofi_features_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    安全计算OFI特征（避免NaN）
    
    关键改进：
    1. 使用min_periods确保rolling不会产生NaN
    2. 填充初始值为0
    3. 更严格的inf/nan处理
    """
    df = df.copy()
    
    # 基础订单流特征
    df['net_market_flow'] = df['mb_intst'].fillna(0) - df['ma_intst'].fillna(0)
    df['net_limit_flow'] = df['lb_intst'].fillna(0) - df['la_intst'].fillna(0)
    df['net_cancel_flow'] = df['cb_intst'].fillna(0) - df['ca_intst'].fillna(0)
    
    # 累积特征（使用min_periods=1避免NaN）
    for window in [5, 10, 20]:
        df[f'cum_ofi_{window}'] = df['net_market_flow'].rolling(window=window, min_periods=1).sum().fillna(0)
    
    # 波动性特征
    for window in [10, 20]:
        df[f'ofi_volatility_{window}'] = df['net_market_flow'].rolling(window=window, min_periods=1).std().fillna(0)
    
    df['ofi_abs_sum_20'] = df['net_market_flow'].abs().rolling(window=20, min_periods=1).sum().fillna(0)
    
    # 压力特征
    total_order_flow = (df['mb_intst'].fillna(0) + df['ma_intst'].fillna(0)).clip(lower=1e-8)
    df['buy_pressure'] = (df['mb_intst'].fillna(0) / total_order_flow).clip(-1, 1)
    df['sell_pressure'] = (df['ma_intst'].fillna(0) / total_order_flow).clip(-1, 1)
    
    df['total_imbalance'] = (df['mb_intst'].fillna(0) + df['lb_intst'].fillna(0)) - (df['ma_intst'].fillna(0) + df['la_intst'].fillna(0))
    
    # 动量特征
    df['ofi_momentum_5'] = df['net_market_flow'].rolling(window=5, min_periods=1).mean().fillna(0)
    df['ofi_momentum_10'] = df['net_market_flow'].rolling(window=10, min_periods=1).mean().fillna(0)
    df['ofi_acceleration'] = df['net_market_flow'].diff().fillna(0)
    
    # 比率特征（添加分母保护）
    denom = df['net_market_flow'].abs().clip(lower=1e-8)
    df['ofi_limit_ratio'] = ((df['net_limit_flow'] - df['net_market_flow']) / denom).clip(-10, 10)
    df['ofi_cancel_ratio'] = (df['net_cancel_flow'] / (df['net_market_flow'].abs() + df['net_limit_flow'].abs() + 1e-8)).clip(-10, 10)
    df['ofi_pressure_imbalance'] = (df['buy_pressure'] - df['sell_pressure']).clip(-1, 1)
    
    # 统计特征
    df['ofi_skewness_10'] = df['net_market_flow'].rolling(window=10, min_periods=3).skew().fillna(0)
    df['ofi_kurtosis_20'] = df['net_market_flow'].rolling(window=20, min_periods=5).kurt().fillna(0)
    df['ofi_max_10'] = df['net_market_flow'].rolling(window=10, min_periods=1).max()
    df['ofi_min_10'] = df['net_market_flow'].rolling(window=10, min_periods=1).min()
    df['ofi_range_10'] = (df['ofi_max_10'] - df['ofi_min_10']).fillna(0)
    
    # 符号一致性
    df['ofi_sign_consistency_5'] = (df['net_market_flow'] > 0).astype(float).rolling(window=5, min_periods=1).mean().fillna(0.5)
    df['ofi_sign_consistency_10'] = (df['net_market_flow'] > 0).astype(float).rolling(window=10, min_periods=1).mean().fillna(0.5)
    
    # EWI特征（指数加权移动平均）
    total_bsize = df['totalbsize'] if 'totalbsize' in df.columns else df[[f'bsize{i}' for i in range(1, 11)]].sum(axis=1).fillna(0)
    total_asize = df['totalasize'] if 'totalasize' in df.columns else df[[f'asize{i}' for i in range(1, 11)]].sum(axis=1).fillna(0)
    
    imb_denom = (total_bsize + total_asize).clip(lower=1e-8)
    imb = ((total_bsize - total_asize) / imb_denom).clip(-1, 1)
    
    # EWM（自动处理NaN）
    df['ewi_5'] = imb.ewm(alpha=0.129, adjust=False, min_periods=1).mean().fillna(0)
    df['ewi_10'] = imb.ewm(alpha=0.067, adjust=False, min_periods=1).mean().fillna(0)
    df['ewi_20'] = imb.ewm(alpha=0.034, adjust=False, min_periods=1).mean().fillna(0)
    
    df['ewi_5_diff'] = df['ewi_5'].diff().fillna(0)
    df['ewi_10_diff'] = df['ewi_10'].diff().fillna(0)
    
    df['ewi_5_deviation'] = (imb - df['ewi_5']).clip(-1, 1)
    df['ewi_10_deviation'] = (imb - df['ewi_10']).clip(-1, 1)
    
    df['ewi_sign_5'] = (df['ewi_5'] > 0).astype(float).rolling(window=5, min_periods=1).mean().fillna(0.5)
    df['ewi_sign_10'] = (df['ewi_10'] > 0).astype(float).rolling(window=10, min_periods=1).mean().fillna(0.5)
    
    # 最终清理：替换所有剩余的inf和nan
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
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


class PreprocessedDataset(Dataset):
    """
    预处理后的数据集
    
    策略：
    1. 首次运行时预处理所有数据并保存为.npy文件
    2. 后续直接加载.npy文件（极快）
    3. 使用内存映射避免OOM
    """
    
    def __init__(
        self, 
        data_dir: str, 
        feature_cols: List[str], 
        label_cols: List[str],
        seq_len: int = 100, 
        stride: int = 5,
        max_files: int = 300,
        augment: bool = False,
        cache_dir: str = None
    ):
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        
        # 缓存目录
        if cache_dir is None:
            cache_dir = os.path.join(data_dir, '..', '_cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 检查是否有缓存的预处理数据
        cache_file = os.path.join(cache_dir, f'data_seq{seq_len}_stride{stride}_files{max_files}.npy')
        label_cache_file = os.path.join(cache_dir, f'labels_seq{seq_len}_stride{stride}_files{max_files}.npy')
        
        if os.path.exists(cache_file) and os.path.exists(label_cache_file):
            print(f"✓ 加载预处理的缓存数据...")
            self.features = np.load(cache_file, mmap_mode='r')  # 内存映射
            self.labels = np.load(label_cache_file, mmap_mode='r')
            print(f"✓ 总样本数: {len(self.features)}")
        else:
            print(f"⚠️ 首次运行，开始预处理数据（这可能需要几分钟）...")
            self._preprocess_and_cache(data_dir, feature_cols, label_cols, seq_len, stride, max_files, cache_file, label_cache_file)
    
    def _preprocess_and_cache(self, data_dir, feature_cols, label_cols, seq_len, stride, max_files, cache_file, label_cache_file):
        """预处理所有数据并缓存"""
        files = sorted(glob.glob(os.path.join(data_dir, "snapshot_sym*.parquet")))[:max_files]
        print(f"   处理 {len(files)} 个文件...")
        
        all_features = []
        all_labels = []
        
        for f in tqdm(files, desc="   预处理"):
            try:
                df = pd.read_parquet(f)
                df = compute_ofi_features_safe(df)  # 使用安全版本
                
                # 确保所有特征列存在
                for col in feature_cols:
                    if col not in df.columns:
                        df[col] = 0
                
                feature_data = df[feature_cols].values.astype(np.float32)
                label_data = df[label_cols].values.astype(np.int64)
                
                # 提取样本
                for i in range(0, len(df) - seq_len, stride):
                    all_features.append(feature_data[i:i+seq_len])
                    all_labels.append(label_data[i+seq_len-1])
                    
            except Exception as e:
                continue
        
        # 转换为数组
        self.features = np.array(all_features, dtype=np.float32)
        self.labels = np.array(all_labels, dtype=np.int64)
        
        # 清理内存
        del all_features, all_labels
        
        # 最终清理NaN
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"   ✓ 预处理完成，总样本数: {len(self.features)}")
        
        # 保存到缓存
        print(f"   ✓ 保存缓存到: {cache_file}")
        np.save(cache_file, self.features)
        np.save(label_cache_file, self.labels)
        print(f"   ✓ 缓存保存完成！下次将直接加载缓存。")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        X = self.features[idx].copy()
        y = self.labels[idx].copy()
        
        if self.augment and np.random.random() < 0.5:
            X = X * np.float32(np.random.uniform(0.98, 1.02))
        
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
        return self.dropout(torch.cat([
            self.branch1(x), self.branch3(x), 
            self.branch5(x), self.branch7(x)
        ], dim=1))


class DilatedMultiScaleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        branch_channels = out_channels // 4
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU()
            ) for d in [1, 2, 4, 8]
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(torch.cat([b(x) for b in self.branches], dim=1))


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
        return F.relu(residual + self.dropout(out))


class MultiScaleTCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, num_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        
        self.ms_blocks = nn.ModuleList([MultiScaleConvBlock(hidden_channels, hidden_channels, dropout) for _ in range(num_blocks)])
        self.dilated_blocks = nn.ModuleList([DilatedMultiScaleBlock(hidden_channels, hidden_channels, dropout) for _ in range(num_blocks)])
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_channels, dropout) for _ in range(num_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_blocks)])
    
    def forward(self, x):
        x = self.input_proj(x)
        for i in range(len(self.ms_blocks)):
            combined = self.ms_blocks[i](x) + self.dilated_blocks[i](x)
            combined = self.norms[i](combined.transpose(1, 2)).transpose(1, 2)
            x = x + combined
            x = self.residual_blocks[i](x)
        return x


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch, channels, seq = x.shape
        squeeze = x.mean(dim=2)
        att = self.channel_att(squeeze).unsqueeze(2)
        x = x * att
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        spatial = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return x * spatial


class HFTModel(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 128, num_blocks: int = 3,
                 num_heads: int = 4, num_transformer_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.embed = nn.Sequential(
            nn.Linear(num_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        
        self.tcn = MultiScaleTCN(hidden_dim, hidden_dim, num_blocks, dropout)
        self.cbam = CBAM(hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.attention_pool = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2), nn.LayerNorm(hidden_dim*2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout/2)
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout/2), nn.Linear(64, 3))
            for _ in range(5)
        ])
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        
        x = self.embed(x)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        x = self.cbam(x.transpose(1, 2)).transpose(1, 2)
        x = self.pos_enc(x)
        x = self.transformer(x)
        
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        x, _ = self.attention_pool(query, x, x)
        x = x.squeeze(1)
        x = self.shared_fc(x)
        
        return tuple(head(x) for head in self.heads)


class AsymmetricFocalLoss(nn.Module):
    """不对称Focal Loss"""
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
    """收益导向损失"""
    def __init__(self, fee_rate=FEE_RATE, profit_weight=2.0, penalty_weight=3.0, hold_weight=0.05):
        super().__init__()
        weight_matrix = torch.tensor([
            [profit_weight, hold_weight, -penalty_weight],
            [-penalty_weight/2, hold_weight, -penalty_weight/2],
            [-penalty_weight, hold_weight, profit_weight]
        ])
        self.register_buffer('weight_matrix', weight_matrix)
    
    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        weight_matrix = self.weight_matrix.to(inputs.device)
        target_weights = weight_matrix[targets]
        expected_profit = (probs * target_weights).sum(dim=1)
        return -expected_profit.mean()


def train_one_epoch(model, loader, criteria, optimizer, device, clip_norm=1.0):
    model.train()
    total_loss = 0
    correct = [0] * 5
    total = 0
    valid_batches = 0
    
    pbar = tqdm(loader, desc="训练", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(inputs)
        loss = sum(criteria[i](outputs[i], targets[:, i]) for i in range(5)) / 5
        
        # 修复：ProfitLoss返回负数，不能用loss.item() < 1e-8判断
        # 只检查nan和inf
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        
        total_loss += loss.item()
        valid_batches += 1
        for i in range(5):
            _, preds = outputs[i].max(1)
            correct[i] += preds.eq(targets[:, i]).sum().item()
        total += targets.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / max(valid_batches, 1), [c / max(total, 1) for c in correct]


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_probs = [[] for _ in range(5)]
    all_targets = [[] for _ in range(5)]
    
    for inputs, targets in tqdm(loader, desc="验证", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        
        for i in range(5):
            probs = F.softmax(outputs[i], dim=1)
            all_probs[i].append(probs.cpu())
            all_targets[i].append(targets[:, i])
    
    return [torch.cat(p) for p in all_probs], [torch.cat(t) for t in all_targets]


def simulate_trading(probs, targets, threshold=0.55):
    max_probs, preds = probs.max(dim=1)
    trade_mask = (preds != 1) & (max_probs > threshold)
    
    trade_preds = preds[trade_mask]
    trade_targets = targets[trade_mask]
    n_trades = len(trade_preds)
    
    accuracy = (preds == targets).float().mean().item()
    hold_rate = 1 - n_trades / len(preds)
    
    if n_trades == 0:
        return {'trade_count': 0, 'avg_profit': 0, 'win_rate': 0, 'accuracy': accuracy, 'hold_rate': hold_rate}
    
    profits = []
    for j in range(n_trades):
        pred, true = trade_preds[j].item(), trade_targets[j].item()
        if pred == true == 2 or pred == true == 0:
            profits.append(0.001 - FEE_RATE)
        elif pred != true and pred in [0, 2] and true in [0, 2]:
            profits.append(-0.002 - FEE_RATE)
        else:
            profits.append(-FEE_RATE)
    
    avg_profit = np.mean(profits)
    win_rate = sum(1 for p in profits if p > 0) / len(profits)
    
    return {
        'trade_count': n_trades,
        'avg_profit': avg_profit,
        'win_rate': win_rate,
        'accuracy': accuracy,
        'hold_rate': hold_rate
    }


def optimize_threshold(probs, targets):
    best_thresh, best_score, best_metrics = 0.55, -float('inf'), None
    
    for t in np.arange(0.45, 0.85, 0.01):
        m = simulate_trading(probs, targets, t)
        if m['trade_count'] >= 10:
            score = m['avg_profit'] * (m['trade_count'] ** 0.25)
            if score > best_score:
                best_score = score
                best_thresh = t
                best_metrics = m
    
    return best_thresh, best_metrics or {'trade_count': 0, 'avg_profit': 0, 'win_rate': 0, 'accuracy': 0, 'hold_rate': 1.0}


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_profit, 
                    best_state, best_thresholds, best_epoch, patience_counter, args, feature_cols):
    """
    保存训练检查点
    
    包含：
    - 模型状态
    - 优化器状态
    - 学习率调度器状态
    - 当前epoch
    - 最佳指标
    - 训练参数
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_profit': best_profit,
        'best_state': best_state,
        'best_thresholds': best_thresholds,
        'best_epoch': best_epoch,
        'patience_counter': patience_counter,
        'args': vars(args),
        'num_features': len(feature_cols),
        'feature_cols': feature_cols,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"  💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """
    加载训练检查点
    
    返回：
    - start_epoch: 从哪个epoch继续训练
    - best_profit: 最佳收益
    - best_state: 最佳模型状态
    - best_thresholds: 最佳阈值
    - best_epoch: 最佳epoch
    - patience_counter: 早停计数器
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_profit = checkpoint['best_profit']
    best_state = checkpoint['best_state']
    best_thresholds = checkpoint['best_thresholds']
    best_epoch = checkpoint['best_epoch']
    patience_counter = checkpoint['patience_counter']
    
    print(f"✅ Checkpoint loaded from epoch {checkpoint['epoch']}")
    print(f"   Best profit: {best_profit:.6f} (Epoch {best_epoch})")
    
    return start_epoch, best_profit, best_state, best_thresholds, best_epoch, patience_counter


def main():
    parser = argparse.ArgumentParser(description="高频量化预测模型 - 高性能版")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output_profit')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--max_files', type=int, default=300)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='从checkpoint继续训练')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='指定checkpoint路径（默认使用output_dir/latest_checkpoint.pt）')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("高频量化预测模型 - 高性能版 (High-Performance)")
    print("=" * 70)
    print(f"设备: {device} | GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'N/A'}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB" if device.type=='cuda' else "")
    print("\n⚡ 性能优化:")
    print("   1. 预处理缓存机制（首次后秒级加载）")
    print("   2. 安全的特征计算（无NaN）")
    print("   3. 高效数据管道（pin_memory + non_blocking）")
    print("=" * 70)
    
    feature_cols = list(BASE_FEATURE_COLS) + OFI_FEATURE_COLS
    print(f"\n特征数量: {len(feature_cols)}")
    
    # 加载数据（带缓存）
    dataset = PreprocessedDataset(
        args.data_dir, feature_cols, LABEL_COLS,
        seq_len=100, stride=args.stride, max_files=args.max_files,
        augment=args.augment
    )
    
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds = torch.utils.data.Subset(dataset, range(train_size))
    val_ds = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    print(f"\n训练集: {len(train_ds)} | 验证集: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    model = HFTModel(len(feature_cols), args.hidden_dim, args.num_blocks, dropout=args.dropout).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    criteria = [AsymmetricFocalLoss() for _ in range(5)]
    profit_criteria = ProfitLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 初始化训练状态
    best_profit, best_state, best_thresholds, best_epoch = -float('inf'), None, {}, 0
    patience_counter, patience = 0, 15
    start_epoch = 0
    
    # 检查是否需要从checkpoint恢复
    if args.resume:
        checkpoint_path = args.checkpoint_path or os.path.join(args.output_dir, 'latest_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            start_epoch, best_profit, best_state, best_thresholds, best_epoch, patience_counter = \
                load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
            print(f"   Resuming from epoch {start_epoch}")
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print("   Starting from scratch...")
    
    print("\n开始训练...")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}\n{'='*60}")
        
        crit = [profit_criteria]*5 if epoch >= 30 else criteria
        loss_name = "Profit" if epoch >= 30 else "AsymFocal"
        
        train_loss, accs = train_one_epoch(model, train_loader, crit, optimizer, device)
        print(f"[{loss_name}] Train Loss: {train_loss:.4f} | Avg Acc: {np.mean(accs):.4f}")
        
        probs, targets = validate(model, val_loader, device)
        
        window_results = []
        for i in range(5):
            thresh, metrics = optimize_threshold(probs[i], targets[i])
            window_results.append((thresh, metrics))
            
            print(f"\n  Window {i} (label_{[5,10,20,40,60][i]}):")
            print(f"    Acc={metrics['accuracy']:.4f} | Thresh={thresh:.2f} | TradeRate={1-metrics['hold_rate']:.4f}")
            print(f"    Trades={metrics['trade_count']} | AvgProfit={metrics['avg_profit']:.6f} | WinRate={metrics['win_rate']:.4f}")
        
        avg_p = np.mean([m[1]['avg_profit'] for m in window_results if m[1]['trade_count'] > 0]) if any(m[1]['trade_count']>0 for m in window_results) else 0
        print(f"\n  ★ Avg Single-Trade Profit: {avg_p:.6f}")
        
        if avg_p > best_profit:
            best_profit = avg_p
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            for i, (t, _) in enumerate(window_results):
                best_thresholds[str(i)] = t
            patience_counter = 0
            print(f"  ✓ Best model saved! (Epoch {best_epoch})")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\nEarly stopped at epoch {epoch+1}")
                # 保存最终checkpoint
                save_checkpoint(
                    os.path.join(args.output_dir, 'latest_checkpoint.pt'),
                    model, optimizer, scheduler, epoch, best_profit, 
                    best_state, best_thresholds, best_epoch, patience_counter, args, feature_cols
                )
                break
        
        scheduler.step()
        
        # 每轮保存checkpoint（用于断点续训）
        save_checkpoint(
            os.path.join(args.output_dir, 'latest_checkpoint.pt'),
            model, optimizer, scheduler, epoch, best_profit, 
            best_state, best_thresholds, best_epoch, patience_counter, args, feature_cols
        )
    
    if best_state:
        torch.save({
            'model_state': best_state, 'num_features': len(feature_cols),
            'feature_cols': feature_cols, 'hidden_dim': args.hidden_dim,
            'best_avg_profit': best_profit, 'best_epoch': best_epoch,
            'thresholds': best_thresholds
        }, os.path.join(args.output_dir, 'best_model.pt'))
    
    config = {"python_version": "3.10", "batch": args.batch_size, "feature": feature_cols,
              "label": LABEL_COLS, "thresholds": best_thresholds}
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best Epoch: {best_epoch} | Best Avg Profit: {best_profit:.6f}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
