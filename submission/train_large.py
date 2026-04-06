"""
高频量化预测模型 - 大规模版

针对 A10 GPU 优化：
1. 更大的隐藏维度（384）
2. 更深的网络结构（6层TCN + 4层Transformer）
3. SE Block 通道注意力
4. 多分支特征融合
5. 目标参数量：15-20M

充分利用 GPU 算力，追求更高收益
"""

from __future__ import annotations

import os
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
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

LABEL_COLS = ["label_5", "label_10", "label_20", "label_40", "label_60"]
FEE_RATE = 0.0002


def compute_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算增强特征"""
    df = df.copy()
    
    # ========== 1. 价格动量特征 ==========
    for window in [3, 5, 10, 20, 40, 60]:
        df[f'price_momentum_{window}'] = df['midprice'].diff(window).fillna(0)
        if window >= 5:
            df[f'price_velocity_{window}'] = df['midprice'].diff(window).diff().fillna(0)
    
    df['price_acceleration'] = df['midprice'].diff().diff().fillna(0)
    
    for window in [5, 10, 20, 40]:
        high = df['high'].rolling(window=window, min_periods=1).max()
        low = df['low'].rolling(window=window, min_periods=1).min()
        range_val = (high - low).clip(lower=1e-8)
        df[f'price_position_{window}'] = ((df['midprice'] - low) / range_val).clip(0, 1)
    
    # ========== 2. 订单簿深度特征 ==========
    for i in range(1, 11):
        df[f'imbalance_{i}'] = (
            (df[f'bsize{i}'] - df[f'asize{i}']) / 
            (df[f'bsize{i}'] + df[f'asize{i}']).clip(lower=1e-8)
        ).clip(-1, 1)
    
    df['cum_imbalance_3'] = sum(df[f'imbalance_{i}'] for i in range(1, 4)) / 3
    df['cum_imbalance_5'] = sum(df[f'imbalance_{i}'] for i in range(1, 6)) / 5
    df['cum_imbalance_10'] = sum(df[f'imbalance_{i}'] for i in range(1, 11)) / 10
    
    df['bid_pressure_total'] = sum(df[f'bsize{i}'] for i in range(1, 11)).clip(lower=1e-8)
    df['ask_pressure_total'] = sum(df[f'asize{i}'] for i in range(1, 11)).clip(lower=1e-8)
    df['pressure_ratio'] = (
        df['bid_pressure_total'] - df['ask_pressure_total']
    ) / (df['bid_pressure_total'] + df['ask_pressure_total']).clip(lower=1e-8)
    
    # 加权中间价
    for i in range(1, 11):
        df[f'weighted_mid_{i}'] = (
            df[f'bid{i}'] * df[f'bsize{i}'] + df[f'ask{i}'] * df[f'asize{i}']
        ) / (df[f'bsize{i}'] + df[f'asize{i}']).clip(lower=1e-8)
    
    # ========== 3. 成交量特征 ==========
    for window in [3, 5, 10, 20]:
        df[f'volume_momentum_{window}'] = df['volume_delta'].rolling(window=window, min_periods=1).sum().fillna(0)
        df[f'amount_momentum_{window}'] = df['amount_delta'].rolling(window=window, min_periods=1).sum().fillna(0)
    
    volume_mean = df['volume_delta'].rolling(window=20, min_periods=1).mean().clip(lower=1e-8)
    df['volume_ratio'] = (df['volume_delta'] / volume_mean).clip(0, 10)
    
    # ========== 4. 波动率特征 ==========
    for window in [5, 10, 20, 40]:
        df[f'volatility_{window}'] = df['midprice'].rolling(window=window, min_periods=1).std().fillna(0)
        df[f'price_range_{window}'] = (
            df['high'].rolling(window=window, min_periods=1).max() - 
            df['low'].rolling(window=window, min_periods=1).min()
        ).fillna(0)
    
    # ========== 5. 订单流特征 ==========
    df['net_flow'] = df['mb_intst'].fillna(0) - df['ma_intst'].fillna(0)
    df['net_limit'] = df['lb_intst'].fillna(0) - df['la_intst'].fillna(0)
    df['net_cancel'] = df['cb_intst'].fillna(0) - df['ca_intst'].fillna(0)
    
    for window in [3, 5, 10, 20]:
        df[f'flow_momentum_{window}'] = df['net_flow'].rolling(window=window, min_periods=1).sum().fillna(0)
    
    total_flow = (df['mb_intst'].fillna(0) + df['ma_intst'].fillna(0)).clip(lower=1e-8)
    df['flow_intensity'] = df['net_flow'].abs() / total_flow
    
    # ========== 6. 时间特征 ==========
    price_change = df['midprice'].diff()
    for window in [3, 5, 10, 20]:
        df[f'direction_persistence_{window}'] = (
            (price_change > 0).rolling(window=window, min_periods=1).mean() - 0.5
        ).fillna(0)
    
    # ========== 7. 高级特征 ==========
    # 价格动量加速度
    df['momentum_acceleration'] = df['price_momentum_5'].diff().fillna(0)
    
    # 订单簿斜率
    bid_slope = (df['bid10'] - df['bid1']) / 9
    ask_slope = (df['ask10'] - df['ask1']) / 9
    df['orderbook_slope'] = (ask_slope - bid_slope).fillna(0)
    
    # 价差加权
    total_size = sum(df[f'bsize{i}'] + df[f'asize{i}'] for i in range(1, 11))
    df['weighted_spread'] = sum(
        (df[f'ask{i}'] - df[f'bid{i}']) * (df[f'bsize{i}'] + df[f'asize{i}'])
        for i in range(1, 11)
    ) / total_size.clip(lower=1e-8)
    
    # 最终清理
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    # ========== 8. 特征归一化 ==========
    # 对可能值很大的特征进行标准化
    large_value_cols = ['amount_delta', 'amount_momentum_3', 'amount_momentum_5', 
                        'amount_momentum_10', 'amount_momentum_20']
    for col in large_value_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / (std + 1e-8)
            df[col] = df[col].clip(-10, 10)
    
    # 对累积特征进行缩放
    cumsum_cols = [c for c in df.columns if 'momentum' in c or 'cum' in c]
    for col in cumsum_cols:
        if col in df.columns and col not in large_value_cols:
            max_val = df[col].abs().max()
            if max_val > 10:
                df[col] = df[col] / max_val * 5
    
    return df


FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume_delta', 'amount_delta', 'midprice',
    'bid1', 'bsize1', 'bid2', 'bsize2', 'bid3', 'bsize3', 'bid4', 'bsize4', 'bid5', 'bsize5',
    'bid6', 'bsize6', 'bid7', 'bsize7', 'bid8', 'bsize8', 'bid9', 'bsize9', 'bid10', 'bsize10',
    'ask1', 'asize1', 'ask2', 'asize2', 'ask3', 'asize3', 'ask4', 'asize4', 'ask5', 'asize5',
    'ask6', 'asize6', 'ask7', 'asize7', 'ask8', 'asize8', 'ask9', 'asize9', 'ask10', 'asize10',
    'avgbid', 'avgask', 'totalbsize', 'totalasize', 'cumspread', 'imbalance',
    'midprice1', 'midprice2', 'midprice3', 'midprice4', 'midprice5',
    'midprice6', 'midprice7', 'midprice8', 'midprice9', 'midprice10',
    'spread1', 'spread2', 'spread3', 'spread4', 'spread5',
    'spread6', 'spread7', 'spread8', 'spread9', 'spread10',
    'lb_intst', 'la_intst', 'mb_intst', 'ma_intst', 'cb_intst', 'ca_intst',
    'lb_ind', 'la_ind', 'mb_ind', 'ma_ind', 'cb_ind', 'ca_ind',
    'lb_acc', 'la_acc', 'mb_acc', 'ma_acc', 'cb_acc', 'ca_acc',
    'price_momentum_3', 'price_momentum_5', 'price_momentum_10', 'price_momentum_20', 'price_momentum_40', 'price_momentum_60',
    'price_velocity_5', 'price_velocity_10', 'price_velocity_20', 'price_velocity_40',
    'price_acceleration',
    'price_position_5', 'price_position_10', 'price_position_20', 'price_position_40',
    'imbalance_1', 'imbalance_2', 'imbalance_3', 'imbalance_4', 'imbalance_5',
    'imbalance_6', 'imbalance_7', 'imbalance_8', 'imbalance_9', 'imbalance_10',
    'cum_imbalance_3', 'cum_imbalance_5', 'cum_imbalance_10',
    'bid_pressure_total', 'ask_pressure_total', 'pressure_ratio',
    'volume_momentum_3', 'volume_momentum_5', 'volume_momentum_10', 'volume_momentum_20',
    'amount_momentum_3', 'amount_momentum_5', 'amount_momentum_10', 'amount_momentum_20',
    'volume_ratio',
    'volatility_5', 'volatility_10', 'volatility_20', 'volatility_40',
    'price_range_5', 'price_range_10', 'price_range_20', 'price_range_40',
    'net_flow', 'net_limit', 'net_cancel',
    'flow_momentum_3', 'flow_momentum_5', 'flow_momentum_10', 'flow_momentum_20',
    'flow_intensity',
    'direction_persistence_3', 'direction_persistence_5', 'direction_persistence_10', 'direction_persistence_20',
    'momentum_acceleration', 'orderbook_slope', 'weighted_spread',
]


class CachedDataset(Dataset):
    def __init__(self, data_dir: str, feature_cols: List[str], label_cols: List[str],
                 seq_len: int = 100, stride: int = 5, max_files: int = 300, augment: bool = True):
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.seq_len = seq_len
        self.augment = augment
        
        cache_dir = os.path.join(data_dir, '..', '_cache_large')
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f'data_seq{seq_len}_stride{stride}_files{max_files}.npy')
        label_cache_file = os.path.join(cache_dir, f'labels_seq{seq_len}_stride{stride}_files{max_files}.npy')
        
        if os.path.exists(cache_file) and os.path.exists(label_cache_file):
            print(f"✓ 加载缓存数据...")
            self.features = np.load(cache_file, mmap_mode='r')
            self.labels = np.load(label_cache_file, mmap_mode='r')
            print(f"✓ 样本数: {len(self.features)}")
        else:
            print(f"⚠️ 首次运行，预处理数据...")
            self._preprocess(data_dir, cache_file, label_cache_file, max_files, stride)
    
    def _preprocess(self, data_dir: str, cache_file: str, label_cache_file: str, max_files: int, stride: int):
        files = sorted(glob.glob(os.path.join(data_dir, "snapshot_sym*.parquet")))[:max_files]
        print(f"   处理 {len(files)} 个文件...")
        
        all_features, all_labels = [], []
        
        for f in tqdm(files, desc="预处理"):
            try:
                df = pd.read_parquet(f)
                df = compute_enhanced_features(df)
                
                for col in self.feature_cols:
                    if col not in df.columns:
                        df[col] = 0
                
                feature_data = df[self.feature_cols].values.astype(np.float32)
                label_data = df[self.label_cols].values.astype(np.int64)
                
                for i in range(0, len(df) - self.seq_len, stride):
                    all_features.append(feature_data[i:i+self.seq_len])
                    all_labels.append(label_data[i+self.seq_len-1])
            except Exception as e:
                continue
        
        self.features = np.array(all_features, dtype=np.float32)
        self.labels = np.array(all_labels, dtype=np.int64)
        
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"   ✓ 预处理完成，样本数: {len(self.features)}")
        np.save(cache_file, self.features)
        np.save(label_cache_file, self.labels)
        print(f"   ✓ 缓存已保存")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        X = self.features[idx].copy()
        y = self.labels[idx].copy()
        
        if self.augment and np.random.random() < 0.3:
            X = X * np.float32(np.random.uniform(0.98, 1.02))
            X = X + np.float32(np.random.normal(0, 0.001, X.shape))
        
        return torch.from_numpy(X), torch.from_numpy(y)


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


class FocalLoss(nn.Module):
    def __init__(self, alpha: List[float] = [2.5, 0.15, 2.5], gamma: float = 2.0, 
                 label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        alpha = self.alpha.to(inputs.device)
        at = alpha[targets]
        focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer, scheduler, device: torch.device, 
                    scaler=None, epoch: int = 0) -> Tuple[float, List[float]]:
    """
    训练一个epoch
    
    返回：
    - 平均损失
    - 各窗口准确率列表
    """
    model.train()
    total_loss = 0
    correct = [0] * 5
    total = 0
    batch_count = 0
    nan_count = 0
    
    print(f"\n  📊 开始训练 Epoch {epoch+1}...")
    print(f"  ├─ 总批次: {len(loader)}")
    
    # 调试：打印第一个批次的信息
    debug_printed = False
    
    pbar = tqdm(loader, desc="  ├─ 训练进度", leave=False, ncols=100)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # 调试信息
        if not debug_printed and batch_idx == 0:
            print(f"\n  🔍 调试信息 (批次 0):")
            print(f"     ├─ 输入形状: {inputs.shape}")
            print(f"     ├─ 输入范围: [{inputs.min():.4f}, {inputs.max():.4f}]")
            print(f"     ├─ 目标形状: {targets.shape}")
            print(f"     ├─ 目标唯一值: {torch.unique(targets)}")
            print(f"     └─ 目标分布: {torch.bincount(targets[:, 0], minlength=3).tolist()}")
            debug_printed = True
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            
            # 调试：检查输出
            if batch_idx == 0 and debug_printed:
                print(f"\n     ├─ 输出形状: {[o.shape for o in outputs]}")
                print(f"     ├─ 输出范围: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
            
            # 计算损失
            try:
                loss = sum(criterion(outputs[i], targets[:, i]) for i in range(5)) / 5
            except Exception as e:
                print(f"\n  ❌ 损失计算错误 (批次 {batch_idx}): {e}")
                print(f"     ├─ outputs[0] 形状: {outputs[0].shape}")
                print(f"     ├─ targets[:, 0] 形状: {targets[:, 0].shape}")
                print(f"     └─ targets[:, 0] 值: {targets[:, 0][:10]}")
                raise
        
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 3:
                print(f"  ⚠️  批次 {batch_idx}: 检测到异常损失值={loss.item()}, 跳过")
                # 打印调试信息
                with torch.no_grad():
                    debug_out = model(inputs)
                    print(f"     ├─ 输出范围: [{debug_out[0].min():.4f}, {debug_out[0].max():.4f}]")
                    print(f"     └─ 是否有inf/nan: {torch.isnan(debug_out[0]).any() or torch.isinf(debug_out[0]).any()}")
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        batch_count += 1
        for i in range(5):
            _, preds = outputs[i].max(1)
            correct[i] += preds.eq(targets[:, i]).sum().item()
        total += targets.size(0)
        
        # 更新进度条信息
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            '损失': f'{loss.item():.4f}',
            'LR': f'{current_lr:.2e}',
            '梯度范数': f'{grad_norm:.2f}'
        })
        
        # 每50个批次打印一次详细信息
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / batch_count
            avg_acc = np.mean([c / max(total, 1) for c in correct])
            print(f"  │  批次 {batch_idx+1}/{len(loader)}: 损失={loss.item():.4f}, 平均损失={avg_loss:.4f}, 平均准确率={avg_acc:.4f}")
    
    # Epoch 训练完成总结
    avg_loss = total_loss / max(batch_count, 1)
    avg_acc = np.mean([c / max(total, 1) for c in correct])
    
    print(f"  ├─ 训练完成统计:")
    print(f"  │  ├─ 有效批次: {batch_count}/{len(loader)}")
    print(f"  │  ├─ 跳过批次: {nan_count}")
    print(f"  │  ├─ 平均损失: {avg_loss:.4f}")
    print(f"  │  └─ 平均准确率: {avg_acc:.4f}")
    
    # 打印各窗口准确率
    accs = [c / max(total, 1) for c in correct]
    print(f"  └─ 各窗口准确率:")
    for i, acc in enumerate(accs):
        window_name = ['5tick', '10tick', '20tick', '40tick', '60tick'][i]
        print(f"     ├─ Window {i} ({window_name}): {acc:.4f}")
    
    return avg_loss, accs


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    验证模型
    
    返回：
    - 各窗口的概率预测
    - 各窗口的真实标签
    """
    model.eval()
    all_probs = [[] for _ in range(5)]
    all_targets = [[] for _ in range(5)]
    
    print(f"\n  🔍 开始验证...")
    print(f"  ├─ 验证批次: {len(loader)}")
    
    pbar = tqdm(loader, desc="  ├─ 验证进度", leave=False, ncols=100)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
        
        for i in range(5):
            probs = F.softmax(outputs[i], dim=1)
            all_probs[i].append(probs.cpu())
            all_targets[i].append(targets[:, i].cpu())
        
        pbar.set_postfix({'批次': f'{batch_idx+1}/{len(loader)}'})
    
    # 合并所有批次
    probs_cat = [torch.cat(p) for p in all_probs]
    targets_cat = [torch.cat(t) for t in all_targets]
    
    print(f"  └─ 验证完成: 共 {len(probs_cat[0])} 个样本")
    
    return probs_cat, targets_cat


def simulate_trading(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.55) -> Dict:
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
    
    return {
        'trade_count': n_trades,
        'avg_profit': np.mean(profits),
        'win_rate': sum(1 for p in profits if p > 0) / len(profits),
        'accuracy': accuracy,
        'hold_rate': hold_rate
    }


def optimize_threshold(probs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, Dict]:
    best_thresh, best_score, best_metrics = 0.55, -float('inf'), None
    
    for t in np.arange(0.40, 0.85, 0.01):
        m = simulate_trading(probs, targets, t)
        if m['trade_count'] >= 10:
            score = m['avg_profit'] * (m['trade_count'] ** 0.3)
            if score > best_score:
                best_score = score
                best_thresh = t
                best_metrics = m
    
    return best_thresh, best_metrics or {'trade_count': 0, 'avg_profit': 0, 'win_rate': 0, 'accuracy': 0, 'hold_rate': 1.0}


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, scaler, epoch: int, 
                    best_profit: float, best_state: dict, best_thresholds: dict, 
                    best_epoch: int, patience_counter: int, args, feature_cols: List[str]):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_profit': best_profit,
        'best_state': best_state,
        'best_thresholds': best_thresholds,
        'best_epoch': best_epoch,
        'patience_counter': patience_counter,
        'args': vars(args),
        'feature_cols': feature_cols,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="高频量化预测模型 - 大规模版")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output_large')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--num_tcn_layers', type=int, default=6)
    parser.add_argument('--num_transformer_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.12)
    parser.add_argument('--max_files', type=int, default=300)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--resume', action='store_true', default=False)
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("高频量化预测模型 - 大规模版")
    print("=" * 70)
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print("\n模型配置:")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  TCN层数: {args.num_tcn_layers}")
    print(f"  Transformer层数: {args.num_transformer_layers}")
    print(f"  注意力头数: {args.num_heads}")
    print("=" * 70)
    
    dataset = CachedDataset(
        args.data_dir, FEATURE_COLS, LABEL_COLS,
        seq_len=100, stride=args.stride, max_files=args.max_files, augment=args.augment
    )
    
    train_size = int(0.85 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(train_size))
    val_ds = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    print(f"\n训练集: {len(train_ds)} | 验证集: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    model = HFTModelLarge(
        len(FEATURE_COLS), args.hidden_dim, args.num_tcn_layers,
        args.num_transformer_layers, args.num_heads, args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    
    criterion = FocalLoss(alpha=[2.5, 0.15, 2.5], gamma=2.0, label_smoothing=0.05)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 5, epochs=args.epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.05, anneal_strategy='cos'
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    best_profit, best_state, best_thresholds, best_epoch = -float('inf'), None, {}, 0
    patience_counter, patience = 0, 15
    start_epoch = 0
    
    if args.resume:
        checkpoint_path = os.path.join(args.output_dir, 'latest_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_profit = checkpoint['best_profit']
            best_state = checkpoint['best_state']
            best_thresholds = checkpoint['best_thresholds']
            best_epoch = checkpoint['best_epoch']
            patience_counter = checkpoint['patience_counter']
            print(f"✅ 从 epoch {checkpoint['epoch']} 恢复训练")
    
    print("\n" + "=" * 70)
    print("🚀 开始训练...")
    print("=" * 70)
    print(f"训练配置:")
    print(f"  ├─ 总轮数: {args.epochs}")
    print(f"  ├─ 批次大小: {args.batch_size}")
    print(f"  ├─ 初始学习率: {args.lr:.2e}")
    print(f"  ├─ 最大学习率: {args.lr * 5:.2e}")
    print(f"  ├─ 早停耐心: {patience} 轮")
    print(f"  └─ 混合精度训练: 启用")
    print("=" * 70)
    
    import time
    epoch_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_begin_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"📅 Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        # 打印GPU显存使用情况
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"  💾 GPU显存: 已分配 {allocated:.2f}GB / 已预留 {reserved:.2f}GB")
        
        # 训练
        train_loss, accs = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler, epoch)
        
        # 验证
        probs, targets = validate(model, val_loader, device)
        
        # 评估各窗口
        print(f"\n  📈 验证结果:")
        window_results = []
        for i in range(5):
            thresh, metrics = optimize_threshold(probs[i], targets[i])
            window_results.append((thresh, metrics))
            
            window_name = ['5tick', '10tick', '20tick', '40tick', '60tick'][i]
            print(f"\n  ┌─ Window {i} ({window_name}) ─────────────────────")
            print(f"  │  📊 准确率: {metrics['accuracy']:.4f}")
            print(f"  │  🎯 最佳阈值: {thresh:.2f}")
            print(f"  │  📊 交易率: {1-metrics['hold_rate']:.4f} ({metrics['trade_count']} 笔交易)")
            print(f"  │  💰 平均收益: {metrics['avg_profit']:.6f}")
            print(f"  │  🏆 胜率: {metrics['win_rate']:.4f}")
            print(f"  └────────────────────────────────")
        
        # 计算平均收益
        valid_profits = [m[1]['avg_profit'] for m in window_results if m[1]['trade_count'] > 0]
        avg_p = np.mean(valid_profits) if valid_profits else 0
        
        # 计算epoch耗时
        epoch_time = time.time() - epoch_begin_time
        total_time = time.time() - epoch_start_time
        avg_epoch_time = total_time / (epoch + 1 - start_epoch)
        eta = avg_epoch_time * (args.epochs - epoch - 1)
        
        print(f"\n  ⏱️  Epoch耗时: {epoch_time:.1f}秒")
        print(f"  ⏱️  平均每轮: {avg_epoch_time:.1f}秒")
        print(f"  ⏱️  预计剩余: {eta/60:.1f}分钟")
        print(f"\n  ⭐ 平均单笔收益: {avg_p:.6f}")
        
        # 保存最佳模型
        if avg_p > best_profit:
            best_profit = avg_p
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            for i, (t, _) in enumerate(window_results):
                best_thresholds[str(i)] = t
            patience_counter = 0
            
            print(f"\n  ✅ 发现更好的模型！")
            print(f"     ├─ 最佳Epoch: {best_epoch}")
            print(f"     ├─ 最佳收益: {best_profit:.6f}")
            print(f"     └─ 模型已保存")
        else:
            patience_counter += 1
            print(f"\n  ⚠️  未提升 ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n  🛑 早停触发！连续 {patience} 轮未提升")
                save_checkpoint(os.path.join(args.output_dir, 'latest_checkpoint.pt'),
                    model, optimizer, scheduler, scaler, epoch, best_profit, best_state, 
                    best_thresholds, best_epoch, patience_counter, args, FEATURE_COLS)
                break
        
        # 保存checkpoint
        save_checkpoint(os.path.join(args.output_dir, 'latest_checkpoint.pt'),
            model, optimizer, scheduler, scaler, epoch, best_profit, best_state, 
            best_thresholds, best_epoch, patience_counter, args, FEATURE_COLS)
        print(f"\n  💾 Checkpoint已保存: latest_checkpoint.pt")
    
    # 训练完成
    total_training_time = time.time() - epoch_start_time
    
    if best_state:
        torch.save({
            'model_state': best_state,
            'num_features': len(FEATURE_COLS),
            'feature_cols': FEATURE_COLS,
            'hidden_dim': args.hidden_dim,
            'best_avg_profit': best_profit,
            'best_epoch': best_epoch,
            'thresholds': best_thresholds
        }, os.path.join(args.output_dir, 'best_model.pt'))
    
    config = {
        "python_version": "3.10",
        "batch": args.batch_size,
        "feature": FEATURE_COLS,
        "label": LABEL_COLS,
        "thresholds": best_thresholds
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"🎉 训练完成！")
    print(f"{'='*70}")
    print(f"📊 训练统计:")
    print(f"  ├─ 总训练时间: {total_training_time/60:.1f} 分钟")
    print(f"  ├─ 最佳Epoch: {best_epoch}")
    print(f"  ├─ 最佳平均收益: {best_profit:.6f}")
    print(f"  └─ 最佳阈值: {best_thresholds}")
    print(f"\n📁 输出文件:")
    print(f"  ├─ best_model.pt (最佳模型)")
    print(f"  ├─ latest_checkpoint.pt (最新检查点)")
    print(f"  └─ config.json (配置文件)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
