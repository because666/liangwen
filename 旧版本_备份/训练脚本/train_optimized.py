"""
高频量化预测模型 - 优化版

基于主流量化方法的关键优化：
1. 特征工程：价格动量、订单簿深度、成交量加权、波动率
2. 模型架构：轻量级TCN + Attention，更适合高频数据
3. 训练策略：OneCycleLR + 梯度裁剪 + Label Smoothing
4. 阈值优化：基于验证集的动态阈值选择

避免的问题：
- 不使用 ProfitLoss（梯度不稳定）
- 不使用两阶段训练（复杂且容易出错）
- 不使用过于复杂的模型（容易过拟合）
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
    """
    计算增强特征
    
    基于主流量化方法：
    1. 价格动量特征（多窗口）
    2. 订单簿深度特征
    3. 成交量加权特征
    4. 波动率特征
    5. 价差特征
    """
    df = df.copy()
    
    # ========== 1. 价格动量特征 ==========
    # 多窗口价格变化率
    for window in [5, 10, 20, 40]:
        df[f'price_momentum_{window}'] = df['midprice'].diff(window).fillna(0)
        df[f'price_velocity_{window}'] = df['midprice'].diff(window).diff().fillna(0)
    
    # 价格加速度
    df['price_acceleration'] = df['midprice'].diff().diff().fillna(0)
    
    # 价格相对位置（相对于最高最低价）
    for window in [10, 20]:
        high = df['high'].rolling(window=window, min_periods=1).max()
        low = df['low'].rolling(window=window, min_periods=1).min()
        range_val = (high - low).clip(lower=1e-8)
        df[f'price_position_{window}'] = ((df['midprice'] - low) / range_val).clip(0, 1)
    
    # ========== 2. 订单簿深度特征 ==========
    # 加权中间价（按量加权）
    for i in range(1, 11):
        df[f'weighted_mid_{i}'] = (
            df[f'bid{i}'] * df[f'bsize{i}'] + df[f'ask{i}'] * df[f'asize{i}']
        ) / (df[f'bsize{i}'] + df[f'asize{i}']).clip(lower=1e-8)
    
    # 订单簿不平衡（多档位）
    for i in range(1, 11):
        total_size = df[f'bsize{i}'] + df[f'asize{i}']
        df[f'imbalance_{i}'] = (
            (df[f'bsize{i}'] - df[f'asize{i}']) / total_size.clip(lower=1e-8)
        ).clip(-1, 1)
    
    # 累计不平衡
    df['cum_imbalance_5'] = sum(df[f'imbalance_{i}'] for i in range(1, 6)) / 5
    df['cum_imbalance_10'] = sum(df[f'imbalance_{i}'] for i in range(1, 11)) / 10
    
    # 订单簿斜率（价格与量的关系）
    bid_prices = [df[f'bid{i}'] for i in range(1, 11)]
    bid_sizes = [df[f'bsize{i}'] for i in range(1, 11)]
    ask_prices = [df[f'ask{i}'] for i in range(1, 11)]
    ask_sizes = [df[f'asize{i}'] for i in range(1, 11)]
    
    # 买单压力和卖单压力
    df['bid_pressure_total'] = sum(bid_sizes).clip(lower=1e-8)
    df['ask_pressure_total'] = sum(ask_sizes).clip(lower=1e-8)
    df['pressure_ratio'] = (
        df['bid_pressure_total'] - df['ask_pressure_total']
    ) / (df['bid_pressure_total'] + df['ask_pressure_total']).clip(lower=1e-8)
    
    # ========== 3. 成交量加权特征 ==========
    # 成交量动量
    for window in [5, 10, 20]:
        df[f'volume_momentum_{window}'] = df['volume_delta'].rolling(window=window, min_periods=1).sum().fillna(0)
        df[f'amount_momentum_{window}'] = df['amount_delta'].rolling(window=window, min_periods=1).sum().fillna(0)
    
    # 成交量相对强度
    volume_mean = df['volume_delta'].rolling(window=20, min_periods=1).mean().clip(lower=1e-8)
    df['volume_ratio'] = (df['volume_delta'] / volume_mean).clip(0, 10)
    
    # VWAP 相关
    cum_amount = df['amount_delta'].cumsum()
    cum_volume = df['volume_delta'].cumsum().clip(lower=1e-8)
    df['vwap_proxy'] = cum_amount / cum_volume
    
    # ========== 4. 波动率特征 ==========
    # 多窗口波动率
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df['midprice'].rolling(window=window, min_periods=1).std().fillna(0)
        df[f'volatility_ratio_{window}'] = (
            df[f'volatility_{window}'] / df['midprice'].abs().clip(lower=1e-8)
        ).clip(0, 1)
    
    # 价格范围
    for window in [5, 10, 20]:
        df[f'price_range_{window}'] = (
            df['high'].rolling(window=window, min_periods=1).max() - 
            df['low'].rolling(window=window, min_periods=1).min()
        ).fillna(0)
    
    # ========== 5. 价差特征 ==========
    # 多档位价差
    for i in range(1, 11):
        df[f'spread_{i}'] = df[f'ask{i}'] - df[f'bid{i}']
    
    # 加权价差
    total_bid_size = sum(df[f'bsize{i}'] for i in range(1, 11))
    total_ask_size = sum(df[f'asize{i}'] for i in range(1, 11))
    df['weighted_spread'] = sum(
        df[f'spread_{i}'] * (df[f'bsize{i}'] + df[f'asize{i}']) 
        for i in range(1, 11)
    ) / (total_bid_size + total_ask_size).clip(lower=1e-8)
    
    # 价差变化率
    df['spread_change'] = df['spread1'].diff().fillna(0)
    
    # ========== 6. 订单流特征 ==========
    # 净订单流
    df['net_flow'] = df['mb_intst'].fillna(0) - df['ma_intst'].fillna(0)
    df['net_limit'] = df['lb_intst'].fillna(0) - df['la_intst'].fillna(0)
    df['net_cancel'] = df['cb_intst'].fillna(0) - df['ca_intst'].fillna(0)
    
    # 订单流动量
    for window in [5, 10, 20]:
        df[f'flow_momentum_{window}'] = df['net_flow'].rolling(window=window, min_periods=1).sum().fillna(0)
    
    # 订单流强度比
    total_flow = (df['mb_intst'].fillna(0) + df['ma_intst'].fillna(0)).clip(lower=1e-8)
    df['flow_intensity'] = df['net_flow'].abs() / total_flow
    
    # ========== 7. 时间特征 ==========
    # 价格变化方向持续性
    price_change = df['midprice'].diff()
    df['direction_persistence_5'] = (
        (price_change > 0).rolling(window=5, min_periods=1).mean() - 0.5
    ).fillna(0)
    df['direction_persistence_10'] = (
        (price_change > 0).rolling(window=10, min_periods=1).mean() - 0.5
    ).fillna(0)
    
    # 最终清理
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df


# 特征列定义
FEATURE_COLS = [
    # 基础行情
    'open', 'high', 'low', 'close', 'volume_delta', 'amount_delta', 'midprice',
    # 十档订单簿
    'bid1', 'bsize1', 'bid2', 'bsize2', 'bid3', 'bsize3', 'bid4', 'bsize4', 'bid5', 'bsize5',
    'bid6', 'bsize6', 'bid7', 'bsize7', 'bid8', 'bsize8', 'bid9', 'bsize9', 'bid10', 'bsize10',
    'ask1', 'asize1', 'ask2', 'asize2', 'ask3', 'asize3', 'ask4', 'asize4', 'ask5', 'asize5',
    'ask6', 'asize6', 'ask7', 'asize7', 'ask8', 'asize8', 'ask9', 'asize9', 'ask10', 'asize10',
    # 订单簿衍生
    'avgbid', 'avgask', 'totalbsize', 'totalasize', 'cumspread', 'imbalance',
    'midprice1', 'midprice2', 'midprice3', 'midprice4', 'midprice5',
    'midprice6', 'midprice7', 'midprice8', 'midprice9', 'midprice10',
    'spread1', 'spread2', 'spread3', 'spread4', 'spread5',
    'spread6', 'spread7', 'spread8', 'spread9', 'spread10',
    # 订单流
    'lb_intst', 'la_intst', 'mb_intst', 'ma_intst', 'cb_intst', 'ca_intst',
    'lb_ind', 'la_ind', 'mb_ind', 'ma_ind', 'cb_ind', 'ca_ind',
    'lb_acc', 'la_acc', 'mb_acc', 'ma_acc', 'cb_acc', 'ca_acc',
    # 价格动量
    'price_momentum_5', 'price_momentum_10', 'price_momentum_20', 'price_momentum_40',
    'price_velocity_5', 'price_velocity_10', 'price_velocity_20', 'price_velocity_40',
    'price_acceleration',
    'price_position_10', 'price_position_20',
    # 订单簿深度
    'imbalance_1', 'imbalance_2', 'imbalance_3', 'imbalance_4', 'imbalance_5',
    'imbalance_6', 'imbalance_7', 'imbalance_8', 'imbalance_9', 'imbalance_10',
    'cum_imbalance_5', 'cum_imbalance_10',
    'bid_pressure_total', 'ask_pressure_total', 'pressure_ratio',
    # 成交量
    'volume_momentum_5', 'volume_momentum_10', 'volume_momentum_20',
    'amount_momentum_5', 'amount_momentum_10', 'amount_momentum_20',
    'volume_ratio',
    # 波动率
    'volatility_5', 'volatility_10', 'volatility_20',
    'volatility_ratio_5', 'volatility_ratio_10', 'volatility_ratio_20',
    'price_range_5', 'price_range_10', 'price_range_20',
    # 价差
    'weighted_spread', 'spread_change',
    # 订单流
    'net_flow', 'net_limit', 'net_cancel',
    'flow_momentum_5', 'flow_momentum_10', 'flow_momentum_20',
    'flow_intensity',
    # 时间特征
    'direction_persistence_5', 'direction_persistence_10',
]


class CachedDataset(Dataset):
    """缓存数据集"""
    
    def __init__(self, data_dir: str, feature_cols: List[str], label_cols: List[str],
                 seq_len: int = 100, stride: int = 5, max_files: int = 300, augment: bool = True):
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.seq_len = seq_len
        self.augment = augment
        
        cache_dir = os.path.join(data_dir, '..', '_cache_optimized')
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


class TemporalBlock(nn.Module):
    """时序卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        out = self.conv(x)[:, :, :x.size(2)]
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + self.residual(x)


class LightweightTCN(nn.Module):
    """轻量级TCN"""
    
    def __init__(self, in_channels: int, hidden_channels: int = 128, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append(TemporalBlock(in_channels, hidden_channels, 3, 1, dropout))
        for i in range(num_layers - 1):
            self.layers.append(TemporalBlock(hidden_channels, hidden_channels, 3, 2 ** (i + 1), dropout))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class HFTModelOptimized(nn.Module):
    """优化版高频交易模型"""
    
    def __init__(self, num_features: int, hidden_dim: int = 128, num_layers: int = 4, 
                 num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        
        # 输入嵌入
        self.input_embed = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # TCN
        self.tcn = LightweightTCN(hidden_dim, hidden_dim, num_layers, dropout)
        
        # 注意力
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        
        # 全局池化
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 多任务头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, 3)
            ) for _ in range(5)
        ])
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.input_embed(x)  # [batch, seq_len, hidden]
        
        # TCN
        x = x.transpose(1, 2)  # [batch, hidden, seq_len]
        x = self.tcn(x)
        x = x.transpose(1, 2)  # [batch, seq_len, hidden]
        
        # 注意力
        x = self.attention(x)
        
        # 全局池化（使用最后一个时间步）
        x = x[:, -1, :]  # [batch, hidden]
        x = self.global_pool(x)
        
        # 多任务输出
        return tuple(head(x) for head in self.heads)


class FocalLoss(nn.Module):
    """Focal Loss with asymmetric weights"""
    
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
                    optimizer: torch.optim.Optimizer, scheduler, device: torch.device) -> Tuple[float, List[float]]:
    model.train()
    total_loss = 0
    correct = [0] * 5
    total = 0
    
    pbar = tqdm(loader, desc="训练", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(inputs)
        loss = sum(criterion(outputs[i], targets[:, i]) for i in range(5)) / 5
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        for i in range(5):
            _, preds = outputs[i].max(1)
            correct[i] += preds.eq(targets[:, i]).sum().item()
        total += targets.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / max(len(loader), 1), [c / max(total, 1) for c in correct]


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    model.eval()
    all_probs = [[] for _ in range(5)]
    all_targets = [[] for _ in range(5)]
    
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        
        for i in range(5):
            probs = F.softmax(outputs[i], dim=1)
            all_probs[i].append(probs.cpu())
            all_targets[i].append(targets[:, i].cpu())
    
    return [torch.cat(p) for p in all_probs], [torch.cat(t) for t in all_targets]


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


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, epoch: int, 
                    best_profit: float, best_state: dict, best_thresholds: dict, 
                    best_epoch: int, patience_counter: int, args, feature_cols: List[str]):
    torch.save({
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
        'feature_cols': feature_cols,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="高频量化预测模型 - 优化版")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output_optimized')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.15)
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
    print("高频量化预测模型 - 优化版")
    print("=" * 70)
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print("\n优化策略:")
    print("  1. 增强特征工程（价格动量、订单簿深度、成交量加权、波动率）")
    print("  2. 轻量级TCN + Attention架构")
    print("  3. OneCycleLR学习率调度")
    print("  4. Focal Loss with asymmetric weights")
    print("  5. 动态阈值优化")
    print("=" * 70)
    
    # 数据加载
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
    
    # 模型
    model = HFTModelOptimized(
        len(FEATURE_COLS), args.hidden_dim, args.num_layers, dropout=args.dropout
    ).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数
    criterion = FocalLoss(alpha=[2.5, 0.15, 2.5], gamma=2.0, label_smoothing=0.05)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度器 - OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 3, epochs=args.epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos'
    )
    
    # 训练状态
    best_profit, best_state, best_thresholds, best_epoch = -float('inf'), None, {}, 0
    patience_counter, patience = 0, 12
    start_epoch = 0
    
    # 恢复训练
    if args.resume:
        checkpoint_path = os.path.join(args.output_dir, 'latest_checkpoint.pt')
        if os.path.exists(checkpoint_path):
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
            print(f"✅ 从 epoch {checkpoint['epoch']} 恢复训练")
    
    print("\n开始训练...")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{args.epochs}\n{'='*60}")
        
        # 训练
        train_loss, accs = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f} | Avg Acc: {np.mean(accs):.4f}")
        
        # 验证
        probs, targets = validate(model, val_loader, device)
        
        # 评估每个窗口
        window_results = []
        for i in range(5):
            thresh, metrics = optimize_threshold(probs[i], targets[i])
            window_results.append((thresh, metrics))
            
            print(f"\n  Window {i} (label_{[5,10,20,40,60][i]}):")
            print(f"    Acc={metrics['accuracy']:.4f} | Thresh={thresh:.2f} | TradeRate={1-metrics['hold_rate']:.4f}")
            print(f"    Trades={metrics['trade_count']} | AvgProfit={metrics['avg_profit']:.6f} | WinRate={metrics['win_rate']:.4f}")
        
        # 计算平均收益
        avg_p = np.mean([m[1]['avg_profit'] for m in window_results if m[1]['trade_count'] > 0]) if any(m[1]['trade_count']>0 for m in window_results) else 0
        print(f"\n  ★ Avg Single-Trade Profit: {avg_p:.6f}")
        
        # 保存最佳模型
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
                save_checkpoint(os.path.join(args.output_dir, 'latest_checkpoint.pt'),
                    model, optimizer, scheduler, epoch, best_profit, best_state, 
                    best_thresholds, best_epoch, patience_counter, args, FEATURE_COLS)
                break
        
        # 保存checkpoint
        save_checkpoint(os.path.join(args.output_dir, 'latest_checkpoint.pt'),
            model, optimizer, scheduler, epoch, best_profit, best_state, 
            best_thresholds, best_epoch, patience_counter, args, FEATURE_COLS)
    
    # 保存最终模型
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
    
    # 保存配置
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
    print(f"Training Complete!")
    print(f"Best Epoch: {best_epoch} | Best Avg Profit: {best_profit:.6f}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
