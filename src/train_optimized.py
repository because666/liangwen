"""
T-KAN Pro 训练脚本 - 优化版本

优化点：
1. 混合精度训练 (AMP) - V100上加速2-3倍
2. 多进程数据加载 - num_workers=4
3. CUDA优化 - cudnn.benchmark
4. 数据预取 - prefetch_factor=2
5. 向量化特征计算 - 减少Python循环
6. 梯度累积 - 模拟更大batch_size

训练策略：
1. Warm-up: 5 epochs, lr: 1e-5 → 3e-5
2. 主训练: 15 epochs, lr: 3e-5 → 1e-5
3. 微调: 5 epochs, lr: 1e-6
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import TKANPro, create_model, count_parameters
from losses import CompositeProfitLoss, compute_trading_metrics, compute_window_metrics


RAW_FEATURE_COLS = [
    'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'bid6', 'bid7', 'bid8', 'bid9', 'bid10',
    'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask6', 'ask7', 'ask8', 'ask9', 'ask10',
    'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'bsize6', 'bsize7', 'bsize8', 'bsize9', 'bsize10',
    'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'asize6', 'asize7', 'asize8', 'asize9', 'asize10',
    'mb_intst', 'ma_intst', 'lb_intst', 'la_intst', 'cb_intst', 'ca_intst',
]

DERIVED_FEATURE_COLS = [
    'midprice', 'imbalance', 'cumspread',
    'ofi_raw', 'ofi_ewm', 'ofi_velocity', 'ofi_volatility',
]

ALL_FEATURE_COLS = RAW_FEATURE_COLS + DERIVED_FEATURE_COLS

LABEL_COLS = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']

WINDOW_SIZES = [5, 10, 20, 40, 60]


def clean_features(features: np.ndarray, feature_cols: list) -> np.ndarray:
    """清洗特征数据：处理 NaN、Inf 和异常值"""
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 向量化裁剪
    for i, col in enumerate(feature_cols):
        if col.startswith('bid') or col.startswith('ask'):
            features[:, i] = np.clip(features[:, i], -0.3, 0.3)
        elif col.startswith('bsize') or col.startswith('asize'):
            features[:, i] = np.clip(features[:, i], 0, 100)
        elif 'intst' in col:
            features[:, i] = np.clip(features[:, i], -100, 100)
        elif col == 'midprice':
            features[:, i] = np.clip(features[:, i], -0.3, 0.3)
        elif col == 'imbalance':
            features[:, i] = np.clip(features[:, i], -1, 1)
        elif col.startswith('cumspread'):
            features[:, i] = np.clip(features[:, i], -1, 1)
        elif col.startswith('ofi'):
            features[:, i] = np.clip(features[:, i], -100, 100)
    
    return features


def clean_returns(returns: np.ndarray) -> np.ndarray:
    """清洗收益率数据"""
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    returns = np.clip(returns, -0.1, 0.1)
    return returns


def compute_derived_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    """快速计算衍生特征 - 向量化优化版本"""
    df = df.copy()
    n = len(df)
    
    # 获取基础数据
    bid1 = df['bid1'].values if 'bid1' in df.columns else np.zeros(n)
    ask1 = df['ask1'].values if 'ask1' in df.columns else np.zeros(n)
    
    # 计算 midprice - 向量化
    midprice = np.zeros(n, dtype=np.float32)
    both = (bid1 != 0) & (ask1 != 0)
    bid0 = (bid1 == 0) & (ask1 != 0)
    ask0 = (ask1 == 0) & (bid1 != 0)
    midprice[both] = (bid1[both] + ask1[both]) / 2
    midprice[bid0] = ask1[bid0]
    midprice[ask0] = bid1[ask0]
    df['midprice'] = midprice
    
    # 计算 imbalance - 向量化
    bsize_cols = [f'bsize{i}' for i in range(1, 11) if f'bsize{i}' in df.columns]
    asize_cols = [f'asize{i}' for i in range(1, 11) if f'asize{i}' in df.columns]
    
    if bsize_cols and asize_cols:
        total_b = df[bsize_cols].sum(axis=1).values
        total_a = df[asize_cols].sum(axis=1).values
        total = total_b + total_a
        imbalance = np.zeros(n, dtype=np.float32)
        mask = total > 0
        imbalance[mask] = (total_b[mask] - total_a[mask]) / total[mask]
        df['imbalance'] = imbalance
    else:
        df['imbalance'] = 0.0
    
    # 计算 cumspread - 向量化
    cumspread = np.zeros(n, dtype=np.float32)
    for i in range(1, 11):
        ask_col = f'ask{i}'
        bid_col = f'bid{i}'
        if ask_col in df.columns and bid_col in df.columns:
            cumspread += df[ask_col].values - df[bid_col].values
    df['cumspread'] = cumspread
    
    # 计算 OFI 特征
    mb = df['mb_intst'].values if 'mb_intst' in df.columns else np.zeros(n)
    ma = df['ma_intst'].values if 'ma_intst' in df.columns else np.zeros(n)
    ofi_raw = mb - ma
    df['ofi_raw'] = ofi_raw
    
    # OFI EWM - 向量化
    alpha = 0.1
    ofi_ewm = np.zeros(n, dtype=np.float32)
    ofi_ewm[0] = ofi_raw[0]
    for i in range(1, n):
        ofi_ewm[i] = (1 - alpha) * ofi_ewm[i-1] + alpha * ofi_raw[i]
    df['ofi_ewm'] = ofi_ewm
    
    # OFI Velocity - 向量化
    ofi_velocity = np.zeros(n, dtype=np.float32)
    ofi_velocity[1:] = ofi_raw[1:] - ofi_raw[:-1]
    df['ofi_velocity'] = ofi_velocity
    
    # OFI Volatility - 向量化滑动窗口
    window = 10
    ofi_volatility = np.zeros(n, dtype=np.float32)
    if n > window:
        for i in range(window, n):
            ofi_volatility[i] = np.std(ofi_raw[i-window:i])
    df['ofi_volatility'] = ofi_volatility
    
    return df


class StreamingHFTDataset(IterableDataset):
    """流式高频交易数据集 - 优化版本"""
    
    def __init__(self, data_dir: str, feature_cols: list, label_cols: list,
                 seq_len: int = 100, file_list: list = None,
                 mean: np.ndarray = None, std: np.ndarray = None,
                 shuffle_files: bool = True):
        self.data_dir = data_dir
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.seq_len = seq_len
        self.mean = mean
        self.std = std
        self.shuffle_files = shuffle_files
        
        if file_list is not None:
            self.all_files = file_list
        else:
            self.all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
    
    def __iter__(self):
        files = self.all_files.copy()
        if self.shuffle_files:
            import random
            random.shuffle(files)
        
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            try:
                # 使用更快的parquet读取
                df = pd.read_parquet(file_path, engine='pyarrow')
                df = compute_derived_features_fast(df)
                
                # 确保所有特征列存在
                for col in self.feature_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                
                features = df[self.feature_cols].values.astype(np.float32)
                labels = df[self.label_cols].values.astype(np.int64)
                
                features = clean_features(features, self.feature_cols)
                
                midprice = df['midprice'].values if 'midprice' in df.columns else np.zeros(len(df))
                midprice = np.nan_to_num(midprice, nan=0.0, posinf=0.0, neginf=0.0)
                midprice = np.clip(midprice, -0.3, 0.3)
                
                # 预计算收益率
                true_returns = np.zeros((len(df), len(WINDOW_SIZES)), dtype=np.float32)
                for i, w in enumerate(WINDOW_SIZES):
                    if w < len(df):
                        safe_midprice = np.where(np.abs(midprice[:-w]) < 1e-8, 1e-8, midprice[:-w])
                        true_returns[:-w, i] = (midprice[w:] - midprice[:-w]) / safe_midprice
                
                true_returns = clean_returns(true_returns)
                
                # 归一化
                if self.mean is not None and self.std is not None:
                    features = (features - self.mean) / self.std
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                max_window = max(WINDOW_SIZES)
                valid_end = len(df) - max_window
                if valid_end <= self.seq_len:
                    continue
                
                # 生成样本
                for i in range(self.seq_len, valid_end):
                    yield (
                        torch.from_numpy(features[i - self.seq_len:i]),
                        torch.from_numpy(labels[i]),
                        torch.from_numpy(true_returns[i])
                    )
            except Exception as e:
                print(f"加载文件 {file} 失败: {e}")
                continue


class EMA:
    """指数移动平均"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model.state_dict())
        self.num_updates = 0
    
    def update(self, model: nn.Module):
        self.num_updates += 1
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    self.shadow[name] = (
                        self.decay * self.shadow[name] + (1 - self.decay) * param
                    )
    
    def state_dict(self):
        return self.shadow
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def train_epoch_amp(model, dataloader, criterion, optimizer, device, scaler,
                    max_grad_norm: float = 0.3, max_batches: int = None,
                    accumulation_steps: int = 1):
    """训练一个 epoch - 混合精度版本"""
    model.train()
    total_loss = 0
    total_ce = 0
    total_return_loss = 0
    num_batches = 0
    nan_count = 0
    
    pbar = tqdm(dataloader, desc="训练中")
    optimizer.zero_grad()
    
    for batch_idx, (features, labels, true_returns) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
        
        features = features.to(device).float()
        labels = labels.to(device).long()
        true_returns = true_returns.to(device).float()
        
        # 混合精度前向
        with autocast(device_type='cuda'):
            logits, return_pred = model(features)
            loss_dict = criterion(logits, return_pred, labels, true_returns, list(model.parameters()))
            loss = loss_dict['loss'] / accumulation_steps
        
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 3:
                print(f"\n[警告] batch {batch_idx} 出现 NaN/Inf，跳过此 batch (第{nan_count}次)")
                continue
            else:
                print(f"\n[错误] 连续出现 {nan_count} 次 NaN，停止训练")
                break
        
        # 混合精度反向
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (batch_idx + 1) % accumulation_steps == 0:
            # 检查梯度
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                print(f"\n[警告] batch {batch_idx} 梯度包含 NaN/Inf，跳过更新")
                optimizer.zero_grad()
                continue
            
            # 梯度裁剪和更新
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            if grad_norm > 10.0:
                print(f"\n[警告] batch {batch_idx} 梯度范数过大: {grad_norm:.4f}")
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        nan_count = 0
        
        total_loss += loss_dict['loss'].item()
        total_ce += loss_dict['ce_loss']
        total_return_loss += loss_dict['return_loss']
        num_batches += 1
        
        if batch_idx % 100 == 0:
            metrics = compute_trading_metrics(logits, labels, true_returns)
            print(f"\n[调试] batch {batch_idx}: loss={loss_dict['loss'].item():.4f}, "
                  f"ce={loss_dict['ce_loss']:.4f}, "
                  f"return_loss={loss_dict['return_loss']:.4f}")
            print(f"        交易率={metrics['trade_rate']:.3f}, "
                  f"累计收益={metrics['cumulative_return']:.6f}")
        
        pbar.set_postfix({
            'loss': f'{loss_dict["loss"].item():.4f}',
            'ce': f'{loss_dict["ce_loss"]:.4f}',
            'ret': f'{loss_dict["return_loss"]:.4f}',
        })
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'ce_loss': total_ce / num_batches if num_batches > 0 else 0,
        'return_loss': total_return_loss / num_batches if num_batches > 0 else 0,
    }


def evaluate_amp(model, dataloader, criterion, device, max_batches: int = None):
    """评估模型 - 混合精度版本"""
    model.eval()
    all_logits = []
    all_labels = []
    all_returns = []
    total_loss = 0
    num_batches = 0
    
