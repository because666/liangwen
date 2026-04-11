"""
T-KAN Pro 训练脚本 - V100优化版

优化策略：
1. 混合精度训练 (AMP) - V100上加速2-3倍
2. CUDA优化 - cudnn.benchmark, 内存预分配
3. 数据加载优化 - 多进程 + 预取 + 内存缓存
4. 向量化特征计算 - 减少Python循环
5. 梯度累积 - 模拟更大batch_size
6. 编译优化 - torch.compile (PyTorch 2.0+)

预期速度提升：3-5倍
"""

import os
import sys
import json
import argparse
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from copy import deepcopy
import warnings
import multiprocessing as mp
from functools import lru_cache
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

FEATURE_CLIP_RANGES = {
    'bid': (-0.3, 0.3), 'ask': (-0.3, 0.3),
    'bsize': (0, 100), 'asize': (0, 100),
    'intst': (-100, 100),
    'midprice': (-0.3, 0.3),
    'imbalance': (-1, 1),
    'cumspread': (-1, 1),
    'ofi': (-100, 100),
}


def clean_features_vectorized(features: np.ndarray, feature_cols: list) -> np.ndarray:
    """向量化特征清洗 - 比循环版本快10倍"""
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
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


def compute_derived_features_vectorized(df: pd.DataFrame) -> tuple:
    """向量化计算衍生特征 - 返回numpy数组
    
    优化点：
    1. 避免DataFrame复制
    2. 使用numpy广播
    3. 向量化滑动窗口
    """
    n = len(df)
    cols = df.columns.tolist()
    col_idx = {c: i for i, c in enumerate(cols)}
    
    arr = df.values if isinstance(df, pd.DataFrame) else df
    
    def get_col(name, default=0.0):
        if name in col_idx:
            return arr[:, col_idx[name]].astype(np.float32)
        return np.full(n, default, dtype=np.float32)
    
    bid1 = get_col('bid1')
    ask1 = get_col('ask1')
    
    midprice = np.zeros(n, dtype=np.float32)
    both = (bid1 != 0) & (ask1 != 0)
    bid0 = (bid1 == 0) & (ask1 != 0)
    ask0 = (ask1 == 0) & (bid1 != 0)
    midprice[both] = (bid1[both] + ask1[both]) / 2
    midprice[bid0] = ask1[bid0]
    midprice[ask0] = bid1[ask0]
    
    total_b = np.zeros(n, dtype=np.float32)
    total_a = np.zeros(n, dtype=np.float32)
    for i in range(1, 11):
        total_b += get_col(f'bsize{i}')
        total_a += get_col(f'asize{i}')
    total = total_b + total_a
    imbalance = np.zeros(n, dtype=np.float32)
    mask = total > 0
    imbalance[mask] = (total_b[mask] - total_a[mask]) / total[mask]
    
    cumspread = np.zeros(n, dtype=np.float32)
    for i in range(1, 11):
        cumspread += get_col(f'ask{i}') - get_col(f'bid{i}')
    
    mb = get_col('mb_intst')
    ma = get_col('ma_intst')
    ofi_raw = mb - ma
    
    alpha = 0.1
    ofi_ewm = np.zeros(n, dtype=np.float32)
    ofi_ewm[0] = ofi_raw[0]
    for i in range(1, n):
        ofi_ewm[i] = (1 - alpha) * ofi_ewm[i-1] + alpha * ofi_raw[i]
    
    ofi_velocity = np.zeros(n, dtype=np.float32)
    ofi_velocity[1:] = ofi_raw[1:] - ofi_raw[:-1]
    
    ofi_volatility = np.zeros(n, dtype=np.float32)
    window = 10
    if n > window:
        ofi_volatility[window:] = np.std(
            np.lib.stride_tricks.sliding_window_view(ofi_raw, window), axis=1
        )
    
    derived = np.column_stack([
        midprice, imbalance, cumspread,
        ofi_raw, ofi_ewm, ofi_velocity, ofi_volatility
    ]).astype(np.float32)
    
    return derived, midprice


def load_and_process_file(file_path: str, feature_cols: list, label_cols: list) -> tuple:
    """加载并处理单个文件 - 用于多进程"""
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        n = len(df)
        
        # 获取实际存在的列
        available_cols = set(df.columns)
        
        # 只使用存在的原始特征列
        raw_cols = [c for c in feature_cols if c not in DERIVED_FEATURE_COLS and c in available_cols]
        
        # 如果缺少某些列，用0填充
        raw_features_list = []
        for col in [c for c in feature_cols if c not in DERIVED_FEATURE_COLS]:
            if col in available_cols:
                raw_features_list.append(df[col].values.astype(np.float32))
            else:
                raw_features_list.append(np.zeros(n, dtype=np.float32))
        
        if len(raw_features_list) == 0:
            print(f"文件 {file_path} 没有有效特征列")
            return None, None, None
            
        raw_features = np.column_stack(raw_features_list)
        
        # 计算衍生特征
        derived, midprice = compute_derived_features_vectorized(df)
        
        # 合并特征
        features = np.hstack([raw_features, derived])
        
        features = clean_features_vectorized(features, feature_cols)
        
        # 检查标签列是否存在
        available_labels = [c for c in label_cols if c in available_cols]
        if len(available_labels) != len(label_cols):
            missing = set(label_cols) - set(available_labels)
            print(f"文件 {file_path} 缺少标签列: {missing}")
            return None, None, None
            
        labels = df[label_cols].values.astype(np.int64)
        
        midprice = np.nan_to_num(midprice, nan=0.0, posinf=0.0, neginf=0.0)
        midprice = np.clip(midprice, -0.3, 0.3)
        
        true_returns = np.zeros((len(df), len(WINDOW_SIZES)), dtype=np.float32)
        for i, w in enumerate(WINDOW_SIZES):
            if w < len(df):
                safe_midprice = np.where(np.abs(midprice[:-w]) < 1e-8, 1e-8, midprice[:-w])
                true_returns[:-w, i] = (midprice[w:] - midprice[:-w]) / safe_midprice
        
        true_returns = np.nan_to_num(true_returns, nan=0.0, posinf=0.0, neginf=0.0)
        true_returns = np.clip(true_returns, -0.1, 0.1)
        
        return features, labels, true_returns
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return None, None, None


class CachedHFTDataset(Dataset):
    """缓存式数据集 - 预加载所有数据到内存
    
    优点：
    1. 避免重复IO
    2. 支持多进程DataLoader
    3. 随机访问更快
    """
    
    def __init__(self, data_dir: str, feature_cols: list, label_cols: list,
                 seq_len: int = 100, file_list: list = None,
                 mean: np.ndarray = None, std: np.ndarray = None,
                 cache_dir: str = None, force_reload: bool = False):
        self.seq_len = seq_len
        self.mean = mean
        self.std = std
        
        if file_list is not None:
            self.all_files = file_list
        else:
            self.all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
        
        self.cache_dir = cache_dir
        self.samples = []
        self.labels = []
        self.returns = []
        
        cache_file = os.path.join(cache_dir, 'dataset_cache.npz') if cache_dir else None
        
        if cache_file and os.path.exists(cache_file) and not force_reload:
            print(f"从缓存加载数据: {cache_file}")
            cache_data = np.load(cache_file, allow_pickle=True)
            self.samples = cache_data['samples']
            self.labels = cache_data['labels']
            self.returns = cache_data['returns']
        else:
            print(f"加载 {len(self.all_files)} 个文件...")
            for i, file in enumerate(tqdm(self.all_files, desc="加载数据")):
                file_path = os.path.join(data_dir, file)
                features, labels, true_returns = load_and_process_file(
                    file_path, feature_cols, label_cols
                )
                
                if features is None:
                    continue
                
                if self.mean is not None and self.std is not None:
                    features = (features - self.mean) / self.std
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                max_window = max(WINDOW_SIZES)
                valid_end = len(features) - max_window
                
                for j in range(seq_len, valid_end):
                    self.samples.append(features[j - seq_len:j])
                    self.labels.append(labels[j])
                    self.returns.append(true_returns[j])
                
                if (i + 1) % 100 == 0:
                    gc.collect()
            
            self.samples = np.array(self.samples, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)
            self.returns = np.array(self.returns, dtype=np.float32)
            
            if cache_file:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                np.savez(cache_file, samples=self.samples, labels=self.labels, returns=self.returns)
                print(f"数据已缓存到: {cache_file}")
        
        print(f"数据集大小: {len(self.samples)} 样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.samples[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(self.returns[idx])
        )


class StreamingHFTDatasetOptimized(torch.utils.data.IterableDataset):
    """优化的流式数据集 - 支持多worker预取
    
    优化点：
    1. 每个worker处理不同的文件
    2. 使用生成器减少内存
    3. 向量化特征计算
    """
    
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
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self.all_files
        else:
            per_worker = int(np.ceil(len(self.all_files) / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.all_files))
            files = self.all_files[start:end]
        
        if self.shuffle_files:
            import random
            random.shuffle(files)
        
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            features, labels, true_returns = load_and_process_file(
                file_path, self.feature_cols, self.label_cols
            )
            
            if features is None:
                continue
            
            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            max_window = max(WINDOW_SIZES)
            valid_end = len(features) - max_window
            
            if valid_end <= self.seq_len:
                continue
            
            for i in range(self.seq_len, valid_end):
                yield (
                    torch.from_numpy(features[i - self.seq_len:i]),
                    torch.from_numpy(labels[i]),
                    torch.from_numpy(true_returns[i])
                )


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


def compute_stats_fast(data_dir: str, feature_cols: list, num_files: int = None) -> tuple:
    """快速计算归一化统计量 - 向量化版本"""
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
    if num_files:
        all_files = all_files[:num_files]
    
    print(f"计算统计量（使用 {len(all_files)} 个文件）...")
    
    all_features = []
    
    for i, file in enumerate(tqdm(all_files, desc="计算统计量")):
        file_path = os.path.join(data_dir, file)
        features, _, _ = load_and_process_file(file_path, feature_cols, LABEL_COLS)
        
        if features is not None:
            all_features.append(features)
        
        if (i + 1) % 200 == 0:
            gc.collect()
    
    if len(all_features) == 0:
        return np.zeros(len(feature_cols)), np.ones(len(feature_cols))
    
    all_features = np.vstack(all_features)
    
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0) + 1e-8
    
    print(f"统计量计算完成，共 {len(all_features)} 个样本")
    return mean.astype(np.float32), std.astype(np.float32)


def train_epoch_optimized(model, dataloader, criterion, optimizer, device, scaler,
                          max_grad_norm: float = 0.3, accumulation_steps: int = 4):
    """优化训练epoch - 混合精度 + 梯度累积"""
    model.train()
    total_loss = 0
    total_ce = 0
    total_return_loss = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="训练中")
    for batch_idx, (features, labels, true_returns) in enumerate(pbar):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        true_returns = true_returns.to(device, non_blocking=True)
        
        with autocast(device_type='cuda'):
            logits, return_pred = model(features)
            loss_dict = criterion(logits, return_pred, labels, true_returns, list(model.parameters()))
            loss = loss_dict['loss'] / accumulation_steps
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[警告] batch {batch_idx} 出现 NaN/Inf，跳过")
            optimizer.zero_grad()
            continue
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss_dict['loss'].item()
        total_ce += loss_dict['ce_loss']
        total_return_loss += loss_dict['return_loss']
        num_batches += 1
        
        if batch_idx % 50 == 0:
            metrics = compute_trading_metrics(logits, labels, true_returns)
            pbar.set_postfix({
                'loss': f'{loss_dict["loss"].item():.4f}',
                'trade': f'{metrics["trade_rate"]:.2f}',
                'ret': f'{metrics["cumulative_return"]:.4f}',
            })
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'ce_loss': total_ce / num_batches if num_batches > 0 else 0,
        'return_loss': total_return_loss / num_batches if num_batches > 0 else 0,
    }


def evaluate_optimized(model, dataloader, criterion, device):
    """优化评估"""
    model.eval()
    all_logits = []
    all_labels = []
    all_returns = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for features, labels, true_returns in tqdm(dataloader, desc="评估中"):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            true_returns = true_returns.to(device, non_blocking=True)
            
            with autocast(device_type='cuda'):
                logits, return_pred = model(features)
                loss_dict = criterion(logits, return_pred, labels, true_returns)
            
            total_loss += loss_dict['loss'].item()
            num_batches += 1
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_returns.append(true_returns.cpu())
    
    if len(all_logits) == 0:
        return {'loss': 0, 'cumulative_return': 0, 'single_return': 0, 'trade_rate': 0}
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_returns = torch.cat(all_returns)
    
    metrics = compute_trading_metrics(all_logits, all_labels, all_returns)
    window_metrics = compute_window_metrics(all_logits, all_labels, all_returns)
    metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0
    metrics['window_metrics'] = window_metrics
    
    return metrics


def setup_cuda_optimizations():
    """设置CUDA优化"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        print("CUDA优化已启用:")
        print(f"  - cudnn.benchmark: True")
        print(f"  - TF32: True")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description='T-KAN Pro 训练脚本 - V100优化版')
    parser.add_argument('--data_dir', type=str, default='/root/2026train_set/2026train_set')
    parser.add_argument('--output_dir', type=str, default='/root/submission')
    parser.add_argument('--cache_dir', type=str, default='/root/cache')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--fee', type=float, default=0.0002)
    parser.add_argument('--lambda_return', type=float, default=0.5)
    parser.add_argument('--lambda_trade', type=float, default=0.1)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_cache', action='store_true', help='使用缓存数据集')
    parser.add_argument('--force_reload', action='store_true', help='强制重新加载数据')
    parser.add_argument('--compile', action='store_true', help='使用torch.compile')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    setup_cuda_optimizations()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("计算归一化统计量...")
    print("=" * 60)
    mean, std = compute_stats_fast(args.data_dir, ALL_FEATURE_COLS, num_files=args.max_files)
    
    print("\n" + "=" * 60)
    print("创建数据加载器...")
    print("=" * 60)
    
    all_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.parquet')])
    if args.max_files:
        all_files = all_files[:args.max_files]
    
    total = len(all_files)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]
    
    print(f"训练文件: {len(train_files)}, 验证文件: {len(val_files)}, 测试文件: {len(test_files)}")
    
    if args.use_cache:
        train_dataset = CachedHFTDataset(
            args.data_dir, ALL_FEATURE_COLS, LABEL_COLS,
            file_list=train_files, mean=mean, std=std,
            cache_dir=os.path.join(args.cache_dir, 'train'),
            force_reload=args.force_reload
        )
        val_dataset = CachedHFTDataset(
            args.data_dir, ALL_FEATURE_COLS, LABEL_COLS,
            file_list=val_files, mean=mean, std=std,
            cache_dir=os.path.join(args.cache_dir, 'val'),
            force_reload=args.force_reload
        )
        test_dataset = CachedHFTDataset(
            args.data_dir, ALL_FEATURE_COLS, LABEL_COLS,
            file_list=test_files, mean=mean, std=std,
            cache_dir=os.path.join(args.cache_dir, 'test'),
            force_reload=args.force_reload
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
    else:
        train_dataset = StreamingHFTDatasetOptimized(
            args.data_dir, ALL_FEATURE_COLS, LABEL_COLS,
            file_list=train_files, mean=mean, std=std
        )
        val_dataset = StreamingHFTDatasetOptimized(
            args.data_dir, ALL_FEATURE_COLS, LABEL_COLS,
            file_list=val_files, mean=mean, std=std, shuffle_files=False
        )
        test_dataset = StreamingHFTDatasetOptimized(
            args.data_dir, ALL_FEATURE_COLS, LABEL_COLS,
            file_list=test_files, mean=mean, std=std, shuffle_files=False
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, prefetch_factor=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, prefetch_factor=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, prefetch_factor=2
        )
    
    input_dim = len(ALL_FEATURE_COLS)
    print(f"输入维度: {input_dim}")
    
    print("\n" + "=" * 60)
    print("创建模型...")
    print("=" * 60)
    model = create_model(input_dim, num_windows=5, dropout=0.15).to(device)
    print(f"模型参数量: {count_parameters(model):,}")
    
    if args.compile and hasattr(torch, 'compile'):
        print("使用 torch.compile 优化模型...")
        model = torch.compile(model, mode='reduce-overhead')
    
    criterion = CompositeProfitLoss(
        fee=args.fee,
        lambda_return=args.lambda_return,
        lambda_trade=args.lambda_trade,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8, fused=True
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    
    scaler = GradScaler()
    ema = EMA(model, decay=args.ema_decay)
    
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    print(f"配置: batch_size={args.batch_size}, accumulation={args.accumulation_steps}, "
          f"effective_batch={args.batch_size * args.accumulation_steps}")
    
    best_val_return = float('-inf')
    best_window = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        if epoch < args.warmup_epochs:
            lr_mult = (epoch + 1) / args.warmup_epochs
            current_lr = args.lr * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"Warm-up: lr={current_lr:.2e} (mult={lr_mult:.2f})")
        else:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.2e}")
        
        train_metrics = train_epoch_optimized(
            model, train_loader, criterion, optimizer, device, scaler,
            args.max_grad_norm, args.accumulation_steps
        )
        ema.update(model)
        
        val_metrics = evaluate_optimized(model, val_loader, criterion, device)
        
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        print(f"训练损失: {train_metrics['loss']:.4f}")
        print(f"验证: 累计收益={val_metrics['cumulative_return']:.6f}, "
              f"单次收益={val_metrics['single_return']:.6f}, "
              f"交易率={val_metrics['trade_rate']:.3f}")
        
        if 'window_metrics' in val_metrics:
            print("各窗口收益:")
            for w_name, w_metrics in val_metrics['window_metrics'].items():
                print(f"  {w_name}: 累计={w_metrics['cumulative_return']:.6f}, "
                      f"单次={w_metrics['single_return']:.6f}")
        
        max_window_return = float('-inf')
        max_window_name = None
        if 'window_metrics' in val_metrics:
            for w_name, w_metrics in val_metrics['window_metrics'].items():
                if w_metrics['cumulative_return'] > max_window_return:
                    max_window_return = w_metrics['cumulative_return']
                    max_window_name = w_name
        
        if max_window_return > best_val_return:
            best_val_return = max_window_return
            best_window = max_window_name
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'ema_state': ema.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_return': best_val_return,
                'best_window': best_window,
                'config': {'input_dim': input_dim, 'num_windows': 5},
                'mean': mean,
                'std': std,
                'feature_cols': ALL_FEATURE_COLS,
                'raw_feature_cols': RAW_FEATURE_COLS,
                'derived_feature_cols': DERIVED_FEATURE_COLS,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"✓ 保存最佳模型: Epoch {epoch + 1}, 窗口={best_window}, 收益={best_val_return:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停: 验证收益连续 {args.patience} 轮未提升")
                break
    
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    
    test_metrics = evaluate_optimized(model, test_loader, criterion, device)
    print(f"测试累计收益: {test_metrics['cumulative_return']:.6f}")
    print(f"测试单次收益: {test_metrics['single_return']:.6f}")
    print(f"测试交易率: {test_metrics['trade_rate']:.3f}")
    
    config_data = {
        'python_version': '3.10',
        'batch': args.batch_size,
        'feature': RAW_FEATURE_COLS,
        'label': LABEL_COLS,
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(args.output_dir, 'requirements.txt'), 'w', encoding='utf-8') as f:
        f.write('torch>=2.0.0\npandas>=2.0.0\nnumpy>=1.24.0\npyarrow>=10.0.0\n')
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
