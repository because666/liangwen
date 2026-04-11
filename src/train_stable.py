"""
T-KAN Pro 训练脚本 - 稳定版

基于已验证可工作的数据处理逻辑，
只添加必要的性能优化（AMP、CUDA等）
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
    """特征清洗"""
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


def compute_derived_features(df: pd.DataFrame) -> tuple:
    """计算衍生特征"""
    n = len(df)
    
    bid1 = df['bid1'].values.astype(np.float32)
    ask1 = df['ask1'].values.astype(np.float32)
    
    midprice = np.zeros(n, dtype=np.float32)
    both_nonzero = (bid1 != 0) & (ask1 != 0)
    bid_zero = (bid1 == 0) & (ask1 != 0)
    ask_zero = (ask1 == 0) & (bid1 != 0)
    midprice[both_nonzero] = (bid1[both_nonzero] + ask1[both_nonzero]) / 2
    midprice[bid_zero] = ask1[bid_zero]
    midprice[ask_zero] = bid1[ask_zero]
    
    total_bsize = sum(df[f'bsize{i}'].values for i in range(1, 11)).astype(np.float32)
    total_asize = sum(df[f'asize{i}'].values for i in range(1, 11)).astype(np.float32)
    total_size = total_bsize + total_asize
    
    imbalance = np.zeros(n, dtype=np.float32)
    mask = total_size > 0
    imbalance[mask] = (total_bsize[mask] - total_asize[mask]) / total_size[mask]
    
    cumspread = sum(
        df[f'ask{i}'].values - df[f'bid{i}'].values 
        for i in range(1, 11)
    ).astype(np.float32)
    
    mb_intst = df['mb_intst'].values.astype(np.float32)
    ma_intst = df['ma_intst'].values.astype(np.float32)
    ofi_raw = mb_intst - ma_intst
    
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
        for i in range(window, n):
            ofi_volatility[i] = np.std(ofi_raw[i-window:i])
    
    derived = np.column_stack([
        midprice, imbalance, cumspread,
        ofi_raw, ofi_ewm, ofi_velocity, ofi_volatility
    ]).astype(np.float32)
    
    return derived, midprice


def load_and_process_file(file_path: str) -> tuple:
    """加载并处理单个文件"""
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        
        # 检查必要列是否存在
        required_cols = RAW_FEATURE_COLS + LABEL_COLS
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if len(missing_cols) > 0:
            print(f"文件 {os.path.basename(file_path)} 缺少列: {missing_cols[:5]}...")
            return None
        
        # 计算衍生特征
        derived, midprice = compute_derived_features(df)
        
        # 提取原始特征（只取存在的列）
        raw_features_list = []
        for col in RAW_FEATURE_COLS:
            if col in df.columns:
                raw_features_list.append(df[col].values.astype(np.float32))
            else:
                raw_features_list.append(np.zeros(len(df), dtype=np.float32))
        
        raw_features = np.column_stack(raw_features_list)
        
        # 合并特征
        features = np.hstack([raw_features, derived])
        features = clean_features(features, ALL_FEATURE_COLS)
        
        # 标签
        labels = df[LABEL_COLS].values.astype(np.int64)
        
        # 真实收益
        midprice = np.nan_to_num(midprice, nan=0.0, posinf=0.0, neginf=0.0)
        midprice = np.clip(midprice, -0.3, 0.3)
        
        true_returns = np.zeros((len(df), len(WINDOW_SIZES)), dtype=np.float32)
        for i, w in enumerate(WINDOW_SIZES):
            if w < len(df):
                safe_mid = np.where(np.abs(midprice[:-w]) < 1e-8, 1e-8, midprice[:-w])
                true_returns[:-w, i] = (midprice[w:] - midprice[:-w]) / safe_mid
        
        true_returns = np.nan_to_num(true_returns, nan=0.0, posinf=0.0, neginf=0.0)
        true_returns = np.clip(true_returns, -0.1, 0.1)
        
        return features, labels, true_returns
        
    except Exception as e:
        print(f"加载失败 {os.path.basename(file_path)}: {e}")
        return None


class HFTDataset(Dataset):
    """HFT数据集"""
    
    def __init__(self, data_dir: str, seq_len: int = 100, file_list: list = None,
                 mean: np.ndarray = None, std: np.ndarray = None,
                 max_files: int = None):
        self.seq_len = seq_len
        self.mean = mean
        self.std = std
        
        if file_list is not None:
            self.all_files = file_list
        else:
            self.all_files = sorted([
                f for f in os.listdir(data_dir) if f.endswith('.parquet')
            ])
        
        if max_files:
            self.all_files = self.all_files[:max_files]
        
        print(f"\n加载数据集 ({len(self.all_files)} 个文件)...")
        
        self.samples = []
        self.labels = []
        self.returns = []
        
        success_count = 0
        fail_count = 0
        
        for i, file in enumerate(tqdm(self.all_files, desc="加载数据")):
            file_path = os.path.join(data_dir, file)
            result = load_and_process_file(file_path)
            
            if result is None:
                fail_count += 1
                continue
            
            features, labels, true_returns = result
            
            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            max_window = max(WINDOW_SIZES)
            valid_end = len(features) - max_window
            
            for j in range(self.seq_len, valid_end):
                self.samples.append(features[j - self.seq_len:j])
                self.labels.append(labels[j])
                self.returns.append(true_returns[j])
            
            success_count += 1
            
            if (i + 1) % 100 == 0:
                gc.collect()
        
        if len(self.samples) > 0:
            self.samples = np.array(self.samples, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)
            self.returns = np.array(self.returns, dtype=np.float32)
        else:
            self.samples = np.array([], dtype=np.float32).reshape(0, self.seq_len, len(ALL_FEATURE_COLS))
            self.labels = np.array([], dtype=np.int64).reshape(0, len(LABEL_COLS))
            self.returns = np.array([], dtype=np.float32).reshape(0, len(WINDOW_SIZES))
        
        print(f"数据加载完成: 成功={success_count}, 失败={fail_count}, 样本数={len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.samples[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(self.returns[idx])
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


def compute_stats(data_dir: str, num_files: int = None) -> tuple:
    """计算归一化统计量"""
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
    if num_files:
        all_files = all_files[:num_files]
    
    print(f"计算统计量 ({len(all_files)} 个文件)...")
    
    all_features = []
    
    for i, file in enumerate(tqdm(all_files, desc="统计量")):
        result = load_and_process_file(os.path.join(data_dir, file))
        
        if result is not None:
            features, _, _ = result
            all_features.append(features)
        
        if (i + 1) % 200 == 0:
            gc.collect()
    
    if len(all_features) == 0:
        return np.zeros(len(ALL_FEATURE_COLS)), np.ones(len(ALL_FEATURE_COLS))
    
    all_features = np.vstack(all_features)
    
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0) + 1e-8
    
    print(f"统计量完成: {len(all_features)} 样本")
    return mean.astype(np.float32), std.astype(np.float32)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler,
               max_grad_norm: float = 0.3, accumulation_steps: int = 4):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_ce = 0
    total_return_loss = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="训练")
    for batch_idx, (features, labels, true_returns) in enumerate(pbar):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        true_returns = true_returns.to(device, non_blocking=True)
        
        if torch.cuda.is_available():
            with autocast():
                logits, return_pred = model(features)
                loss_dict = criterion(logits, return_pred, labels, true_returns, list(model.parameters()))
                loss = loss_dict['loss'] / accumulation_steps
        else:
            logits, return_pred = model(features)
            loss_dict = criterion(logits, return_pred, labels, true_returns, list(model.parameters()))
            loss = loss_dict['loss'] / accumulation_steps
        
        if torch.isnan(loss) or torch.isinf(loss):
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
        'loss': total_loss / max(num_batches, 1),
        'ce_loss': total_ce / max(num_batches, 1),
        'return_loss': total_return_loss / max(num_batches, 1),
    }


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    all_logits = []
    all_labels = []
    all_returns = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for features, labels, true_returns in tqdm(dataloader, desc="评估"):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            true_returns = true_returns.to(device, non_blocking=True)
            
            if torch.cuda.is_available():
                with autocast():
                    logits, return_pred = model(features)
                    loss_dict = criterion(logits, return_pred, labels, true_returns)
            else:
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
    metrics['loss'] = total_loss / max(num_batches, 1)
    metrics['window_metrics'] = window_metrics
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='T-KAN Pro Training')
    parser.add_argument('--data_dir', type=str, default='2026train_set/2026train_set')
    parser.add_argument('--output_dir', type=str, default='submission')
    parser.add_argument('--cache_dir', type=str, default=None)
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
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("计算归一化统计量")
    print("=" * 60)
    mean, std = compute_stats(args.data_dir, num_files=args.max_files)
    
    print("\n" + "=" * 60)
    print("创建数据集")
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
    
    print(f"训练: {len(train_files)}, 验证: {len(val_files)}, 测试: {len(test_files)}")
    
    train_dataset = HFTDataset(
        args.data_dir, file_list=train_files, mean=mean, std=std
    )
    val_dataset = HFTDataset(
        args.data_dir, file_list=val_files, mean=mean, std=std
    )
    test_dataset = HFTDataset(
        args.data_dir, file_list=test_files, mean=mean, std=std
    )
    
    if len(train_dataset) == 0:
        print("错误：训练集为空！请检查数据和路径")
        return
    
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
    
    input_dim = len(ALL_FEATURE_COLS)
    print(f"输入维度: {input_dim}")
    
    print("\n" + "=" * 60)
    print("创建模型")
    print("=" * 60)
    model = create_model(input_dim, num_windows=5, dropout=0.15).to(device)
    print(f"参数量: {count_parameters(model):,}")
    
    criterion = CompositeProfitLoss(
        fee=args.fee,
        lambda_return=args.lambda_return,
        lambda_trade=args.lambda_trade,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    
    scaler = GradScaler()
    ema = EMA(model, decay=args.ema_decay)
    
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
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
            print(f"Warmup: lr={current_lr:.2e}")
        else:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"学习率: {current_lr:.2e}")
        
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            args.max_grad_norm, args.accumulation_steps
        )
        ema.update(model)
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        print(f"训练损失: {train_metrics['loss']:.4f}")
        print(f"验证: 收益={val_metrics['cumulative_return']:.6f}, "
              f"单次={val_metrics['single_return']:.6f}, "
              f"交易率={val_metrics['trade_rate']:.3f}")
        
        if 'window_metrics' in val_metrics:
            for w_name, w_m in val_metrics['window_metrics'].items():
                print(f"  {w_name}: {w_m['cumulative_return']:.6f}")
        
        max_window_return = float('-inf')
        max_window_name = None
        if 'window_metrics' in val_metrics:
            for w_name, w_m in val_metrics['window_metrics'].items():
                if w_m['cumulative_return'] > max_window_return:
                    max_window_return = w_m['cumulative_return']
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
            print(f"✓ 保存最佳: Epoch {epoch+1}, 窗口={best_window}, 收益={best_val_return:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停: 连续{args.patience}轮未提升")
                break
    
    print("\n" + "=" * 60)
    print("测试评估")
    print("=" * 60)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"测试收益: {test_metrics['cumulative_return']:.6f}")
    print(f"测试单次: {test_metrics['single_return']:.6f}")
    print(f"测试交易率: {test_metrics['trade_rate']:.3f}")
    
    config_data = {
        'python_version': '3.10',
        'batch': args.batch_size,
        'feature': RAW_FEATURE_COLS,
        'label': LABEL_COLS,
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(args.output_dir, 'requirements.txt'), 'w') as f:
        f.write('torch>=2.0.0\npandas>=2.0.0\nnumpy>=1.24.0\npyarrow>=10.0.0\n')
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
