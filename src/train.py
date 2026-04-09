"""
T-KAN Pro 训练脚本

训练策略：
1. Warm-up: 5 epochs, lr: 1e-5 → 3e-5
2. 主训练: 15 epochs, lr: 3e-5 → 1e-5
3. 微调: 5 epochs, lr: 1e-6

数值稳定性保障：
- 梯度裁剪: max_grad_norm = 0.3
- 更保守的学习率
- Warm-up + Cosine
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
from tqdm import tqdm
from copy import deepcopy

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
    features = features.copy()
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


def clean_returns(returns: np.ndarray) -> np.ndarray:
    """清洗收益率数据"""
    returns = returns.copy()
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    returns = np.clip(returns, -0.1, 0.1)
    return returns


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算衍生特征"""
    df = df.copy()
    n = len(df)
    
    bid1 = df['bid1'].values if 'bid1' in df.columns else np.zeros(n)
    ask1 = df['ask1'].values if 'ask1' in df.columns else np.zeros(n)
    midprice = np.zeros(n)
    both = (bid1 != 0) & (ask1 != 0)
    bid0 = (bid1 == 0) & (ask1 != 0)
    ask0 = (ask1 == 0) & (bid1 != 0)
    midprice[both] = (bid1[both] + ask1[both]) / 2
    midprice[bid0] = ask1[bid0]
    midprice[ask0] = bid1[ask0]
    df['midprice'] = midprice
    
    total_b = np.zeros(n)
    total_a = np.zeros(n)
    for i in range(1, 11):
        if f'bsize{i}' in df.columns:
            total_b += df[f'bsize{i}'].values
        if f'asize{i}' in df.columns:
            total_a += df[f'asize{i}'].values
    total = total_b + total_a
    imbalance = np.zeros(n)
    mask = total > 0
    imbalance[mask] = (total_b[mask] - total_a[mask]) / total[mask]
    df['imbalance'] = imbalance
    
    cumspread = np.zeros(n)
    for i in range(1, 11):
        if f'ask{i}' in df.columns and f'bid{i}' in df.columns:
            cumspread += df[f'ask{i}'].values - df[f'bid{i}'].values
    df['cumspread'] = cumspread
    
    mb = df['mb_intst'].values if 'mb_intst' in df.columns else np.zeros(n)
    ma = df['ma_intst'].values if 'ma_intst' in df.columns else np.zeros(n)
    ofi_raw = mb - ma
    df['ofi_raw'] = ofi_raw
    
    ofi_ewm = np.zeros(n)
    alpha = 0.1
    for i in range(n):
        ofi_ewm[i] = ofi_raw[i] if i == 0 else (1 - alpha) * ofi_ewm[i - 1] + alpha * ofi_raw[i]
    df['ofi_ewm'] = ofi_ewm
    
    ofi_velocity = np.zeros(n)
    ofi_velocity[1:] = ofi_raw[1:] - ofi_raw[:-1]
    df['ofi_velocity'] = ofi_velocity
    
    ofi_volatility = np.zeros(n)
    for i in range(10, n):
        ofi_volatility[i] = np.std(ofi_raw[i - 10:i])
    df['ofi_volatility'] = ofi_volatility
    
    return df


class StreamingHFTDataset(IterableDataset):
    """流式高频交易数据集"""
    
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
                df = pd.read_parquet(file_path)
                df = compute_derived_features(df)
                
                for col in self.feature_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                
                features = df[self.feature_cols].values.astype(np.float32)
                labels = df[self.label_cols].values.astype(np.int64)
                
                features = clean_features(features, self.feature_cols)
                
                midprice = df['midprice'].values if 'midprice' in df.columns else np.zeros(len(df))
                midprice = np.nan_to_num(midprice, nan=0.0, posinf=0.0, neginf=0.0)
                midprice = np.clip(midprice, -0.3, 0.3)
                
                true_returns = np.zeros((len(df), len(WINDOW_SIZES)), dtype=np.float32)
                for i, w in enumerate(WINDOW_SIZES):
                    if w < len(df):
                        safe_midprice = np.where(np.abs(midprice[:-w]) < 1e-8, 1e-8, midprice[:-w])
                        true_returns[:-w, i] = (midprice[w:] - midprice[:-w]) / safe_midprice
                
                true_returns = clean_returns(true_returns)
                
                if self.mean is not None and self.std is not None:
                    features = (features - self.mean) / self.std
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                max_window = max(WINDOW_SIZES)
                valid_end = len(df) - max_window
                if valid_end <= self.seq_len:
                    continue
                
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


def compute_stats(data_dir: str, feature_cols: list, num_files: int = None) -> tuple:
    """增量计算归一化统计量"""
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
    if num_files:
        all_files = all_files[:num_files]
    
    print(f"计算统计量（使用 {len(all_files)} 个文件）...")
    n = 0
    mean = None
    M2 = None
    
    for i, file in enumerate(all_files):
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_parquet(file_path)
            df = compute_derived_features(df)
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            features = df[feature_cols].values.astype(np.float32)
            
            features = clean_features(features, feature_cols)
            
            if mean is None:
                mean = np.zeros(features.shape[1])
                M2 = np.zeros(features.shape[1])
            
            for j in range(len(features)):
                n += 1
                delta = features[j] - mean
                mean += delta / n
                delta2 = features[j] - mean
                M2 += delta * delta2
            
            if (i + 1) % 200 == 0:
                print(f"  已处理 {i + 1}/{len(all_files)} 个文件...")
        except Exception as e:
            print(f"  处理文件 {file} 失败: {e}")
            continue
    
    if n == 0:
        return np.zeros(len(feature_cols)), np.ones(len(feature_cols))
    
    std = np.sqrt(M2 / n) + 1e-8
    print(f"统计量计算完成，共使用 {n} 个样本")
    return mean, std


def train_epoch(model, dataloader, criterion, optimizer, device,
                max_grad_norm: float = 0.3, max_batches: int = None):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_ce = 0
    total_return_loss = 0
    num_batches = 0
    nan_count = 0
    
    pbar = tqdm(dataloader, desc="训练中")
    for batch_idx, (features, labels, true_returns) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
        
        features = features.to(device).float()
        labels = labels.to(device).long()
        true_returns = true_returns.to(device).float()
        
        optimizer.zero_grad()
        
        logits, return_pred = model(features)
        loss_dict = criterion(logits, return_pred, labels, true_returns, list(model.parameters()))
        loss = loss_dict['loss']
        
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 3:
                print(f"\n[警告] batch {batch_idx} 出现 NaN/Inf，跳过此 batch (第{nan_count}次)")
                optimizer.zero_grad()
                continue
            else:
                print(f"\n[错误] 连续出现 {nan_count} 次 NaN，停止训练")
                break
        
        loss.backward()
        
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
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        if grad_norm > 10.0:
            print(f"\n[警告] batch {batch_idx} 梯度范数过大: {grad_norm:.4f}")
        
        optimizer.step()
        
        nan_count = 0
        
        total_loss += loss.item()
        total_ce += loss_dict['ce_loss']
        total_return_loss += loss_dict['return_loss']
        num_batches += 1
        
        if batch_idx % 100 == 0:
            metrics = compute_trading_metrics(logits, labels, true_returns)
            print(f"\n[调试] batch {batch_idx}: loss={loss.item():.4f}, "
                  f"ce={loss_dict['ce_loss']:.4f}, "
                  f"return_loss={loss_dict['return_loss']:.4f}, "
                  f"grad_norm={grad_norm:.4f}")
            print(f"        交易率={metrics['trade_rate']:.3f}, "
                  f"累计收益={metrics['cumulative_return']:.6f}")
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{loss_dict["ce_loss"]:.4f}',
            'ret': f'{loss_dict["return_loss"]:.4f}',
        })
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'ce_loss': total_ce / num_batches if num_batches > 0 else 0,
        'return_loss': total_return_loss / num_batches if num_batches > 0 else 0,
    }


def evaluate(model, dataloader, criterion, device, max_batches: int = None):
    """评估模型"""
    model.eval()
    all_logits = []
    all_labels = []
    all_returns = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (features, labels, true_returns) in enumerate(tqdm(dataloader, desc="评估中")):
            if max_batches and batch_idx >= max_batches:
                break
            
            features = features.to(device).float()
            labels = labels.to(device).long()
            true_returns = true_returns.to(device).float()
            
            logits, return_pred = model(features)
            loss_dict = criterion(logits, return_pred, labels, true_returns)
            
            total_loss += loss_dict['loss'].item() if isinstance(loss_dict['loss'], torch.Tensor) else loss_dict['loss']
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


def main():
    parser = argparse.ArgumentParser(description='T-KAN Pro 训练脚本')
    parser.add_argument('--data_dir', type=str,
                        default='d:/量化/良文杯/2026train_set/2026train_set')
    parser.add_argument('--output_dir', type=str,
                        default='d:/量化/良文杯/submission')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--max_batches_per_epoch', type=int, default=None)
    parser.add_argument('--fee', type=float, default=0.0002)
    parser.add_argument('--lambda_return', type=float, default=0.5)
    parser.add_argument('--lambda_trade', type=float, default=0.1)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("计算归一化统计量...")
    print("=" * 50)
    mean, std = compute_stats(args.data_dir, ALL_FEATURE_COLS)
    
    print("\n" + "=" * 50)
    print("创建数据加载器...")
    print("=" * 50)
    
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
    
    train_dataset = StreamingHFTDataset(args.data_dir, ALL_FEATURE_COLS, LABEL_COLS,
                                         file_list=train_files, mean=mean, std=std)
    val_dataset = StreamingHFTDataset(args.data_dir, ALL_FEATURE_COLS, LABEL_COLS,
                                       file_list=val_files, mean=mean, std=std, shuffle_files=False)
    test_dataset = StreamingHFTDataset(args.data_dir, ALL_FEATURE_COLS, LABEL_COLS,
                                        file_list=test_files, mean=mean, std=std, shuffle_files=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    input_dim = len(ALL_FEATURE_COLS)
    print(f"输入维度: {input_dim}")
    
    print("\n" + "=" * 50)
    print("创建模型...")
    print("=" * 50)
    model = create_model(input_dim, num_windows=5, dropout=0.15).to(device)
    print(f"模型参数量: {count_parameters(model):,}")
    
    criterion = CompositeProfitLoss(
        fee=args.fee,
        lambda_return=args.lambda_return,
        lambda_trade=args.lambda_trade,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    
    ema = EMA(model, decay=args.ema_decay)
    
    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50)
    
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
            print(f"Warm-up 阶段: 学习率 = {current_lr:.2e} (乘数: {lr_mult:.2f})")
        else:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.2e}")
        
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            args.max_grad_norm, args.max_batches_per_epoch
        )
        ema.update(model)
        
        val_metrics = evaluate(model, val_loader, criterion, device, args.max_batches_per_epoch)
        
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        print(f"训练损失: {train_metrics['loss']:.4f}")
        print(f"验证累计收益: {val_metrics['cumulative_return']:.6f}, "
              f"单次收益: {val_metrics['single_return']:.6f}, "
              f"交易率: {val_metrics['trade_rate']:.3f}")
        
        if 'window_metrics' in val_metrics:
            print("各窗口收益:")
            for w_name, w_metrics in val_metrics['window_metrics'].items():
                print(f"  {w_name}: 累计收益={w_metrics['cumulative_return']:.6f}, "
                      f"单次收益={w_metrics['single_return']:.6f}, "
                      f"交易率={w_metrics['trade_rate']:.3f}")
        
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
                'config': {
                    'input_dim': input_dim,
                    'num_windows': 5,
                },
                'mean': mean,
                'std': std,
                'feature_cols': ALL_FEATURE_COLS,
                'raw_feature_cols': RAW_FEATURE_COLS,
                'derived_feature_cols': DERIVED_FEATURE_COLS,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"保存最佳模型, Epoch {epoch + 1}, 最佳窗口: {best_window}, 验证收益: {best_val_return:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停: 验证收益连续 {args.patience} 轮未提升")
                break
    
    print("\n" + "=" * 50)
    print("测试集评估")
    print("=" * 50)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    
    test_metrics = evaluate(model, test_loader, criterion, device, args.max_batches_per_epoch)
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
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()
