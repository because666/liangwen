"""
T-KAN 编码器回归预训练脚本

训练流程：
1. 加载 parquet 数据，提取 40 维原始价量特征
2. 计算 midprice 及 5 个窗口的变化率作为回归目标
3. 使用 Huber Loss 预训练 T-KAN 编码器
4. 保存编码器权重 + 归一化统计量

使用方式：
    python train_tkan.py --data_dir ../../2026train_set/2026train_set --output_dir output
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

from model import TKANRegressionModel, TKANEncoder, create_regression_model, count_parameters

FEATURE_COLS = [
    'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'bid6', 'bid7', 'bid8', 'bid9', 'bid10',
    'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask6', 'ask7', 'ask8', 'ask9', 'ask10',
    'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'bsize6', 'bsize7', 'bsize8', 'bsize9', 'bsize10',
    'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'asize6', 'asize7', 'asize8', 'asize9', 'asize10',
]

LABEL_COLS = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']

WINDOW_SIZES = [5, 10, 20, 40, 60]

INPUT_DIM = len(FEATURE_COLS)


def compute_midprice(bid1: np.ndarray, ask1: np.ndarray) -> np.ndarray:
    """计算中间价

    参数：
        bid1: 买一价数组
        ask1: 卖一价数组
    返回：
        midprice: 中间价数组
    """
    n = len(bid1)
    midprice = np.zeros(n, dtype=np.float32)
    both = (bid1 != 0) & (ask1 != 0)
    bid0 = (bid1 == 0) & (ask1 != 0)
    ask0 = (ask1 == 0) & (bid1 != 0)
    midprice[both] = (bid1[both] + ask1[both]) / 2
    midprice[bid0] = ask1[bid0]
    midprice[ask0] = bid1[ask0]
    return midprice


def clean_features(features: np.ndarray) -> np.ndarray:
    """特征清洗：处理 NaN/Inf，按列类型裁剪"""
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    for i, col in enumerate(FEATURE_COLS):
        if col.startswith('bid') or col.startswith('ask'):
            features[:, i] = np.clip(features[:, i], -0.3, 0.3)
        elif col.startswith('bsize') or col.startswith('asize'):
            features[:, i] = np.clip(features[:, i], 0, 100)
    return features


def load_and_process_file(file_path: str) -> tuple:
    """加载并处理单个 parquet 文件

    返回：
        (features, labels, return_targets) 或 None
        - features: (n_ticks, 40) 原始特征
        - labels: (n_ticks, 5) 分类标签
        - return_targets: (n_ticks, 5) 回归目标（midprice 变化率）
    """
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')

        missing = [c for c in FEATURE_COLS + LABEL_COLS if c not in df.columns]
        if missing:
            return None

        feature_list = []
        for col in FEATURE_COLS:
            feature_list.append(df[col].values.astype(np.float32))
        features = np.column_stack(feature_list)
        features = clean_features(features)

        labels = df[LABEL_COLS].values.astype(np.int64)

        bid1 = df['bid1'].values.astype(np.float32)
        ask1 = df['ask1'].values.astype(np.float32)
        midprice = compute_midprice(bid1, ask1)
        midprice = np.nan_to_num(midprice, nan=0.0, posinf=0.0, neginf=0.0)
        midprice = np.clip(midprice, -0.3, 0.3)

        return_targets = np.zeros((len(df), len(WINDOW_SIZES)), dtype=np.float32)
        for i, w in enumerate(WINDOW_SIZES):
            if w < len(df):
                return_targets[:-w, i] = midprice[w:] - midprice[:-w]

        return_targets = np.nan_to_num(return_targets, nan=0.0, posinf=0.0, neginf=0.0)
        return_targets = np.clip(return_targets, -0.1, 0.1)

        return features, labels, return_targets

    except Exception as e:
        print(f"加载失败 {os.path.basename(file_path)}: {e}")
        return None


class HFTDataset(Dataset):
    """HFT 数据集（预加载模式）

    参数：
        data_dir: 数据目录
        seq_len: 序列长度
        file_list: 文件列表
        mean: 归一化均值
        std: 归一化标准差
        max_files: 最大文件数
    """

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
        self.return_targets = []

        success_count = 0
        fail_count = 0

        for i, file in enumerate(tqdm(self.all_files, desc="加载数据")):
            file_path = os.path.join(data_dir, file)
            result = load_and_process_file(file_path)

            if result is None:
                fail_count += 1
                continue

            features, labels, return_targets = result

            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            max_window = max(WINDOW_SIZES)
            valid_end = len(features) - max_window

            for j in range(self.seq_len, valid_end):
                self.samples.append(features[j - self.seq_len:j])
                self.labels.append(labels[j])
                self.return_targets.append(return_targets[j])

            success_count += 1

            if (i + 1) % 100 == 0:
                gc.collect()

        if len(self.samples) > 0:
            self.samples = np.array(self.samples, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)
            self.return_targets = np.array(self.return_targets, dtype=np.float32)
        else:
            self.samples = np.array([], dtype=np.float32).reshape(0, self.seq_len, INPUT_DIM)
            self.labels = np.array([], dtype=np.int64).reshape(0, len(LABEL_COLS))
            self.return_targets = np.array([], dtype=np.float32).reshape(0, len(WINDOW_SIZES))

        print(f"数据加载完成: 成功={success_count}, 失败={fail_count}, 样本数={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.samples[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(self.return_targets[idx]),
        )


class EMA:
    """指数移动平均权重"""

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
    """计算归一化统计量（均值和标准差）"""
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
        return np.zeros(INPUT_DIM, dtype=np.float32), np.ones(INPUT_DIM, dtype=np.float32)

    all_features = np.vstack(all_features)
    mean = np.mean(all_features, axis=0).astype(np.float32)
    std = np.std(all_features, axis=0).astype(np.float32) + 1e-8

    print(f"统计量完成: {len(all_features)} 样本")
    return mean, std


def train_epoch(model, dataloader, criterion, optimizer, device, scaler,
                max_grad_norm: float = 1.0, accumulation_steps: int = 4):
    """训练一个 epoch

    参数：
        model: T-KAN 回归模型
        dataloader: 训练数据加载器
        criterion: Huber Loss
        optimizer: 优化器
        device: 计算设备
        scaler: AMP 梯度缩放器
        max_grad_norm: 梯度裁剪阈值
        accumulation_steps: 梯度累积步数
    返回：
        训练指标字典
    """
    model.train()
    total_loss = 0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="训练")
    for batch_idx, (features, labels, return_targets) in enumerate(pbar):
        features = features.to(device, non_blocking=True)
        return_targets = return_targets.to(device, non_blocking=True)

        if torch.cuda.is_available():
            with autocast():
                predictions = model(features)
                loss = criterion(predictions, return_targets) / accumulation_steps
        else:
            predictions = model(features)
            loss = criterion(predictions, return_targets) / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        if batch_idx % 50 == 0:
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.6f}'})

    return {'loss': total_loss / max(num_batches, 1)}


def evaluate(model, dataloader, criterion, device):
    """评估模型

    返回：
        评估指标字典（含每个窗口的回归指标）
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, labels, return_targets in tqdm(dataloader, desc="评估"):
            features = features.to(device, non_blocking=True)
            return_targets = return_targets.to(device, non_blocking=True)

            if torch.cuda.is_available():
                with autocast():
                    predictions = model(features)
                    loss = criterion(predictions, return_targets)
            else:
                predictions = model(features)
                loss = criterion(predictions, return_targets)

            total_loss += loss.item()
            num_batches += 1

            all_preds.append(predictions.cpu())
            all_targets.append(return_targets.cpu())

    if len(all_preds) == 0:
        return {'loss': 0}

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    window_metrics = {}
    for w_idx, w_name in enumerate(LABEL_COLS):
        w_pred = all_preds[:, w_idx]
        w_target = all_targets[:, w_idx]
        mae = (w_pred - w_target).abs().mean().item()
        corr = torch.corrcoef(torch.stack([w_pred, w_target]))[0, 1].item()
        window_metrics[w_name] = {'mae': mae, 'correlation': corr}

    return {
        'loss': total_loss / max(num_batches, 1),
        'window_metrics': window_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='T-KAN 编码器回归预训练')
    parser.add_argument('--data_dir', type=str, default='../../2026train_set/2026train_set')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--huber_delta', type=float, default=0.01)

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
    print("步骤 1: 计算归一化统计量")
    print("=" * 60)
    mean, std = compute_stats(args.data_dir, num_files=args.max_files)

    np.save(os.path.join(args.output_dir, 'feature_mean.npy'), mean)
    np.save(os.path.join(args.output_dir, 'feature_std.npy'), std)
    print(f"统计量已保存到 {args.output_dir}")

    print("\n" + "=" * 60)
    print("步骤 2: 创建数据集")
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

    train_dataset = HFTDataset(args.data_dir, file_list=train_files, mean=mean, std=std)
    val_dataset = HFTDataset(args.data_dir, file_list=val_files, mean=mean, std=std)
    test_dataset = HFTDataset(args.data_dir, file_list=test_files, mean=mean, std=std)

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

    print("\n" + "=" * 60)
    print("步骤 3: 创建模型")
    print("=" * 60)
    model = create_regression_model(
        input_dim=INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_windows=5,
        dropout=args.dropout,
    ).to(device)
    print(f"参数量: {count_parameters(model):,}")

    criterion = nn.HuberLoss(delta=args.huber_delta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    scaler = GradScaler()
    ema = EMA(model, decay=args.ema_decay)

    print("\n" + "=" * 60)
    print("步骤 4: 开始训练")
    print("=" * 60)

    best_val_loss = float('inf')
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

        print(f"训练损失: {train_metrics['loss']:.6f}")
        print(f"验证损失: {val_metrics['loss']:.6f}")

        if 'window_metrics' in val_metrics:
            for w_name, w_m in val_metrics['window_metrics'].items():
                print(f"  {w_name}: MAE={w_m['mae']:.6f}, Corr={w_m['correlation']:.4f}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            encoder_state = model.encoder.state_dict()
            torch.save({
                'epoch': epoch + 1,
                'encoder_state': encoder_state,
                'ema_encoder_state': {k.replace('encoder.', ''): v
                                      for k, v in ema.state_dict().items()
                                      if k.startswith('encoder.')},
                'optimizer_state': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': {
                    'input_dim': INPUT_DIM,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'grid_size': 8,
                    'spline_order': 3,
                    'dropout': 0.0,
                    'num_windows': 5,
                },
                'mean': mean,
                'std': std,
                'feature_cols': FEATURE_COLS,
            }, os.path.join(args.output_dir, 'tkan_encoder.pt'))
            print(f"✓ 保存最佳: Epoch {epoch+1}, 验证损失={best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停: 连续{args.patience}轮未提升")
                break

    print("\n" + "=" * 60)
    print("步骤 5: 测试评估")
    print("=" * 60)

    checkpoint = torch.load(os.path.join(args.output_dir, 'tkan_encoder.pt'),
                            weights_only=False)
    model.encoder.load_state_dict(checkpoint['encoder_state'])

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"测试损失: {test_metrics['loss']:.6f}")
    if 'window_metrics' in test_metrics:
        for w_name, w_m in test_metrics['window_metrics'].items():
            print(f"  {w_name}: MAE={w_m['mae']:.6f}, Corr={w_m['correlation']:.4f}")

    print("\n" + "=" * 60)
    print("T-KAN 编码器预训练完成！")
    print(f"模型保存位置: {os.path.join(args.output_dir, 'tkan_encoder.pt')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
