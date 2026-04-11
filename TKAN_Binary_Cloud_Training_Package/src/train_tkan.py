"""
T-KAN 编码器预训练脚本（收益加权版本）

核心改进：
1. V1 设置：40 维原始价量特征，128 hidden_dim，3 层
2. 收益加权 Huber 损失：大幅价格变动的样本获得更高权重
3. EMA（指数移动平均）权重平滑
4. 早停机制

训练流程：
1. 加载 parquet 数据 → 构造 100 tick 序列
2. 计算目标：5 个窗口的中间价变化率
3. 训练 T-KAN 回归模型（收益加权 Huber Loss）
4. 保存编码器权重 + 归一化参数
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import (
    create_regression_model, create_encoder,
    ProfitWeightedHuberLoss, count_parameters
)

FEATURE_COLS = [
    'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'bid6', 'bid7', 'bid8', 'bid9', 'bid10',
    'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask6', 'ask7', 'ask8', 'ask9', 'ask10',
    'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'bsize6', 'bsize7', 'bsize8', 'bsize9', 'bsize10',
    'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'asize6', 'asize7', 'asize8', 'asize9', 'asize10',
]

WINDOW_SIZES = [5, 10, 20, 40, 60]
LABEL_COLS = [f'label_{w}' for w in WINDOW_SIZES]


def clean_features(features: np.ndarray) -> np.ndarray:
    """特征清洗：处理 NaN/Inf，按列类型裁剪

    参数：
        features: (n_ticks, n_features) numpy 数组
    返回：
        清洗后的 numpy 数组
    """
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    for i, col in enumerate(FEATURE_COLS):
        if col.startswith('bid') or col.startswith('ask'):
            features[:, i] = np.clip(features[:, i], -0.3, 0.3)
        elif col.startswith('bsize') or col.startswith('asize'):
            features[:, i] = np.clip(features[:, i], 0, 100)
    return features


class SequenceDataset(Dataset):
    """100 tick 序列数据集

    每个样本包含 100 个 tick 的特征数据，
    目标为 5 个窗口的中间价变化率。
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.from_numpy(sequences).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx]


def load_data(data_dir: str, max_files: int = None, sample_interval: int = 10):
    """加载 parquet 数据并构造序列

    参数：
        data_dir: 数据目录路径
        max_files: 最大加载文件数（None 表示全部加载）
        sample_interval: 采样间隔，每 N 个 tick 取 1 个样本（降低内存占用）
    返回：
        (sequences, targets, feature_mean, feature_std)
        sequences: (n_samples, 100, 40)
        targets: (n_samples, 5)
    """
    parquet_files = sorted([
        f for f in os.listdir(data_dir) if f.endswith('.parquet')
    ])
    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    print(f"找到 {len(parquet_files)} 个数据文件")

    all_sequences = []
    all_targets = []
    max_window = max(WINDOW_SIZES)

    for i, fname in enumerate(parquet_files):
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            print(f"  跳过文件 {fname}: {e}")
            continue

        if len(df) < 100 + max_window:
            continue

        feature_data = df[FEATURE_COLS].values.astype(np.float32)
        feature_data = clean_features(feature_data)

        midprice = df['midprice'].values.astype(np.float32)

        # 采样：每 sample_interval 个 tick 取 1 个样本
        for t in range(100, len(df) - max_window, sample_interval):
            seq = feature_data[t - 100:t]
            targets = np.array([
                midprice[t + w] - midprice[t] for w in WINDOW_SIZES
            ], dtype=np.float32)
            all_sequences.append(seq)
            all_targets.append(targets)

        if (i + 1) % 20 == 0:
            print(f"  已加载 {i + 1}/{len(parquet_files)} 个文件, "
                  f"累计 {len(all_sequences)} 个样本")

    sequences = np.stack(all_sequences)
    targets = np.stack(all_targets)

    feature_mean = sequences.reshape(-1, sequences.shape[-1]).mean(axis=0)
    feature_std = sequences.reshape(-1, sequences.shape[-1]).std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    print(f"数据加载完成: {sequences.shape[0]} 个样本, "
          f"特征维度 {sequences.shape[-1]}")

    return sequences, targets, feature_mean, feature_std


class EMA:
    """指数移动平均权重平滑

    参数：
        model: PyTorch 模型
        decay: 衰减系数（0.999 表示每步保留 99.9% 的旧权重）
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch: int, total_epochs: int):
    """训练一个 epoch（带进度条）

    参数：
        model: T-KAN 回归模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        loss_fn: 损失函数（收益加权 Huber Loss）
        device: 计算设备
        epoch: 当前 epoch
        total_epochs: 总 epoch 数
    返回：
        平均损失值
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [训练]", leave=False)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, epoch: int, total_epochs: int):
    """评估模型（带进度条）

    参数：
        model: T-KAN 回归模型
        dataloader: 验证数据加载器
        loss_fn: 损失函数
        device: 计算设备
        epoch: 当前 epoch
        total_epochs: 总 epoch 数
    返回：
        (平均损失, 各窗口 MAE, 各窗口相关系数)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [验证]", leave=False)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / max(n_batches, 1)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    maes = []
    corrs = []
    for w_idx in range(len(WINDOW_SIZES)):
        mae = np.mean(np.abs(all_preds[:, w_idx] - all_targets[:, w_idx]))
        corr = np.corrcoef(all_preds[:, w_idx], all_targets[:, w_idx])[0, 1]
        if np.isnan(corr):
            corr = 0.0
        maes.append(mae)
        corrs.append(corr)

    return avg_loss, maes, corrs


def main():
    parser = argparse.ArgumentParser(description='T-KAN 编码器预训练（收益加权版本）')
    parser.add_argument('--data_dir', type=str, default='../../2026train_set/2026train_set')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--profit_scale', type=float, default=10.0)
    parser.add_argument('--profit_min_weight', type=float, default=0.5)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("步骤 1: 加载数据")
    print("=" * 60)

    sequences, targets, feature_mean, feature_std = load_data(
        args.data_dir, args.max_files, args.sample_interval
    )

    sequences = (sequences - feature_mean) / feature_std
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples = len(sequences)
    n_val = min(int(n_samples * 0.1), 50000)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = SequenceDataset(sequences[train_idx], targets[train_idx])
    val_dataset = SequenceDataset(sequences[val_idx], targets[val_idx])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=False
    )

    print(f"训练集: {n_train} 样本, 验证集: {n_val} 样本")

    print("\n" + "=" * 60)
    print("步骤 2: 创建模型")
    print("=" * 60)

    input_dim = len(FEATURE_COLS)
    model = create_regression_model(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    print(f"模型参数量: {count_parameters(model):,}")
    print(f"配置: input_dim={input_dim}, hidden_dim={args.hidden_dim}, "
          f"num_layers={args.num_layers}")

    loss_fn = ProfitWeightedHuberLoss(
        scale=args.profit_scale,
        min_weight=args.profit_min_weight,
    )
    print(f"损失函数: 收益加权 Huber Loss (scale={args.profit_scale}, "
          f"min_weight={args.profit_min_weight})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ema = EMA(model, decay=args.ema_decay)

    print("\n" + "=" * 60)
    print("步骤 3: 训练")
    print("=" * 60)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch + 1, args.epochs)
        ema.update()
        scheduler.step()

        val_loss, maes, corrs = evaluate(model, val_loader, loss_fn, device, epoch + 1, args.epochs)

        elapsed = time.time() - t0

        print(f"\nEpoch {epoch + 1}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  训练损失: {train_loss:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        for w_idx, w in enumerate(WINDOW_SIZES):
            print(f"  label_{w}: MAE={maes[w_idx]:.6f}, Corr={corrs[w_idx]:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint = {
                'encoder_state': model.encoder.state_dict(),
                'ema_encoder_state': ema.state_dict(),
                'config': {
                    'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'grid_size': 8,
                    'spline_order': 3,
                },
                'mean': feature_mean,
                'std': feature_std,
                'epoch': epoch + 1,
                'val_loss': val_loss,
            }
            save_path = os.path.join(args.output_dir, 'tkan_encoder.pt')
            torch.save(checkpoint, save_path)
            print(f"  ✓ 最佳模型已保存 (Epoch {best_epoch})")
        else:
            patience_counter += 1
            print(f"  ✗ 未提升 ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print(f"\n早停触发: 连续 {args.patience} 轮未提升")
            break

    print("\n" + "=" * 60)
    print(f"训练完成! 最佳 Epoch: {best_epoch}, 验证损失: {best_val_loss:.6f}")
    print(f"模型保存路径: {os.path.join(args.output_dir, 'tkan_encoder.pt')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
