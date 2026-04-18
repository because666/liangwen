"""
T-KAN 端到端分类模型

架构：
- T-KAN 编码器：提取时序特征 (100, 40) → (128,)
- 分类头：预测 5 个窗口的标签 (128,) → (5,) 每个位置是三分类

训练方式：
- 使用交叉熵损失
- 支持类别权重（处理样本不平衡）
- 支持收益加权（关注大波动样本）

使用方法：
python train_tkan_classifier.py
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

FEATURE_COLS = [
    'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'bid6', 'bid7', 'bid8', 'bid9', 'bid10',
    'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask6', 'ask7', 'ask8', 'ask9', 'ask10',
    'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'bsize6', 'bsize7', 'bsize8', 'bsize9', 'bsize10',
    'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'asize6', 'asize7', 'asize8', 'asize9', 'asize10',
]

WINDOW_SIZES = [5, 10, 20, 40, 60]
LABEL_COLS = [f'label_{w}' for w in WINDOW_SIZES]
FEE = 0.0001


class StableSplineLinear(nn.Module):
    """数值稳定的 B-spline 线性层"""

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 8, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        grid = torch.linspace(-1, 1, grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.base_weight, gain=0.5)
        nn.init.normal_(self.spline_weight, std=0.01)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -2.0, 2.0)
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            denom1 = grid[k:-1] - grid[:-(k + 1)]
            denom1 = torch.where(denom1.abs() < 1e-8, torch.ones_like(denom1), denom1)
            denom2 = grid[k + 1:] - grid[1:-k]
            denom2 = torch.where(denom2.abs() < 1e-8, torch.ones_like(denom2), denom2)
            bases = (
                (x - grid[:-(k + 1)]) / denom1 * bases[..., :-1]
                + (grid[k + 1:] - x) / denom2 * bases[..., 1:]
            )
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        base_output = F.linear(x, self.base_weight)
        spline_basis = self.b_splines(x)
        spline_output = torch.einsum('bij,oij->bo', spline_basis, self.spline_weight)
        output = base_output + spline_output
        return output.view(*original_shape[:-1], self.out_features)


class TKANLayer(nn.Module):
    """T-KAN 单层（带残差连接）"""

    def __init__(self, features: int, grid_size: int = 8,
                 spline_order: int = 3, dropout: float = 0.0):
        super().__init__()
        self.features = features
        self.spline_linear = StableSplineLinear(features, features, grid_size, spline_order)
        self.layer_norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.spline_linear(x)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = x + residual
        return x


class TKANClassifier(nn.Module):
    """T-KAN 端到端分类器
    
    架构：
    1. 特征嵌入: Linear(40, hidden_dim) + LayerNorm + GELU + Dropout
    2. T-KAN 层 x num_layers: 带残差连接
    3. 时序平均池化: mean(dim=1) → (batch, hidden_dim)
    4. 分类头: Linear(hidden_dim, num_windows * 3) → 预测 5 个窗口的标签
    """

    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, num_layers: int = 3,
                 grid_size: int = 8, spline_order: int = 3, dropout: float = 0.15,
                 num_windows: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_windows = num_windows

        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.tkan_layers = nn.ModuleList([
            TKANLayer(hidden_dim, grid_size, spline_order, dropout)
            for _ in range(num_layers)
        ])

        # 分类头：每个窗口预测 3 个类别
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_windows * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: (batch, seq_len, input_dim)
        返回：
            logits: (batch, num_windows * 3) 或 (batch, num_windows, 3)
        """
        x = self.feature_embedding(x)
        for layer in self.tkan_layers:
            x = layer(x)
        x = x.mean(dim=1)  # (batch, hidden_dim)
        logits = self.classifier(x)  # (batch, num_windows * 3)
        return logits.view(-1, self.num_windows, 3)  # (batch, num_windows, 3)


class StockDataset(Dataset):
    """股票数据集（支持数据增强）"""

    def __init__(self, sequences, labels, returns=None, weights=None, 
                 augment=False, noise_scale=0.01, drop_prob=0.1):
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()
        self.returns = torch.from_numpy(returns).float() if returns is not None else None
        self.weights = torch.from_numpy(weights).float() if weights is not None else None
        self.augment = augment
        self.noise_scale = noise_scale
        self.drop_prob = drop_prob

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 数据增强（只在训练时使用）
        if self.augment:
            # 添加高斯噪声
            noise = torch.randn_like(seq) * self.noise_scale
            seq = seq + noise
            
            # 随机时间步丢弃（Temporal Dropout）
            if torch.rand(1) < 0.5:
                mask = torch.rand(seq.shape[0]) > self.drop_prob
                seq = seq * mask.unsqueeze(-1).float()
        
        if self.returns is not None and self.weights is not None:
            return seq, self.labels[idx], self.returns[idx], self.weights[idx]
        elif self.returns is not None:
            return seq, self.labels[idx], self.returns[idx]
        else:
            return seq, self.labels[idx]


def clean_features(features: np.ndarray) -> np.ndarray:
    """特征清洗"""
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    for i, col in enumerate(FEATURE_COLS):
        if col.startswith('bid') or col.startswith('ask'):
            features[:, i] = np.clip(features[:, i], -0.3, 0.3)
        elif col.startswith('bsize') or col.startswith('asize'):
            features[:, i] = np.clip(features[:, i], 0, 100)
    return features


def load_data(data_dir, max_files=None, sample_interval=5):
    """加载数据"""
    parquet_files = sorted([
        f for f in os.listdir(data_dir) if f.endswith('.parquet')
    ])
    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    print(f"找到 {len(parquet_files)} 个数据文件")

    all_sequences = []
    all_labels = []
    all_returns = []
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
        label_data = df[LABEL_COLS].values.astype(np.int32)

        for t in range(100, len(df) - max_window, sample_interval):
            seq = feature_data[t - 100:t]
            lbl = label_data[t]
            ret = np.array([
                midprice[t + w] - midprice[t] for w in WINDOW_SIZES
            ], dtype=np.float32)
            all_sequences.append(seq)
            all_labels.append(lbl)
            all_returns.append(ret)

        if (i + 1) % 20 == 0:
            print(f"  已加载 {i + 1}/{len(parquet_files)} 个文件, "
                  f"累计 {len(all_sequences)} 个样本")

    sequences = np.stack(all_sequences)
    labels = np.stack(all_labels)
    returns = np.stack(all_returns)

    feature_mean = sequences.reshape(-1, sequences.shape[-1]).mean(axis=0)
    feature_std = sequences.reshape(-1, sequences.shape[-1]).std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    print(f"数据加载完成: {sequences.shape[0]} 个样本")

    return sequences, labels, returns, feature_mean, feature_std


def compute_sample_weights(returns: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """计算样本权重（基于收益绝对值）"""
    abs_returns = np.abs(returns).max(axis=1)
    weights = np.exp(abs_returns * scale)
    weights = np.clip(weights, 0.5, 10.0)
    return weights.astype(np.float32)


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, logits, targets, weights=None):
        """
        参数：
            logits: (batch, num_classes)
            targets: (batch,)
            weights: (batch,) 可选的样本权重
        """
        batch_size, num_classes = logits.shape
        
        # 创建平滑后的标签
        smoothed_targets = torch.zeros_like(logits)
        smoothed_targets.fill_(self.smoothing / (num_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # 计算交叉熵
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smoothed_targets * log_probs).sum(dim=-1)
        
        if weights is not None:
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
            
        return loss


def train_epoch(model, dataloader, optimizer, criterion, device, use_weights=False):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="训练", leave=False)
    for batch in pbar:
        if use_weights:
            sequences, labels, returns, weights = batch
            weights = weights.to(device)
        else:
            sequences, labels = batch[:2]
            weights = None

        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)  # (batch, num_windows, 3)

        loss = 0.0
        for w_idx in range(len(WINDOW_SIZES)):
            if use_weights and weights is not None:
                loss += criterion(logits[:, w_idx, :], labels[:, w_idx], weights)
            else:
                loss += criterion(logits[:, w_idx, :], labels[:, w_idx])

        loss = loss / len(WINDOW_SIZES)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        total_samples += sequences.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / total_samples


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch[0].to(device)
            labels = batch[1]

            logits = model(sequences)  # (batch, num_windows, 3)
            preds = logits.argmax(dim=-1).cpu().numpy()  # (batch, num_windows)

            all_preds.append(preds)
            all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 计算准确率
    accuracies = {}
    for w_idx, w in enumerate(WINDOW_SIZES):
        acc = (preds[:, w_idx] == labels[:, w_idx]).mean()
        accuracies[w] = acc

    return preds, labels, accuracies


def main():
    parser = argparse.ArgumentParser(description='T-KAN 端到端分类器训练')
    parser.add_argument('--data_dir', type=str, default='../2026train_set/2026train_set')
    parser.add_argument('--output_dir', type=str, default='output_tkan_classifier')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_weights', action='store_true', help='使用收益加权')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='标签平滑系数')
    parser.add_argument('--augment', action='store_true', help='启用数据增强')
    parser.add_argument('--noise_scale', type=float, default=0.02, help='数据增强噪声强度')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='学习率预热轮数')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    print("\n" + "=" * 80)
    print("加载数据")
    print("=" * 80)

    sequences, labels, returns, data_mean, data_std = load_data(
        args.data_dir, args.max_files, args.sample_interval
    )

    # 归一化
    sequences = (sequences - data_mean) / data_std
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)

    # 计算样本权重
    if args.use_weights:
        sample_weights = compute_sample_weights(returns)
        print(f"样本权重范围: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
    else:
        sample_weights = None

    # 划分数据集
    n_samples = len(sequences)
    n_val = min(int(n_samples * 0.1), 50000)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    print(f"训练集: {n_train} 样本, 验证集: {n_val} 样本")

    # 创建数据集
    train_dataset = StockDataset(
        sequences[train_idx], labels[train_idx], 
        returns[train_idx] if args.use_weights else None,
        sample_weights[train_idx] if args.use_weights else None,
        augment=args.augment,
        noise_scale=args.noise_scale
    )
    val_dataset = StockDataset(sequences[val_idx], labels[val_idx], augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # 创建模型
    print("\n" + "=" * 80)
    print("创建模型")
    print("=" * 80)

    model = TKANClassifier(
        input_dim=len(FEATURE_COLS),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_windows=len(WINDOW_SIZES),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")

    # 损失函数和优化器
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度：预热 + 余弦退火
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练
    print("\n" + "=" * 80)
    print("开始训练")
    print(f"配置：dropout={args.dropout}, weight_decay={args.weight_decay}, "
          f"label_smoothing={args.label_smoothing}, augment={args.augment}")
    print("=" * 80)

    best_acc = 0.0
    best_epoch = 0
    no_improve_count = 0
    min_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, args.use_weights
        )
        scheduler.step()

        preds, labels_val, accuracies = evaluate(model, val_loader, device)

        avg_acc = np.mean(list(accuracies.values()))
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Acc: {avg_acc:.4f}, "
              f"LR: {current_lr:.6f}")

        for w, acc in accuracies.items():
            print(f"  label_{w}: {acc:.4f}")

        # 保存最佳模型（基于验证准确率）
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_epoch = epoch + 1
            no_improve_count = 0
            torch.save({
                'model_state': model.state_dict(),
                'config': {
                    'input_dim': len(FEATURE_COLS),
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_windows': len(WINDOW_SIZES),
                },
                'mean': data_mean,
                'std': data_std,
                'epoch': epoch + 1,
                'best_acc': best_acc,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  保存最佳模型 (Epoch {best_epoch}, Acc: {best_acc:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"\n早停：连续 {args.patience} 个 epoch 无改善")
                break

    print("\n" + "=" * 80)
    print(f"训练完成！最佳 Epoch: {best_epoch}, 最佳 Acc: {best_acc:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
