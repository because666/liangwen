"""
高频量化模型训练脚本 - 优化版

优化点：
1. 数据增强 + 因子工程
2. 新的模型架构（TCN + Transformer）
3. Focal Loss 处理类别不平衡
4. 学习率预热 + 余弦退火
5. 梯度裁剪防止nan
6. 混合精度训练（可选）
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from model_v2 import HFTPredictor, HFTPredictorLite, FocalLoss, create_model


BASE_FEATURE_COLS = [
    "open", "high", "low", "close",
    "volume_delta", "amount_delta",
    "bid1", "bsize1", "bid2", "bsize2", "bid3", "bsize3", "bid4", "bsize4", "bid5", "bsize5",
    "bid6", "bsize6", "bid7", "bsize7", "bid8", "bsize8", "bid9", "bsize9", "bid10", "bsize10",
    "ask1", "asize1", "ask2", "asize2", "ask3", "asize3", "ask4", "asize4", "ask5", "asize5",
    "ask6", "asize6", "ask7", "asize7", "ask8", "asize8", "ask9", "asize9", "ask10", "asize10",
    "avgbid", "avgask", "totalbsize", "totalasize",
    "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
    "lb_ind", "la_ind", "mb_ind", "ma_ind", "cb_ind", "ca_ind",
    "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc",
    "midprice1", "midprice2", "midprice3", "midprice4", "midprice5",
    "midprice6", "midprice7", "midprice8", "midprice9", "midprice10",
    "spread1", "spread2", "spread3", "spread4", "spread5",
    "spread6", "spread7", "spread8", "spread9", "spread10",
    "bid_diff1", "bid_diff2", "bid_diff3", "bid_diff4", "bid_diff5",
    "bid_diff6", "bid_diff7", "bid_diff8", "bid_diff9", "bid_diff10",
    "ask_diff1", "ask_diff2", "ask_diff3", "ask_diff4", "ask_diff5",
    "ask_diff6", "ask_diff7", "ask_diff8", "ask_diff9", "ask_diff10",
    "bid_mean", "ask_mean", "bsize_mean", "asize_mean",
    "cumspread", "imbalance",
    "bid_rate1", "bid_rate2", "bid_rate3", "bid_rate4", "bid_rate5",
    "bid_rate6", "bid_rate7", "bid_rate8", "bid_rate9", "bid_rate10",
    "ask_rate1", "ask_rate2", "ask_rate3", "ask_rate4", "ask_rate5",
    "ask_rate6", "ask_rate7", "ask_rate8", "ask_rate9", "ask_rate10",
    "bsize_rate1", "bsize_rate2", "bsize_rate3", "bsize_rate4", "bsize_rate5",
    "bsize_rate6", "bsize_rate7", "bsize_rate8", "bsize_rate9", "bsize_rate10",
    "asize_rate1", "asize_rate2", "asize_rate3", "asize_rate4", "asize_rate5",
    "asize_rate6", "asize_rate7", "asize_rate8", "asize_rate9", "asize_rate10"
]

LABEL_COLS = ["label_5", "label_10", "label_20", "label_40", "label_60"]


class LOBDataset(Dataset):
    """
    订单簿数据集
    """
    
    def __init__(
        self,
        data_dir: str,
        feature_cols: list,
        label_cols: list,
        seq_len: int = 100,
        stride: int = 5,
        max_files: int = 50,
        augment: bool = False,
        is_train: bool = True
    ):
        self.data_dir = data_dir
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment and is_train
        self.is_train = is_train
        
        self.features = []
        self.labels = []
        
        file_pattern = os.path.join(data_dir, "snapshot_sym*.parquet")
        files = sorted(glob.glob(file_pattern))[:max_files]
        
        print(f"加载数据文件: {len(files)} 个")
        
        for f in tqdm(files, desc="加载数据"):
            try:
                df = pd.read_parquet(f)
                
                df = df.fillna(0)
                df = df.replace([np.inf, -np.inf], 0)
                
                for col in self.feature_cols:
                    if col not in df.columns:
                        df[col] = 0
                
                feature_data = df[self.feature_cols].values.astype(np.float32)
                
                for col in self.label_cols:
                    if col not in df.columns:
                        df[col] = 1
                label_data = df[self.label_cols].values.astype(np.int64)
                
                for i in range(0, len(df) - seq_len, stride):
                    self.features.append(feature_data[i:i+seq_len])
                    self.labels.append(label_data[i+seq_len-1])
            
            except Exception as e:
                print(f"加载文件 {f} 失败: {e}")
                continue
        
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"总样本数: {len(self.features)}")
        print(f"特征形状: {self.features.shape}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        X = self.features[idx].copy()
        y = self.labels[idx].copy()
        
        if self.augment:
            X = self._augment(X)
        
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        
        return X, y
    
    def _augment(self, X: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.005, X.shape).astype(np.float32)
            X = X + noise
        
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.98, 1.02)
            X = X * scale
        
        if np.random.random() < 0.2:
            flip_idx = np.random.randint(0, len(X))
            X[flip_idx:] = X[flip_idx:][::-1]
        
        return X


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    gradient_clip_norm=1.0
):
    model.train()
    total_loss = 0
    correct = [0] * 5
    total = 0
    
    pbar = tqdm(train_loader, desc="训练")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                outputs = model(inputs)
                loss = 0
                for i, output in enumerate(outputs):
                    loss += criterion[i](output, targets[:, i])
                loss = loss / len(outputs)
            
            scaler.scale(loss).backward()
            
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = 0
            for i, output in enumerate(outputs):
                loss += criterion[i](output, targets[:, i])
            loss = loss / len(outputs)
            
            loss.backward()
            
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            optimizer.step()
        
        total_loss += loss.item()
        total += inputs.size(0)
        
        for i, output in enumerate(outputs):
            pred = output.argmax(dim=1)
            correct[i] += (pred == targets[:, i]).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{correct[0]/total:.3f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracies = [c / total for c in correct]
    
    return avg_loss, accuracies


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = [0] * 5
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="验证"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            loss = 0
            for i, output in enumerate(outputs):
                loss += criterion[i](output, targets[:, i])
            loss = loss / len(outputs)
            
            total_loss += loss.item()
            total += inputs.size(0)
            
            for i, output in enumerate(outputs):
                pred = output.argmax(dim=1)
                correct[i] += (pred == targets[:, i]).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracies = [c / total for c in correct]
    
    return avg_loss, accuracies


def main():
    parser = argparse.ArgumentParser(description="高频量化模型训练")
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--max_files', type=int, default=100, help='最大文件数')
    parser.add_argument('--stride', type=int, default=3, help='采样步长')
    parser.add_argument('--seq_len', type=int, default=100, help='序列长度')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--model_type', type=str, default='lite', choices=['full', 'lite'], help='模型类型')
    parser.add_argument('--augment', action='store_true', help='是否数据增强')
    parser.add_argument('--use_amp', action='store_true', help='是否使用混合精度')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='预热轮数')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    feature_cols = [c for c in BASE_FEATURE_COLS]
    
    print(f"加载训练数据...")
    train_dataset = LOBDataset(
        args.data_dir,
        feature_cols,
        LABEL_COLS,
        seq_len=args.seq_len,
        stride=args.stride,
        max_files=int(args.max_files * 0.8),
        augment=args.augment,
        is_train=True
    )
    
    print(f"加载验证数据...")
    val_dataset = LOBDataset(
        args.data_dir,
        feature_cols,
        LABEL_COLS,
        seq_len=args.seq_len,
        stride=args.stride * 2,
        max_files=int(args.max_files * 0.2),
        augment=False,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    num_features = len(feature_cols)
    num_classes_per_head = [3] * 5
    
    print(f"创建模型: {args.model_type}, 特征数: {num_features}")
    model = create_model(
        model_type=args.model_type,
        num_features=num_features,
        num_classes_per_head=num_classes_per_head,
        hidden_dim=128 if args.model_type == 'full' else 64,
        dropout=0.3
    )
    model = model.to(device)
    
    dummy_input = torch.randn(1, args.seq_len, num_features).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    
    class_weights = []
    for i in range(5):
        labels = train_dataset.labels[:, i]
        weights = compute_class_weights(torch.from_numpy(labels), num_classes=3)
        class_weights.append(weights.to(device))
        print(f"类别权重 (label_{LABEL_COLS[i]}): {weights.numpy()}")
    
    criterion = [FocalLoss(alpha=weights.cpu().numpy().tolist(), gamma=2.0) for weights in class_weights]
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    
    scaler = GradScaler() if args.use_amp else None
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"\n开始训练...")
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
            print(f"学习率 (预热): {optimizer.param_groups[0]['lr']:.6f}")
        else:
            cosine_scheduler.step()
            print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_accs = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.gradient_clip
        )
        
        val_loss, val_accs = validate(model, val_loader, criterion, device)
        
        print(f"训练Loss: {train_loss:.4f}")
        print(f"训练准确率: {', '.join([f'{acc:.4f}' for acc in train_accs])}")
        print(f"验证Loss: {val_loss:.4f}")
        print(f"验证准确率: {', '.join([f'{acc:.4f}' for acc in val_accs])}")
        
        avg_val_acc = np.mean(val_accs)
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_state = model.state_dict().copy()
            print(f"✓ 保存最佳模型 (验证准确率: {best_val_acc:.4f})")
            
            checkpoint = {
                'model_state': best_model_state,
                'epoch': epoch,
                'val_acc': best_val_acc,
                'meta': {
                    'num_classes_per_head': num_classes_per_head,
                    'model_type': args.model_type,
                    'feature_cols': feature_cols
                }
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
    
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")
    
    config = {
        "python_version": "3.10",
        "batch": 256,
        "feature": feature_cols,
        "label": LABEL_COLS
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def compute_class_weights(labels, num_classes=3):
    class_counts = np.bincount(labels.numpy().flatten(), minlength=num_classes).astype(float)
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-8)
    weights = weights / weights.sum()
    return torch.from_numpy(weights)


if __name__ == '__main__':
    main()
