"""
T-KAN+OFI 模型训练脚本（流式数据加载版本）

解决内存不足问题，使用流式加载
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import TKANOFIModel, create_model, count_parameters
from models.tkan_encoder import TKANEncoder, RKANEncoder, HybridEncoder
from training.losses import HuberLoss, ProfitAwareLoss, CombinedLoss, compute_cumulative_return, compute_single_return


def compute_ofi_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算 OFI 特征"""
    df = df.copy()
    n = len(df)
    
    mb_intst = df['mb_intst'].values if 'mb_intst' in df.columns else np.zeros(n)
    ma_intst = df['ma_intst'].values if 'ma_intst' in df.columns else np.zeros(n)
    
    ofi_raw = mb_intst - ma_intst
    df['ofi_raw'] = ofi_raw
    
    ofi_ewm = np.zeros(n)
    alpha = 0.1
    for i in range(n):
        if i == 0:
            ofi_ewm[i] = ofi_raw[i]
        else:
            ofi_ewm[i] = (1 - alpha) * ofi_ewm[i-1] + alpha * ofi_raw[i]
    df['ofi_ewm'] = ofi_ewm
    
    weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    ofi_multilevel = np.zeros(n)
    for k, w in enumerate(weights):
        mb_col = 'mb_intst'
        ma_col = 'ma_intst'
        if mb_col in df.columns and ma_col in df.columns:
            ofi_multilevel += w * (df[mb_col].values - df[ma_col].values)
    df['ofi_multilevel'] = ofi_multilevel
    
    ofi_velocity = np.zeros(n)
    ofi_velocity[1:] = ofi_raw[1:] - ofi_raw[:-1]
    df['ofi_velocity'] = ofi_velocity
    
    ofi_volatility = np.zeros(n)
    window = 10
    for i in range(window, n):
        ofi_volatility[i] = np.std(ofi_raw[i-window:i])
    df['ofi_volatility'] = ofi_volatility
    
    return df


def get_ofi_column_names():
    """获取OFI特征列名"""
    return ['ofi_raw', 'ofi_ewm', 'ofi_multilevel', 'ofi_velocity', 'ofi_volatility']


class StreamingHFTDataset(IterableDataset):
    """流式高频交易数据集 - 解决内存不足问题"""
    
    def __init__(self, 
                 data_dir: str,
                 feature_cols: List[str],
                 label_cols: List[str],
                 seq_len: int = 100,
                 max_files: Optional[int] = None,
                 compute_ofi: bool = True,
                 normalize: bool = True,
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None,
                 shuffle_files: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.seq_len = seq_len
        self.max_files = max_files
        self.compute_ofi = compute_ofi
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.shuffle_files = shuffle_files
        
        self.all_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        if self.max_files:
            self.all_files = self.all_files[:self.max_files]
        
        self.ofi_cols = get_ofi_column_names()
        self.all_feature_cols = feature_cols + self.ofi_cols
        
        print(f"数据集包含 {len(self.all_files)} 个文件")
        
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """流式迭代器 - 每次只加载一个文件"""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            files = self.all_files
        else:
            # 多进程时分配文件
            per_worker = len(self.all_files) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.all_files)
            files = self.all_files[start:end]
        
        if self.shuffle_files:
            import random
            random.shuffle(files)
        
        sample_count = 0
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            
            try:
                df = pd.read_parquet(file_path)
                
                if self.compute_ofi:
                    df = compute_ofi_features(df)
                
                for col in self.all_feature_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                
                features = df[self.all_feature_cols].values.astype(np.float32)
                labels = df[self.label_cols].values.astype(np.int64)
                
                midprice = df['midprice'].values if 'midprice' in df.columns else np.zeros(len(df))
                
                # 检查midprice是否有效
                if np.all(midprice == 0) or np.all(np.isnan(midprice)):
                    print(f"  警告: 文件 {file} 的midprice无效，跳过")
                    continue
                
                windows = [5, 10, 20, 40, 60]
                deltas = np.zeros((len(df), len(windows)), dtype=np.float32)
                for i, w in enumerate(windows):
                    if w < len(df):
                        deltas[:-w, i] = midprice[w:] - midprice[:-w]
                
                # 检查deltas是否有效
                if np.all(deltas == 0):
                    print(f"  警告: 文件 {file} 的deltas全为0，跳过")
                    continue
                
                if self.normalize and self.mean is not None and self.std is not None:
                    features = (features - self.mean) / self.std
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 确保有足够的样本
                max_window = max(windows)
                valid_end = len(df) - max_window
                
                if valid_end <= self.seq_len:
                    continue
                
                for i in range(self.seq_len, valid_end):
                    yield (
                        torch.from_numpy(features[i-self.seq_len:i]),
                        torch.from_numpy(labels[i]),
                        torch.from_numpy(deltas[i])
                    )
                    sample_count += 1
                    
            except Exception as e:
                print(f"加载文件 {file} 失败: {e}")
                continue
        
        if sample_count == 0:
            print(f"警告: 没有生成任何样本！")
    
    def compute_stats(self, num_files: int = None):
        """计算归一化统计量（使用全量数据，增量计算）"""
        if num_files is None:
            num_files = len(self.all_files)
        
        print(f"计算统计量（使用 {num_files}/{len(self.all_files)} 个文件）...")
        
        # 使用Welford算法增量计算均值和方差
        n = 0
        mean = None
        M2 = None
        
        for i, file in enumerate(self.all_files[:num_files]):
            file_path = os.path.join(self.data_dir, file)
            try:
                df = pd.read_parquet(file_path)
                
                if self.compute_ofi:
                    df = compute_ofi_features(df)
                
                for col in self.all_feature_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                
                features = df[self.all_feature_cols].values.astype(np.float32)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Welford算法增量更新
                if mean is None:
                    mean = np.zeros(features.shape[1])
                    M2 = np.zeros(features.shape[1])
                
                for j in range(len(features)):
                    n += 1
                    delta = features[j] - mean
                    mean += delta / n
                    delta2 = features[j] - mean
                    M2 += delta * delta2
                
                if (i + 1) % 100 == 0:
                    print(f"  已处理 {i+1}/{num_files} 个文件...")
                    
            except Exception as e:
                print(f"  处理文件 {file} 失败: {e}")
                continue
        
        if n == 0:
            print("警告: 没有有效数据，使用默认统计量")
            return np.zeros(len(self.all_feature_cols)), np.ones(len(self.all_feature_cols))
        
        variance = M2 / n
        std = np.sqrt(variance) + 1e-8
        
        print(f"统计量计算完成，共使用 {n} 个样本")
        return mean, std


def create_dataloaders(data_dir: str,
                       feature_cols: List[str],
                       label_cols: List[str],
                       seq_len: int = 100,
                       batch_size: int = 256,
                       max_files: Optional[int] = None,
                       num_workers: int = 0,
                       compute_ofi: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """创建流式数据加载器"""
    
    # 获取所有文件列表
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    if max_files:
        all_files = all_files[:max_files]
    
    # 创建临时数据集计算统计量（使用全量文件）
    temp_dataset = StreamingHFTDataset(
        data_dir=data_dir,
        feature_cols=feature_cols,
        label_cols=label_cols,
        seq_len=seq_len,
        max_files=None,  # 不限制文件数
        compute_ofi=compute_ofi,
        normalize=False
    )
    temp_dataset.all_files = all_files  # 设置全量文件列表
    
    mean, std = temp_dataset.compute_stats(num_files=None)  # 使用全量数据
    
    # 打印统计量信息
    print(f"均值范围: [{mean.min():.4f}, {mean.max():.4f}], 均值: {mean.mean():.4f}")
    print(f"标准差范围: [{std.min():.4f}, {std.max():.4f}], 均值: {std.mean():.4f}")
    
    # 创建训练、验证、测试数据集（使用同样的all_files）
    total_files = len(all_files)
    train_files = int(total_files * 0.7)
    val_files = int(total_files * 0.15)
    
    train_file_list = all_files[:train_files]
    val_file_list = all_files[train_files:train_files + val_files]
    test_file_list = all_files[train_files + val_files:]
    
    print(f"训练文件: {len(train_file_list)}, 验证文件: {len(val_file_list)}, 测试文件: {len(test_file_list)}")
    
    # 创建数据集
    train_dataset = StreamingHFTDataset(
        data_dir=data_dir,
        feature_cols=feature_cols,
        label_cols=label_cols,
        seq_len=seq_len,
        max_files=None,
        compute_ofi=compute_ofi,
        normalize=True,
        mean=mean,
        std=std,
        shuffle_files=True
    )
    train_dataset.all_files = train_file_list
    
    val_dataset = StreamingHFTDataset(
        data_dir=data_dir,
        feature_cols=feature_cols,
        label_cols=label_cols,
        seq_len=seq_len,
        max_files=None,
        compute_ofi=compute_ofi,
        normalize=True,
        mean=mean,
        std=std,
        shuffle_files=False
    )
    val_dataset.all_files = val_file_list
    
    test_dataset = StreamingHFTDataset(
        data_dir=data_dir,
        feature_cols=feature_cols,
        label_cols=label_cols,
        seq_len=seq_len,
        max_files=None,
        compute_ofi=compute_ofi,
        normalize=True,
        mean=mean,
        std=std,
        shuffle_files=False
    )
    test_dataset.all_files = test_file_list
    
    # 创建DataLoader
    # 注意：IterableDataset必须使用shuffle=False，且num_workers建议为0
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # IterableDataset不支持shuffle
        num_workers=0,  # 流式加载建议用0避免多进程问题
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    params = {
        'mean': mean,
        'std': std,
        'feature_cols': feature_cols + get_ofi_column_names(),
        'label_cols': label_cols,
        'seq_len': seq_len
    }
    
    return train_loader, val_loader, test_loader, params


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                scaler: Optional[GradScaler] = None,
                stage: int = 1,
                max_grad_norm: float = 1.0,
                max_batches: Optional[int] = None) -> Dict:
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_huber = 0.0
    total_profit = 0.0
    total_action_rate = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="训练中")
    
    for batch_idx, (features, labels, deltas) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
        
        # 调试：打印第一批数据的统计信息
        if batch_idx == 0 and num_batches == 0:
            print(f"\n[调试] 第一批数据:")
            print(f"  features形状: {features.shape}, 范围: [{features.min():.4f}, {features.max():.4f}]")
            print(f"  deltas形状: {deltas.shape}, 范围: [{deltas.min():.4f}, {deltas.max():.4f}]")
            print(f"  labels形状: {labels.shape}, 唯一值: {torch.unique(labels).tolist()}")
            
        features = features.to(device).float()  # 确保是float32
        deltas = deltas.to(device).float()
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                pred_delta, action_prob = model(features)
                
                if stage == 1:
                    loss = criterion(pred_delta, deltas)
                    huber_loss = loss.item()
                    profit_loss = 0.0
                    action_rate = 0.0
                else:
                    loss, info = criterion(pred_delta, deltas, action_prob)
                    huber_loss = info.get('huber_loss', 0.0)
                    profit_loss = info.get('profit_loss', 0.0)
                    action_rate = info.get('action_rate', 0.0)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_delta, action_prob = model(features)
            
            if stage == 1:
                loss = criterion(pred_delta, deltas)
                huber_loss = loss.item()
                profit_loss = 0.0
                action_rate = 0.0
            else:
                loss, info = criterion(pred_delta, deltas, action_prob)
                huber_loss = info.get('huber_loss', 0.0)
                profit_loss = info.get('profit_loss', 0.0)
                action_rate = info.get('action_rate', 0.0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        total_loss += loss.item()
        total_huber += huber_loss
        total_profit += profit_loss
        total_action_rate += action_rate
        num_batches += 1
        
        # 调试：每100个batch打印详细信息
        if batch_idx % 100 == 0:
            print(f"\n[调试] batch {batch_idx}:")
            print(f"  loss={loss.item():.6f}, pred_delta范围: [{pred_delta.min():.4f}, {pred_delta.max():.4f}]")
            print(f"  deltas范围: [{deltas.min():.4f}, {deltas.max():.4f}]")
            print(f"  pred_delta均值: {pred_delta.mean():.6f}, deltas均值: {deltas.mean():.6f}")
            print(f"  |pred_delta-deltas|均值: {(pred_delta - deltas).abs().mean():.6f}")
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'huber': f'{huber_loss:.4f}',
            'profit': f'{profit_loss:.4f}'
        })
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'huber_loss': total_huber / num_batches if num_batches > 0 else 0.0,
        'profit_loss': total_profit / num_batches if num_batches > 0 else 0.0,
        'action_rate': total_action_rate / num_batches if num_batches > 0 else 0.0
    }


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             stage: int = 1,
             max_batches: Optional[int] = None) -> Dict:
    """评估模型"""
    model.eval()
    
    total_loss = 0.0
    total_huber = 0.0
    total_profit = 0.0
    total_action_rate = 0.0
    all_predictions = []
    all_deltas = []
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (features, labels, deltas) in enumerate(tqdm(dataloader, desc="评估中")):
            if max_batches and batch_idx >= max_batches:
                break
                
            features = features.to(device).float()  # 确保是float32
            deltas = deltas.to(device).float()
            
            pred_delta, action_prob = model(features)
            
            if stage == 1:
                loss = criterion(pred_delta, deltas)
                huber_loss = loss.item()
                profit_loss = 0.0
                action_rate = 0.0
            else:
                loss, info = criterion(pred_delta, deltas, action_prob)
                huber_loss = info.get('huber_loss', 0.0)
                profit_loss = info.get('profit_loss', 0.0)
                action_rate = info.get('action_rate', 0.0)
            
            total_loss += loss.item()
            total_huber += huber_loss
            total_profit += profit_loss
            total_action_rate += action_rate
            num_batches += 1
            
            predictions = model.predict(features)
            all_predictions.append(predictions.cpu())
            all_deltas.append(deltas.cpu())
    
    if num_batches == 0:
        return {
            'loss': 0.0,
            'huber_loss': 0.0,
            'profit_loss': 0.0,
            'action_rate': 0.0,
            'cumulative_return': 0.0,
            'single_return': 0.0
        }
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_deltas = torch.cat(all_deltas, dim=0)
    
    cumulative_return = compute_cumulative_return(all_predictions, all_deltas)
    single_return = compute_single_return(all_predictions, all_deltas)
    
    return {
        'loss': total_loss / num_batches,
        'huber_loss': total_huber / num_batches,
        'profit_loss': total_profit / num_batches,
        'action_rate': total_action_rate / num_batches,
        'cumulative_return': cumulative_return,
        'single_return': single_return
    }


def main():
    parser = argparse.ArgumentParser(description='T-KAN+OFI 模型训练（流式版本）')
    parser.add_argument('--data_dir', type=str, 
                        default='/mnt/workspace/TEMP-FILE-STATION/良文杯/2026train_set/2026train_set',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, 
                        default='/mnt/workspace/TEMP-FILE-STATION/良文杯/新版本_架构V2/submission',
                        help='输出目录')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏维度')  # 快速：384->256
    parser.add_argument('--output_dim', type=int, default=128, help='输出维度')  # 快速：192->128
    parser.add_argument('--num_layers', type=int, default=2, help='编码器层数')  # 快速：3->2
    parser.add_argument('--grid_size', type=int, default=8, help='B样条网格数')  # 快速：12->8
    parser.add_argument('--spline_order', type=int, default=3, help='B样条阶数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')  # 快速：0.15->0.1
    parser.add_argument('--batch_size', type=int, default=256, help='批大小')  # 增大：128->256（更快）
    parser.add_argument('--stage1_epochs', type=int, default=10, help='Stage 1 训练轮数')  # 快速：30->10
    parser.add_argument('--stage2_epochs', type=int, default=10, help='Stage 2 训练轮数')  # 快速：15->10
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')  # 快速：5e-4->1e-3
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')  # 快速：1e-4->1e-5
    parser.add_argument('--gamma', type=float, default=0.03, help='出手惩罚系数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--encoder_type', type=str, default='tkan', choices=['tkan', 'rkan', 'hybrid'],
                        help='编码器类型')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max_files', type=int, default=None, help='最大文件数（用于测试）')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--max_batches_per_epoch', type=int, default=None, help='每epoch最大batch数')
    parser.add_argument('--skip_stage1', action='store_true', help='跳过Stage 1，直接开始Stage 2')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    feature_cols = [
        'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'bid6', 'bid7', 'bid8', 'bid9', 'bid10',
        'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask6', 'ask7', 'ask8', 'ask9', 'ask10',
        'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'bsize6', 'bsize7', 'bsize8', 'bsize9', 'bsize10',
        'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'asize6', 'asize7', 'asize8', 'asize9', 'asize10',
        'midprice', 'imbalance', 'cumspread',
        'mb_intst', 'ma_intst', 'lb_intst', 'la_intst', 'cb_intst', 'ca_intst'
    ]
    
    label_cols = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
    
    print("\n" + "="*50)
    print("创建流式数据加载器...")
    print("="*50)
    
    train_loader, val_loader, test_loader, params = create_dataloaders(
        data_dir=args.data_dir,
        feature_cols=feature_cols,
        label_cols=label_cols,
        seq_len=100,
        batch_size=args.batch_size,
        max_files=args.max_files,
        num_workers=args.num_workers,
        compute_ofi=True
    )
    
    mean, std = params['mean'], params['std']
    ofi_cols = get_ofi_column_names()
    input_dim = len(feature_cols) + len(ofi_cols)
    
    print(f"输入维度: {input_dim}")
    
    print("\n" + "="*50)
    print("创建模型...")
    print("="*50)
    
    config = {
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'output_dim': args.output_dim,
        'num_windows': 5,
        'encoder_type': args.encoder_type,
        'num_encoder_layers': args.num_layers,
        'grid_size': args.grid_size,
        'spline_order': args.spline_order,
        'dropout': args.dropout
    }
    
    model = create_model(config).to(device)
    print(f"模型参数量: {count_parameters(model):,}")
    
    stage1_criterion = HuberLoss()
    stage2_criterion = CombinedLoss(
        huber_weight=1.0,
        profit_weight=1.0,
        action_weight=0.1,
        gamma=args.gamma
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.stage1_epochs + args.stage2_epochs)
    
    scaler = GradScaler() if args.use_amp else None
    
    best_val_return = float('-inf')
    best_epoch = 0
    patience_counter = 0
    patience = 5
    
    if not args.skip_stage1:
        print("\n" + "="*50)
        print("Stage 1: 预训练 (Huber Loss)")
        print("="*50)
        
        for epoch in range(args.stage1_epochs):
            print(f"\nEpoch {epoch+1}/{args.stage1_epochs}")
            
            train_metrics = train_epoch(
                model, train_loader, stage1_criterion, optimizer, device,
                scaler, stage=1, max_grad_norm=args.max_grad_norm,
                max_batches=args.max_batches_per_epoch
            )
            
            val_metrics = evaluate(model, val_loader, stage1_criterion, device, stage=1,
                                  max_batches=args.max_batches_per_epoch)
            
            scheduler.step()
            
            print(f"训练损失: {train_metrics['loss']:.4f}, 验证损失: {val_metrics['loss']:.4f}")
            print(f"验证累计收益: {val_metrics['cumulative_return']:.6f}, 单次收益: {val_metrics['single_return']:.6f}")
            
            if val_metrics['cumulative_return'] > best_val_return:
                best_val_return = val_metrics['cumulative_return']
                best_epoch = epoch + 1
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_return': best_val_return,
                    'config': config,
                    'mean': mean,
                    'std': std,
                    'feature_cols': feature_cols + ofi_cols
                }, os.path.join(args.output_dir, 'best_model_stage1.pt'))
                print(f"保存最佳模型 (Stage 1), Epoch {best_epoch}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: 验证收益连续 {patience} 轮未提升")
                    break
    else:
        print("\n" + "="*50)
        print("跳过 Stage 1，保存当前模型状态")
        print("="*50)
        torch.save({
            'epoch': 0,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_return': 0.0,
            'config': config,
            'mean': mean,
            'std': std,
            'feature_cols': feature_cols + ofi_cols
        }, os.path.join(args.output_dir, 'best_model_stage1.pt'))
        print(f"保存当前模型作为 Stage 1 结果")
    
    print("\n" + "="*50)
    print("Stage 2: 收益感知微调 (Profit-Aware Loss)")
    print("="*50)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model_stage1.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.stage2_epochs)
    
    best_val_return = float('-inf')
    patience_counter = 0
    
    for epoch in range(args.stage2_epochs):
        print(f"\nEpoch {epoch+1}/{args.stage2_epochs}")
        
        train_metrics = train_epoch(
            model, train_loader, stage2_criterion, optimizer, device,
            scaler, stage=2, max_grad_norm=args.max_grad_norm,
            max_batches=args.max_batches_per_epoch
        )
        
        val_metrics = evaluate(model, val_loader, stage2_criterion, device, stage=2,
                              max_batches=args.max_batches_per_epoch)
        
        scheduler.step()
        
        print(f"训练损失: {train_metrics['loss']:.4f}, Huber: {train_metrics['huber_loss']:.4f}, Profit: {train_metrics['profit_loss']:.4f}")
        print(f"验证累计收益: {val_metrics['cumulative_return']:.6f}, 单次收益: {val_metrics['single_return']:.6f}")
        print(f"出手率: {val_metrics['action_rate']:.4f}")
        
        if val_metrics['cumulative_return'] > best_val_return:
            best_val_return = val_metrics['cumulative_return']
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_return': best_val_return,
                'config': config,
                'mean': mean,
                'std': std,
                'feature_cols': feature_cols + ofi_cols,
                'thresholds': [0.5, 0.5, 0.5, 0.5, 0.5]
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"保存最佳模型, Epoch {best_epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停: 验证收益连续 {patience} 轮未提升")
                break
    
    print("\n" + "="*50)
    print("测试集评估")
    print("="*50)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    
    test_metrics = evaluate(model, test_loader, stage2_criterion, device, stage=2,
                           max_batches=args.max_batches_per_epoch)
    print(f"测试累计收益: {test_metrics['cumulative_return']:.6f}")
    print(f"测试单次收益: {test_metrics['single_return']:.6f}")
    print(f"测试出手率: {test_metrics['action_rate']:.4f}")
    
    config_data = {
        'python_version': '3.10',
        'batch': args.batch_size,
        'feature': feature_cols + ofi_cols,
        'label': label_cols
    }
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(args.output_dir, 'best_thresholds.json'), 'w', encoding='utf-8') as f:
        json.dump({'thresholds': [0.5, 0.5, 0.5, 0.5, 0.5]}, f, indent=2, ensure_ascii=False)
    
    print("\n训练完成！")
    print(f"最佳模型已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()
