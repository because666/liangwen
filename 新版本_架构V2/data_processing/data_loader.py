"""
数据加载模块

加载Parquet数据并构建训练集
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.ofi_features import compute_ofi_features, get_ofi_column_names


class HFTDataset(Dataset):
    """高频交易数据集"""
    
    def __init__(self, 
                 data_dir: str,
                 feature_cols: List[str],
                 label_cols: List[str],
                 seq_len: int = 100,
                 max_samples: Optional[int] = None,
                 compute_ofi: bool = True,
                 normalize: bool = True,
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None):
        """
        Args:
            data_dir: 数据目录
            feature_cols: 特征列名
            label_cols: 标签列名
            seq_len: 序列长度
            max_samples: 最大样本数
            compute_ofi: 是否计算OFI特征
            normalize: 是否归一化
            mean: 归一化均值
            std: 归一化标准差
        """
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.compute_ofi = compute_ofi
        self.normalize = normalize
        
        self.samples = []
        self.labels = []
        self.deltas = []
        
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        
        for file_idx, file in enumerate(all_files):
            file_path = os.path.join(data_dir, file)
            df = pd.read_parquet(file_path)
            
            if compute_ofi:
                df = compute_ofi_features(df)
            
            ofi_cols = get_ofi_column_names()
            all_feature_cols = feature_cols + ofi_cols
            
            for col in all_feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            
            features = df[all_feature_cols].values.astype(np.float32)
            labels = df[label_cols].values.astype(np.int64)
            
            midprice = df['midprice'].values if 'midprice' in df.columns else np.zeros(len(df))
            
            windows = [5, 10, 20, 40, 60]
            deltas = np.zeros((len(df), len(windows)), dtype=np.float32)
            for i, w in enumerate(windows):
                if w < len(df):
                    deltas[:-w, i] = midprice[w:] - midprice[:-w]
            
            for i in range(seq_len, len(df) - max(windows)):
                self.samples.append(features[i-seq_len:i])
                self.labels.append(labels[i])
                self.deltas.append(deltas[i])
                
            if max_samples and len(self.samples) >= max_samples:
                break
                
            print(f"已加载文件 {file_idx + 1}/{len(all_files)}: {file}, 累计样本数: {len(self.samples)}")
        
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.deltas = np.array(self.deltas, dtype=np.float32)
        
        if normalize:
            if mean is None:
                self.mean = np.nanmean(self.samples, axis=(0, 1))
            else:
                self.mean = mean
                
            if std is None:
                self.std = np.nanstd(self.samples, axis=(0, 1)) + 1e-8
            else:
                self.std = std
                
            self.samples = (self.samples - self.mean) / self.std
            self.samples = np.nan_to_num(self.samples, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self.mean = None
            self.std = None
            
        print(f"数据集加载完成，总样本数: {len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.samples[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(self.deltas[idx])
        )
    
    def get_normalization_params(self):
        """获取归一化参数"""
        return self.mean, self.std


def create_dataloaders(data_dir: str,
                       feature_cols: List[str],
                       label_cols: List[str],
                       seq_len: int = 100,
                       batch_size: int = 256,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       num_workers: int = 4,
                       compute_ofi: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    创建数据加载器
    
    Args:
        data_dir: 数据目录
        feature_cols: 特征列名
        label_cols: 标签列名
        seq_len: 序列长度
        batch_size: 批大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        num_workers: 数据加载线程数
        compute_ofi: 是否计算OFI特征
        
    Returns:
        train_loader, val_loader, test_loader, params
    """
    full_dataset = HFTDataset(
        data_dir=data_dir,
        feature_cols=feature_cols,
        label_cols=label_cols,
        seq_len=seq_len,
        compute_ofi=compute_ofi,
        normalize=True
    )
    
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    mean, std = full_dataset.get_normalization_params()
    ofi_cols = get_ofi_column_names()
    
    params = {
        'mean': mean,
        'std': std,
        'feature_cols': feature_cols + ofi_cols,
        'label_cols': label_cols,
        'seq_len': seq_len,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size
    }
    
    print(f"训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")
    
    return train_loader, val_loader, test_loader, params


if __name__ == '__main__':
    data_dir = r'd:\量化\良文杯\2026train_set\2026train_set'
    
    feature_cols = [
        'bid1', 'bid2', 'bid3', 'bid4', 'bid5',
        'ask1', 'ask2', 'ask3', 'ask4', 'ask5',
        'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5',
        'asize1', 'asize2', 'asize3', 'asize4', 'asize5',
        'midprice', 'imbalance', 'cumspread',
        'mb_intst', 'ma_intst', 'lb_intst', 'la_intst'
    ]
    
    label_cols = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
    
    train_loader, val_loader, test_loader, params = create_dataloaders(
        data_dir=data_dir,
        feature_cols=feature_cols,
        label_cols=label_cols,
        seq_len=100,
        batch_size=256,
        train_ratio=0.7,
        val_ratio=0.15,
        num_workers=0,
        compute_ofi=True
    )
    
    for batch in train_loader:
        samples, labels, deltas = batch
        print(f"样本形状: {samples.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"价格变化形状: {deltas.shape}")
        break
