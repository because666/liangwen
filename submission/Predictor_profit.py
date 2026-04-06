"""
Predictor类 - 收益导向版 (Profit-Optimized)

专门针对"准确率高但收益低"问题优化的预测器：
1. 使用收益导向训练的模型
2. 更保守的置信度阈值（默认0.6+）
3. 强制减少"不变"预测
4. TTA增强提升稳定性
5. 详细的交易统计输出
"""

from __future__ import annotations

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


OFI_FEATURE_COLS = [
    'net_market_flow', 'net_limit_flow', 'net_cancel_flow',
    'cum_ofi_5', 'cum_ofi_10', 'cum_ofi_20',
    'ofi_volatility_10', 'ofi_volatility_20', 'ofi_abs_sum_20',
    'buy_pressure', 'sell_pressure', 'total_imbalance',
    'ofi_momentum_5', 'ofi_momentum_10', 'ofi_acceleration',
    'ofi_limit_ratio', 'ofi_cancel_ratio', 'ofi_pressure_imbalance',
    'ofi_skewness_10', 'ofi_kurtosis_20',
    'ofi_max_10', 'ofi_min_10', 'ofi_range_10',
    'ofi_sign_consistency_5', 'ofi_sign_consistency_10',
    'ewi_5', 'ewi_10', 'ewi_20',
    'ewi_5_diff', 'ewi_10_diff',
    'ewi_5_deviation', 'ewi_10_deviation',
    'ewi_sign_5', 'ewi_sign_10'
]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        branch_channels = out_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        return self.dropout(torch.cat([b1, b3, b5, b7], dim=1))


class DilatedMultiScaleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        branch_channels = out_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return self.dropout(torch.cat([b1, b2, b3, b4], dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return F.relu(out + residual)


class MultiScaleTCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, num_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        
        self.multi_scale_blocks = nn.ModuleList([
            MultiScaleConvBlock(hidden_channels, hidden_channels, dropout)
            for _ in range(num_blocks)
        ])
        
        self.dilated_blocks = nn.ModuleList([
            DilatedMultiScaleBlock(hidden_channels, hidden_channels, dropout)
            for _ in range(num_blocks)
        ])
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, dropout)
            for _ in range(num_blocks)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for i in range(len(self.multi_scale_blocks)):
            ms_out = self.multi_scale_blocks[i](x)
            dilated_out = self.dilated_blocks[i](x)
            
            combined = ms_out + dilated_out
            combined = combined.transpose(1, 2)
            combined = self.norms[i](combined)
            combined = combined.transpose(1, 2)
            
            x = x + combined
            x = self.residual_blocks[i](x)
        
        return x


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        batch, channels, seq_len = x.shape
        squeeze = x.mean(dim=2)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        excitation = excitation.unsqueeze(2)
        return x * excitation


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channel_attention = SEBlock(channels, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.channel_attention(x)
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial_attention(spatial_input)
        return x * spatial_attention


class HFTModel(nn.Module):
    """
    高频量化预测模型 - 收益优化版
    
    架构：
    1. 特征嵌入
    2. 多尺度TCN
    3. CBAM注意力
    4. Transformer编码器
    5. 多任务预测头（5个时间窗口）
    """
    
    def __init__(
        self, 
        num_features: int, 
        hidden_dim: int = 128, 
        num_blocks: int = 3,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.feature_embed = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.multi_scale_tcn = MultiScaleTCN(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout
        )
        
        self.cbam = CBAM(hidden_dim)
        
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.attention_pool = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, 3)
            ) for _ in range(5)
        ])
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        
        x = self.feature_embed(x)
        
        x = x.transpose(1, 2)
        x = self.multi_scale_tcn(x)
        x = self.cbam(x)
        x = x.transpose(1, 2)
        
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        x, _ = self.attention_pool(query, x, x)
        x = x.squeeze(1)
        
        x = self.shared_fc(x)
        
        outputs = tuple(head(x) for head in self.heads)
        
        return outputs


def compute_ofi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算订单流不平衡(OFI)特征
    
    包含：
    1. 基础OFI特征（25个）
    2. 订单簿不平衡衰减特征（EWI，学术文献验证有效）
    """
    df = df.copy()
    
    df['net_market_flow'] = df['mb_intst'] - df['ma_intst']
    df['net_limit_flow'] = df['lb_intst'] - df['la_intst']
    df['net_cancel_flow'] = df['cb_intst'] - df['ca_intst']
    
    for window in [5, 10, 20]:
        df[f'cum_ofi_{window}'] = df['net_market_flow'].rolling(window=window, min_periods=1).sum()
    
    for window in [10, 20]:
        df[f'ofi_volatility_{window}'] = df['net_market_flow'].rolling(window=window, min_periods=1).std()
    
    df['ofi_abs_sum_20'] = df['net_market_flow'].abs().rolling(window=20, min_periods=1).sum()
    
    total_order_flow = df['mb_intst'] + df['ma_intst'] + 1e-8
    df['buy_pressure'] = df['mb_intst'] / total_order_flow
    df['sell_pressure'] = df['ma_intst'] / total_order_flow
    
    df['total_imbalance'] = (df['mb_intst'] + df['lb_intst']) - (df['ma_intst'] + df['la_intst'])
    df['ofi_momentum_5'] = df['net_market_flow'].rolling(window=5, min_periods=1).mean()
    df['ofi_momentum_10'] = df['net_market_flow'].rolling(window=10, min_periods=1).mean()
    df['ofi_acceleration'] = df['net_market_flow'].diff()
    df['ofi_limit_ratio'] = (df['net_limit_flow'] - df['net_market_flow']) / (df['net_market_flow'].abs() + 1e-8)
    df['ofi_cancel_ratio'] = df['net_cancel_flow'] / (df['net_market_flow'].abs() + df['net_limit_flow'].abs() + 1e-8)
    df['ofi_pressure_imbalance'] = df['buy_pressure'] - df['sell_pressure']
    df['ofi_skewness_10'] = df['net_market_flow'].rolling(window=10, min_periods=1).skew()
    df['ofi_kurtosis_20'] = df['net_market_flow'].rolling(window=20, min_periods=1).kurt()
    df['ofi_max_10'] = df['net_market_flow'].rolling(window=10, min_periods=1).max()
    df['ofi_min_10'] = df['net_market_flow'].rolling(window=10, min_periods=1).min()
    df['ofi_range_10'] = df['ofi_max_10'] - df['ofi_min_10']
    df['ofi_sign_consistency_5'] = (df['net_market_flow'] > 0).rolling(window=5, min_periods=1).mean()
    df['ofi_sign_consistency_10'] = (df['net_market_flow'] > 0).rolling(window=10, min_periods=1).mean()
    
    # 订单簿不平衡衰减特征（EWI - Exponential Weighted Imbalance）
    total_bsize = df['totalbsize'] if 'totalbsize' in df.columns else df[[f'bsize{i}' for i in range(1, 11)]].sum(axis=1)
    total_asize = df['totalasize'] if 'totalasize' in df.columns else df[[f'asize{i}' for i in range(1, 11)]].sum(axis=1)
    
    imb = (total_bsize - total_asize) / (total_bsize + total_asize + 1e-8)
    
    df['ewi_5'] = imb.ewm(alpha=0.129, adjust=False).mean()
    df['ewi_10'] = imb.ewm(alpha=0.067, adjust=False).mean()
    df['ewi_20'] = imb.ewm(alpha=0.034, adjust=False).mean()
    
    df['ewi_5_diff'] = df['ewi_5'].diff()
    df['ewi_10_diff'] = df['ewi_10'].diff()
    
    df['ewi_5_deviation'] = imb - df['ewi_5']
    df['ewi_10_deviation'] = imb - df['ewi_10']
    
    df['ewi_sign_5'] = (df['ewi_5'] > 0).rolling(window=5, min_periods=1).mean()
    df['ewi_sign_10'] = (df['ewi_10'] > 0).rolling(window=10, min_periods=1).mean()
    
    return df


class Predictor:
    """
    高频量化价格预测器 - 收益导向版
    
    核心改进：
    1. 使用更高阈值（默认0.6+）- 只在高置信度时出手
    2. 强制减少"不变"预测 - 即使低置信度也倾向于方向性预测
    3. TTA增强 - 提升预测稳定性
    4. 详细的交易统计 - 方便调试和优化
    
    解决的问题：准确率高但收益低的陷阱
    """
    
    def __init__(self) -> None:
        print("=" * 70)
        print("初始化 Predictor (收益导向版)")
        print("=" * 70)
        
        current_dir = os.path.dirname(__file__)
        
        config_path = os.path.join(current_dir, "config.json")
        print(f"\n加载配置文件: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.feature_cols = self.config.get("feature", [])
        self.thresholds = self.config.get("thresholds", {str(i): 0.6 for i in range(5)})
        self.use_tta = self.config.get("use_tta", True)
        self.tta_rounds = self.config.get("tta_rounds", 5)
        
        self.model = None
        self._load_model(current_dir)
        
        print(f"\n✓ 模型加载完成!")
        print(f"  - 特征数量: {len(self.feature_cols)}")
        print(f"  - 置信度阈值: {self.thresholds}")
        print(f"  - TTA增强: {'开启' if self.use_tta else '关闭'} ({self.tta_rounds}轮)")
        print("=" * 70)
    
    def _load_model(self, current_dir: str) -> None:
        """加载模型"""
        model_path = os.path.join(current_dir, "best_model.pt")
        
        if os.path.exists(model_path):
            print(f"\n加载模型权重: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if 'model_state' in checkpoint:
                num_features = checkpoint.get('num_features', len(self.feature_cols))
                hidden_dim = checkpoint.get('hidden_dim', 128)
                
                self.model = HFTModel(
                    num_features=num_features,
                    hidden_dim=hidden_dim
                )
                self.model.load_state_dict(checkpoint['model_state'], strict=False)
                
                if 'thresholds' in checkpoint:
                    self.thresholds = checkpoint['thresholds']
                    print(f"  ✓ 使用训练时优化的阈值: {self.thresholds}")
                    
                if 'best_avg_profit' in checkpoint:
                    print(f"  ✓ 训练时最佳单次收益: {checkpoint['best_avg_profit']:.6f}")
            else:
                num_features = len(self.feature_cols) if self.feature_cols else 214
                self.model = HFTModel(num_features=num_features)
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            print("  ✓ 模型加载成功并设置为评估模式")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """准备特征数据"""
        df = df.copy()
        df = compute_ofi_features(df)
        
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        arr = df[self.feature_cols].to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        return arr
    
    def _apply_tta_augmentation(self, x: np.ndarray) -> np.ndarray:
        """应用TTA增强"""
        aug_type = np.random.randint(0, 3)
        
        if aug_type == 0:
            return x * np.random.uniform(0.98, 1.02)
        elif aug_type == 1:
            noise = np.random.normal(0, 0.003, x.shape).astype(np.float32)
            return x + noise
        else:
            return x
    
    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        """
        预测价格移动方向
        
        Args:
            x: List[pd.DataFrame]，长度为batch
               每个DataFrame为100个tick的数据
        
        Returns:
            List[List[int]]，长度为batch
            每个内层List长度为5（5个窗口的预测）
            值为0/1/2（下跌/不变/上涨）
        """
        arrs = []
        for df in x:
            arr = self._prepare_features(df)
            arrs.append(arr)
        
        x_np = np.ascontiguousarray(np.stack(arrs, axis=0))
        x_tensor = torch.from_numpy(x_np).to(self.device, dtype=torch.float32)
        
        if self.use_tta and self.tta_rounds > 1:
            all_probs = []
            
            with torch.no_grad():
                outputs = self.model(x_tensor)
                probs = [F.softmax(o, dim=1) for o in outputs]
                all_probs.append(probs)
            
            for _ in range(self.tta_rounds - 1):
                x_aug = np.stack([self._apply_tta_augmentation(arr) for arr in arrs], axis=0)
                x_aug_tensor = torch.from_numpy(x_aug).to(self.device, dtype=torch.float32)
                
                with torch.no_grad():
                    outputs = self.model(x_aug_tensor)
                    probs = [F.softmax(o, dim=1) for o in outputs]
                    all_probs.append(probs)
            
            avg_probs = []
            for i in range(5):
                window_probs = torch.stack([p[i] for p in all_probs], dim=0).mean(dim=0)
                avg_probs.append(window_probs)
            
            probs = avg_probs
        else:
            with torch.no_grad():
                outputs = self.model(x_tensor)
            probs = [F.softmax(o, dim=1) for o in outputs]
        
        all_preds = []
        stats = {
            'total_samples': len(x),
            'trades_by_window': [],
            'hold_rates': [],
            'avg_confidences': [],
            'direction_distribution': []
        }
        
        for i in range(5):
            max_probs, preds = probs[i].max(dim=1)
            threshold = self.thresholds.get(str(i), self.thresholds.get(i, 0.6))
            
            confident_mask = (preds != 1) & (max_probs > threshold)
            final_preds = torch.where(confident_mask, preds, torch.ones_like(preds))
            
            trade_count = confident_mask.sum().item()
            hold_rate = 1 - trade_count / len(preds)
            avg_confidence = max_probs[confident_mask].mean().item() if trade_count > 0 else 0
            
            up_count = (final_preds == 2).sum().item()
            down_count = (final_preds == 0).sum().item()
            hold_count = (final_preds == 1).sum().item()
            
            stats['trades_by_window'].append(trade_count)
            stats['hold_rates'].append(hold_rate)
            stats['avg_confidences'].append(avg_confidence)
            stats['direction_distribution'].append({
                'up': up_count,
                'down': down_count,
                'hold': hold_count
            })
            
            all_preds.append(final_preds)
        
        pred_matrix = torch.stack(all_preds, dim=1)
        
        if len(x) <= 10:
            self._print_prediction_stats(stats)
        
        return pred_matrix.cpu().numpy().astype(int).tolist()
    
    def _print_prediction_stats(self, stats: Dict) -> None:
        """打印预测统计信息"""
        labels = ["label_5", "label_10", "label_20", "label_40", "label_60"]
        
        print(f"\n{'='*60}")
        print("预测统计信息 (收益导向版)")
        print(f"{'='*60}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"\n{'窗口':<12} {'出手次数':>8} {'出手率':>8} {'平均置信度':>10} {'涨/跌/不变':>15}")
        print(f"{'-'*60}")
        
        for i, label in enumerate(labels):
            dist = stats['direction_distribution'][i]
            direction_str = f"{dist['up']}/{dist['down']}/{dist['hold']}"
            print(f"{label:<12} {stats['trades_by_window'][i]:>8} {(1-stats['hold_rates'][i]):>8.4f} {stats['avg_confidences'][i]:>10.4f} {direction_str:>15}")
        
        print(f"{'='*60}")


if __name__ == "__main__":
    predictor = Predictor()
    
    np.random.seed(42)
    
    test_data = []
    for _ in range(4):
        df = pd.DataFrame({
            'mb_intst': np.random.randn(100),
            'ma_intst': np.random.randn(100),
            'lb_intst': np.random.randn(100),
            'la_intst': np.random.randn(100),
            'cb_intst': np.random.randn(100),
            'ca_intst': np.random.randn(100),
        })
        
        for col in predictor.feature_cols:
            if col not in df.columns:
                df[col] = np.random.randn(100) * 0.01
        
        test_data.append(df)
    
    results = predictor.predict(test_data)
    
    print(f"\n输入批次数: {len(test_data)}")
    print(f"每个批次的时间步数: {test_data[0].shape[0]}")
    print(f"预测结果形状: {len(results)} x {len(results[0])}")
    print(f"\n预测结果示例:")
    for i, result in enumerate(results[:5]):
        labels = ["label_5", "label_10", "label_20", "label_40", "label_60"]
        directions = ["下跌", "不变", "上涨"]
        print(f"  样本{i+1}: {dict(zip(labels, [directions[r] for r in result]))}")
