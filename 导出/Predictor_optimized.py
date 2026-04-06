"""
优化版预测器 - 针对收益率优化

改进点：
1. 降低"不变"预测比例，增加交易次数
2. 置信度阈值动态调整
3. 强制最小交易比例
"""

from __future__ import annotations

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        import math
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


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.padding = padding
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)[:, :, :-self.padding] if self.padding > 0 else self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)[:, :, :-self.padding] if self.padding > 0 else self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(residual)
        out = out + residual
        return self.relu(out)


class TCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=[64, 128], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(hidden_channels)):
            dilation = 2 ** i
            in_c = in_channels if i == 0 else hidden_channels[i - 1]
            out_c = hidden_channels[i]
            layers.append(TemporalBlock(in_c, out_c, kernel_size, dilation=dilation, dropout=dropout))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        x = self.pos_encoder(x)
        return self.transformer(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x):
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        attn_output, _ = self.attention(query, x, x)
        return attn_output.squeeze(1)


class HFTPredictor(nn.Module):
    def __init__(self, num_features, num_classes_per_head=[3, 3, 3, 3, 3], hidden_dim=128, dropout=0.3):
        super().__init__()
        self.num_classes_per_head = num_classes_per_head
        
        self.feature_embed = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.tcn = TCN(hidden_dim, [64, 128, 128], kernel_size=3, dropout=dropout)
        self.transformer = TransformerEncoder(d_model=128, nhead=4, num_layers=2, dropout=dropout)
        self.attention_pool = AttentionPooling(128, num_heads=4)
        
        self.shared_fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, num_classes)
            ) for num_classes in num_classes_per_head
        ])
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self.feature_embed(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = self.attention_pool(x)
        x = self.shared_fc(x)
        return tuple(head(x) for head in self.heads)


class Predictor:
    """优化版集成预测器 - 针对收益率优化"""
    
    def __init__(self) -> None:
        current_dir = os.path.dirname(__file__)
        
        config_path = os.path.join(current_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = self.config.get("feature", [])
        
        self.models = []
        self.weights = []
        self._load_ensemble_models(current_dir)
        
        # 关键参数：降低"不变"预测比例
        self.neutral_threshold = 0.50  # 只有置信度>50%才预测"不变"
        self.min_trade_ratio = 0.4     # 至少40%的交易比例
        
        print(f"集成模型数量: {len(self.models)}")
        print(f"中性阈值: {self.neutral_threshold}")
        print(f"最小交易比例: {self.min_trade_ratio}")
    
    def _load_ensemble_models(self, current_dir: str) -> None:
        model_files = sorted(glob.glob(os.path.join(current_dir, "model_seed*.pt")))
        
        if not model_files:
            raise FileNotFoundError("未找到模型文件 model_seed*.pt")
        
        # 等权重
        self.weights = [1.0 / len(model_files)] * len(model_files)
        
        for model_file in model_files:
            try:
                checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
                
                model = HFTPredictor(
                    num_features=len(self.feature_cols),
                    num_classes_per_head=[3, 3, 3, 3, 3],
                    hidden_dim=128,
                    dropout=0.3
                )
                
                model.load_state_dict(checkpoint, strict=False)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"加载模型: {os.path.basename(model_file)}")
            except Exception as e:
                print(f"加载模型失败 {model_file}: {e}")
    
    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        arrs = []
        for df in x:
            # 创建新DataFrame避免修改原数据
            df_copy = df.copy()
            for col in self.feature_cols:
                if col not in df_copy.columns:
                    df_copy[col] = 0.0
            arr = df_copy[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arrs.append(arr)
        
        x_np = np.ascontiguousarray(np.stack(arrs, axis=0))
        x_tensor = torch.from_numpy(x_np).to(self.device, dtype=torch.float32)
        
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(x_tensor)
                probs = [F.softmax(o, dim=1) for o in outputs]
                all_probs.append(probs)
        
        # 加权平均
        num_heads = len(all_probs[0])
        ensemble_probs = []
        for head_idx in range(num_heads):
            head_probs = torch.zeros_like(all_probs[0][head_idx])
            for model_idx, probs in enumerate(all_probs):
                head_probs += self.weights[model_idx] * probs[head_idx]
            ensemble_probs.append(head_probs)
        
        # 预测 - 优化策略
        predictions = []
        for b in range(ensemble_probs[0].size(0)):
            sample_preds = []
            sample_probs = []
            
            for head_idx, prob in enumerate(ensemble_probs):
                p = prob[b].cpu().numpy()
                
                # 策略1：只有当"不变"置信度足够高时才预测不变
                if p[1] > self.neutral_threshold:
                    pred = 1  # 不变
                elif p[0] > p[2]:
                    pred = 0  # 下跌
                else:
                    pred = 2  # 上涨
                
                sample_preds.append(pred)
                sample_probs.append(p)
            
            # 策略2：强制最小交易比例
            trade_count = sum(1 for p in sample_preds if p != 1)
            trade_ratio = trade_count / len(sample_preds)
            
            if trade_ratio < self.min_trade_ratio:
                # 需要将一些"不变"改为涨跌
                for i in range(len(sample_preds)):
                    if sample_preds[i] == 1:
                        p = sample_probs[i]
                        # 根据涨跌概率的相对大小决定
                        if p[0] > p[2]:
                            sample_preds[i] = 0
                        else:
                            sample_preds[i] = 2
                        
                        # 检查是否达到最小交易比例
                        trade_count = sum(1 for p in sample_preds if p != 1)
                        if trade_count / len(sample_preds) >= self.min_trade_ratio:
                            break
            
            predictions.append(sample_preds)
        
        return predictions


if __name__ == "__main__":
    predictor = Predictor()
    
    np.random.seed(42)
    test_data = [pd.DataFrame(np.random.randn(100, 35)) for _ in range(4)]
    
    results = predictor.predict(test_data)
    
    print(f"\n输入批次数: {len(test_data)}")
    print(f"预测结果形状: {len(results)} x {len(results[0])}")
    print(f"预测结果示例:")
    for i, result in enumerate(results):
        labels = ["label_5", "label_10", "label_20", "label_40", "label_60"]
        directions = ["下跌", "不变", "上涨"]
        print(f"  样本{i+1}: {dict(zip(labels, [directions[r] for r in result]))}")
