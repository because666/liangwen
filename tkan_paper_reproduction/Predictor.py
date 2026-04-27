"""
TKAN 论文复现 - 预测器

评测规范:
1. 使用 os.path.dirname(__file__) 加载模型
2. 处理 Polars/Pandas/PyArrow DataFrame
3. 计算衍生特征 (OFI/imbalance/cumspread)
4. 使用 GPU (如果可用)
5. 返回 List[List[int]] 格式
"""

import os
from typing import List
import numpy as np
import pandas as pd
import torch

from model import TKANModel, create_model

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


def clean_features(features: np.ndarray, feature_cols: list) -> np.ndarray:
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


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
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


class Predictor:
    """TKAN 论文复现 - 预测器

    评测平台调用规范:
    - 输入: List[pd.DataFrame]，长度为 batch
    - 输出: List[List[int]]，长度为 batch，每个内层 List 长度为 5
    """

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)

        self.config = checkpoint.get('config', {})
        self.mean = checkpoint.get('mean', None)
        self.std = checkpoint.get('std', None)
        self.best_window = checkpoint.get('best_window', None)

        input_dim = self.config.get('input_dim', len(ALL_FEATURE_COLS))
        hidden_dim = self.config.get('hidden_dim', 128)
        num_tkan_layers = self.config.get('num_tkan_layers', 2)
        grid_size = self.config.get('grid_size', 5)
        spline_order = self.config.get('spline_order', 3)

        self.model = create_model(
            input_dim=input_dim,
            num_windows=5,
            hidden_dim=hidden_dim,
            num_tkan_layers=num_tkan_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=0.0,
        )
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, x: List[pd.DataFrame]) -> torch.Tensor:
        batch_size = len(x)
        seq_len = 100
        feature_dim = len(ALL_FEATURE_COLS)

        features = np.zeros((batch_size, seq_len, feature_dim), dtype=np.float32)

        for i, df in enumerate(x):
            if hasattr(df, 'to_pandas'):
                df = df.to_pandas()

            df = compute_derived_features(df)

            for col in ALL_FEATURE_COLS:
                if col not in df.columns:
                    df[col] = 0.0

            df_features = df[ALL_FEATURE_COLS].values.astype(np.float32)
            df_features = clean_features(df_features, ALL_FEATURE_COLS)

            actual_len = min(len(df_features), seq_len)
            features[i, :actual_len, :] = df_features[:actual_len]

        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / self.std
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(features).to(self.device)

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        features = self.preprocess(x)

        with torch.no_grad():
            logits, _ = self.model(features)
            preds = logits.argmax(dim=-1)

        preds = preds.cpu().numpy().tolist()
        return preds
