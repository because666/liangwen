"""
OFI (Order Flow Imbalance) 特征计算模块

OFI是订单流不平衡指标，用于捕捉买卖压力的净方向
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def compute_ofi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算5种OFI衍生特征
    
    Args:
        df: 包含订单流统计字段的DataFrame
        
    Returns:
        添加了5列OFI特征的DataFrame
    """
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
        mb_col = f'mb_intst'
        ma_col = f'ma_intst'
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


def compute_ofi_features_batch(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    批量计算OFI特征
    
    Args:
        dfs: DataFrame列表
        
    Returns:
        添加了OFI特征的DataFrame列表
    """
    return [compute_ofi_features(df) for df in dfs]


def get_ofi_column_names() -> List[str]:
    """获取OFI特征列名"""
    return ['ofi_raw', 'ofi_ewm', 'ofi_multilevel', 'ofi_velocity', 'ofi_volatility']


def compute_price_delta(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 40, 60]) -> Dict[str, np.ndarray]:
    """
    计算价格变化率（用于回归目标）
    
    Args:
        df: 包含midprice的DataFrame
        windows: 预测窗口列表
        
    Returns:
        各窗口的价格变化率字典
    """
    midprice = df['midprice'].values if 'midprice' in df.columns else np.zeros(len(df))
    n = len(midprice)
    
    deltas = {}
    for w in windows:
        delta = np.zeros(n)
        if w < n:
            delta[:-w] = midprice[w:] - midprice[:-w]
        deltas[f'delta_{w}'] = delta
    
    return deltas


def normalize_features(df: pd.DataFrame, 
                       feature_cols: List[str], 
                       mean: np.ndarray = None, 
                       std: np.ndarray = None) -> tuple:
    """
    特征归一化
    
    Args:
        df: DataFrame
        feature_cols: 特征列名
        mean: 均值（如果为None则计算）
        std: 标准差（如果为None则计算）
        
    Returns:
        (归一化后的特征矩阵, 均值, 标准差)
    """
    features = df[feature_cols].values.astype(np.float32)
    
    if mean is None:
        mean = np.nanmean(features, axis=0)
    if std is None:
        std = np.nanstd(features, axis=0) + 1e-8
    
    features = (features - mean) / std
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features, mean, std
