"""
高频量化预测类

按照竞赛规范实现 Predictor 类
"""

from typing import List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def force_pandas(df):
    """强制转换为纯 Pandas DataFrame，彻底剥离 Polars"""
    # 检查是否是 Polars DataFrame
    if hasattr(df, '__module__'):
        if 'polars' in str(df.__module__).lower():
            # 这是 Polars DataFrame，需要转换
            pass
        elif 'pandas' in str(df.__module__).lower():
            # 已经是 Pandas，但确保是纯净的
            return pd.DataFrame({col: df[col].values for col in df.columns})
    
    # 尝试使用 pyarrow 作为中介
    try:
        import pyarrow as pa
        
        # 检查是否是 Polars DataFrame
        if hasattr(df, '__arrow_table__'):
            table = df.__arrow_table__()
            if isinstance(table, pa.Table):
                pdf = table.to_pandas()
                return pd.DataFrame({col: pdf[col].values for col in pdf.columns})
        
        # 检查是否有 to_arrow 方法
        if hasattr(df, 'to_arrow'):
            try:
                table = df.to_arrow()
                if isinstance(table, pa.Table):
                    pdf = table.to_pandas()
                    return pd.DataFrame({col: pdf[col].values for col in pdf.columns})
            except Exception:
                pass
    except ImportError:
        pass
    except Exception:
        pass
    
    # 尝试 to_pandas 方法
    if hasattr(df, 'to_pandas'):
        try:
            result = df.to_pandas()
            if isinstance(result, pd.DataFrame):
                return pd.DataFrame({col: result[col].values for col in result.columns})
        except Exception:
            pass
    
    # 逐列提取数据
    if hasattr(df, 'columns') and hasattr(df, '__len__'):
        try:
            n_rows = int(len(df))
            columns_list = list(df.columns)
            
            data = {}
            for col in columns_list:
                col_data = df[col]
                # 尝试各种方法获取数据
                values = None
                if hasattr(col_data, 'to_numpy'):
                    try:
                        values = col_data.to_numpy()
                    except Exception:
                        pass
                if values is None and hasattr(col_data, 'to_list'):
                    try:
                        values = col_data.to_list()
                    except Exception:
                        pass
                if values is None:
                    try:
                        values = list(col_data)
                    except Exception:
                        values = [0.0] * n_rows
                
                data[col] = values
            
            return pd.DataFrame(data)
        except Exception:
            pass
    
    # 最后尝试直接转换
    return pd.DataFrame(df)


def compute_enhanced_features(df) -> pd.DataFrame:
    """计算所有需要的特征"""
    df = force_pandas(df)
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"转换失败，df 类型为: {type(df)}")
    
    # 确保所有列都是纯 numpy 数组
    n = len(df)
    if n == 0:
        return df
    
    # 提取所有原始数据为 numpy 数组
    data_dict = {}
    for col in df.columns:
        try:
            val = df[col].values
            data_dict[col] = np.array(val, dtype=float)
        except Exception:
            data_dict[col] = np.zeros(n)
    
    # 计算中间价
    bid1 = data_dict.get('bid1', np.zeros(n))
    ask1 = data_dict.get('ask1', np.zeros(n))
    midprice = np.where(
        (ask1 != 0) & (bid1 != 0),
        (ask1 + bid1) / 2,
        np.where(bid1 == 0, ask1, bid1)
    )
    data_dict['midprice'] = midprice
    
    # 提取十档数据
    bid_cols = {i: data_dict.get(f'bid{i}', np.zeros(n)) for i in range(1, 11)}
    ask_cols = {i: data_dict.get(f'ask{i}', np.zeros(n)) for i in range(1, 11)}
    bsize_cols = {i: data_dict.get(f'bsize{i}', np.zeros(n)) for i in range(1, 11)}
    asize_cols = {i: data_dict.get(f'asize{i}', np.zeros(n)) for i in range(1, 11)}
    
    # 计算各档中间价
    for i in range(1, 11):
        key = f'midprice{i}'
        if key not in data_dict:
            data_dict[key] = np.where(
                (ask_cols[i] != 0) & (bid_cols[i] != 0),
                (ask_cols[i] + bid_cols[i]) / 2,
                np.where(bid_cols[i] == 0, ask_cols[i], bid_cols[i])
            )
    
    # 计算各档价差
    for i in range(1, 11):
        key = f'spread{i}'
        if key not in data_dict:
            data_dict[key] = ask_cols[i] - bid_cols[i]
    
    # 计算十档均价和均量
    data_dict['bid_mean'] = sum(bid_cols[i] for i in range(1, 11)) / 10
    data_dict['ask_mean'] = sum(ask_cols[i] for i in range(1, 11)) / 10
    data_dict['bsize_mean'] = sum(bsize_cols[i] for i in range(1, 11)) / 10
    data_dict['asize_mean'] = sum(asize_cols[i] for i in range(1, 11)) / 10
    
    # 计算总量和累计价差
    data_dict['totalbsize'] = sum(bsize_cols[i] for i in range(1, 11))
    data_dict['totalasize'] = sum(asize_cols[i] for i in range(1, 11))
    data_dict['cumspread'] = sum(ask_cols[i] - bid_cols[i] for i in range(1, 11))
    data_dict['imbalance'] = (data_dict['totalbsize'] - data_dict['totalasize']) / np.maximum(data_dict['totalbsize'] + data_dict['totalasize'], 1e-8)
    
    # 计算委买/委卖均价
    total_bid_size = sum(bsize_cols[i] for i in range(1, 11))
    total_ask_size = sum(asize_cols[i] for i in range(1, 11))
    data_dict['avgbid'] = sum(bid_cols[i] * bsize_cols[i] for i in range(1, 11)) / np.maximum(total_bid_size, 1e-8)
    data_dict['avgask'] = sum(ask_cols[i] * asize_cols[i] for i in range(1, 11)) / np.maximum(total_ask_size, 1e-8)
    
    # 计算档位不平衡
    imbalance_depth = {}
    for i in range(1, 11):
        denom = bsize_cols[i] + asize_cols[i]
        imbalance_depth[i] = np.clip((bsize_cols[i] - asize_cols[i]) / np.maximum(denom, 1e-8), -1, 1)
        data_dict[f'imbalance_{i}'] = imbalance_depth[i]
    
    # 计算累计不平衡
    data_dict['cum_imbalance_3'] = sum(imbalance_depth[i] for i in range(1, 4)) / 3
    data_dict['cum_imbalance_5'] = sum(imbalance_depth[i] for i in range(1, 6)) / 5
    data_dict['cum_imbalance_10'] = sum(imbalance_depth[i] for i in range(1, 11)) / 10
    
    # 计算压力指标
    bid_pressure_total = np.maximum(sum(bsize_cols[i] for i in range(1, 11)), 1e-8)
    ask_pressure_total = np.maximum(sum(asize_cols[i] for i in range(1, 11)), 1e-8)
    data_dict['bid_pressure_total'] = bid_pressure_total
    data_dict['ask_pressure_total'] = ask_pressure_total
    data_dict['pressure_ratio'] = (bid_pressure_total - ask_pressure_total) / (bid_pressure_total + ask_pressure_total)
    
    # 添加订单流特征（如果不存在）
    order_flow_cols = [
        'lb_intst', 'la_intst', 'mb_intst', 'ma_intst', 'cb_intst', 'ca_intst',
        'lb_ind', 'la_ind', 'mb_ind', 'ma_ind', 'cb_ind', 'ca_ind',
        'lb_acc', 'la_acc', 'mb_acc', 'ma_acc', 'cb_acc', 'ca_acc'
    ]
    for col in order_flow_cols:
        if col not in data_dict:
            data_dict[col] = np.zeros(n)
    
    # 计算价格动量
    for window in [3, 5, 10, 20, 40, 60]:
        key = f'price_momentum_{window}'
        if key not in data_dict:
            momentum = np.zeros(n)
            if window < n:
                momentum[window:] = midprice[window:] - midprice[:-window]
            data_dict[key] = momentum
    
    # 计算价格速度
    for window in [5, 10, 20, 40]:
        key = f'price_velocity_{window}'
        if key not in data_dict:
            velocity = np.zeros(n)
            if window + 1 < n:
                velocity[window + 1:] = midprice[window + 1:] - 2 * midprice[window:-1] + midprice[:-window - 1]
            data_dict[key] = velocity
    
    # 计算价格加速度
    if 'price_acceleration' not in data_dict:
        acc = np.zeros(n)
        if 2 < n:
            acc[2:] = midprice[2:] - 2 * midprice[1:-1] + midprice[:-2]
        data_dict['price_acceleration'] = acc
    
    # 计算价格位置
    for window in [5, 10, 20, 40]:
        key = f'price_position_{window}'
        if key not in data_dict:
            position = np.full(n, 0.5)
            for i in range(window, n):
                start_idx = max(0, i - window + 1)
                high_val = np.max(midprice[start_idx:i + 1])
                low_val = np.min(midprice[start_idx:i + 1])
                range_val = high_val - low_val
                if range_val > 1e-8:
                    position[i] = np.clip((midprice[i] - low_val) / range_val, 0, 1)
            data_dict[key] = position
    
    # 计算成交量和金额动量
    volume_delta = data_dict.get('volume_delta', np.zeros(n))
    amount_delta = data_dict.get('amount_delta', np.zeros(n))
    
    for window in [3, 5, 10, 20]:
        key = f'volume_momentum_{window}'
        if key not in data_dict:
            vol_cum = np.zeros(n)
            for i in range(n):
                vol_cum[i] = np.sum(volume_delta[max(0, i - window + 1):i + 1])
            data_dict[key] = vol_cum
        
        key = f'amount_momentum_{window}'
        if key not in data_dict:
            amt_cum = np.zeros(n)
            for i in range(n):
                amt_cum[i] = np.sum(amount_delta[max(0, i - window + 1):i + 1])
            data_dict[key] = amt_cum
    
    # 计算成交量比率
    if 'volume_ratio' not in data_dict:
        vol_mean = np.zeros(n)
        for i in range(n):
            vol_mean[i] = np.mean(volume_delta[max(0, i - 19):i + 1]) + 1e-8
        data_dict['volume_ratio'] = np.clip(volume_delta / vol_mean, 0, 10)
    
    # 计算波动率
    high_vals = data_dict.get('high', midprice)
    low_vals = data_dict.get('low', midprice)
    
    for window in [5, 10, 20, 40]:
        key = f'volatility_{window}'
        if key not in data_dict:
            vol_vals = np.zeros(n)
            for i in range(n):
                start_idx = max(0, i - window + 1)
                if i - start_idx > 1:
                    vol_vals[i] = np.std(midprice[start_idx:i + 1])
            data_dict[key] = vol_vals
        
        key = f'price_range_{window}'
        if key not in data_dict:
            range_out = np.zeros(n)
            for i in range(n):
                start_idx = max(0, i - window + 1)
                range_out[i] = np.max(high_vals[start_idx:i + 1]) - np.min(low_vals[start_idx:i + 1])
            data_dict[key] = range_out
    
    # 计算订单流指标
    mb_intst = data_dict.get('mb_intst', np.zeros(n))
    ma_intst = data_dict.get('ma_intst', np.zeros(n))
    lb_intst = data_dict.get('lb_intst', np.zeros(n))
    la_intst = data_dict.get('la_intst', np.zeros(n))
    cb_intst = data_dict.get('cb_intst', np.zeros(n))
    ca_intst = data_dict.get('ca_intst', np.zeros(n))
    
    if 'net_flow' not in data_dict:
        data_dict['net_flow'] = mb_intst - ma_intst
    if 'net_limit' not in data_dict:
        data_dict['net_limit'] = lb_intst - la_intst
    if 'net_cancel' not in data_dict:
        data_dict['net_cancel'] = cb_intst - ca_intst
    
    net_flow = mb_intst - ma_intst
    for window in [3, 5, 10, 20]:
        key = f'flow_momentum_{window}'
        if key not in data_dict:
            flow_cum = np.zeros(n)
            for i in range(n):
                flow_cum[i] = np.sum(net_flow[max(0, i - window + 1):i + 1])
            data_dict[key] = flow_cum
    
    if 'flow_intensity' not in data_dict:
        total_flow = np.maximum(mb_intst + ma_intst, 1e-8)
        data_dict['flow_intensity'] = np.abs(net_flow) / total_flow
    
    # 计算方向持续性
    for window in [3, 5, 10, 20]:
        key = f'direction_persistence_{window}'
        if key not in data_dict:
            persistence = np.zeros(n)
            for i in range(n):
                start_idx = max(0, i - window + 1)
                if i - start_idx > 0:
                    changes = midprice[start_idx + 1:i + 1] - midprice[start_idx:i]
                    persistence[i] = np.mean((changes > 0).astype(float)) - 0.5
            data_dict[key] = persistence
    
    # 计算动量加速度
    if 'momentum_acceleration' not in data_dict:
        acc = np.zeros(n)
        if 5 < n and 'price_momentum_5' in data_dict:
            mom5 = data_dict['price_momentum_5']
            acc[5:] = mom5[5:] - mom5[:-5]
        data_dict['momentum_acceleration'] = acc
    
    # 计算订单簿斜率
    if 'orderbook_slope' not in data_dict:
        bid_slope = (bid_cols[10] - bid_cols[1]) / 9
        ask_slope = (ask_cols[10] - ask_cols[1]) / 9
        data_dict['orderbook_slope'] = ask_slope - bid_slope
    
    # 计算加权价差
    if 'weighted_spread' not in data_dict:
        total_size = sum(bsize_cols[i] + asize_cols[i] for i in range(1, 11))
        total_size = np.maximum(total_size, 1e-8)
        weighted = sum((ask_cols[i] - bid_cols[i]) * (bsize_cols[i] + asize_cols[i]) for i in range(1, 11)) / total_size
        data_dict['weighted_spread'] = weighted
    
    # 处理 NaN 和 Inf
    for key in data_dict:
        data_dict[key] = np.nan_to_num(data_dict[key], nan=0.0, posinf=0.0, neginf=0.0)
    
    # 创建新的 DataFrame
    return pd.DataFrame(data_dict)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class TemporalBlock(nn.Module):
    """扩张因果卷积块"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)[:, :, :x.size(2)]
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.se(out)

        return F.gelu(out + residual)


class LargeTCN(nn.Module):
    """大规模TCN"""

    def __init__(self, in_channels: int, hidden_channels: int = 384, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (i % 4)
            self.blocks.append(TemporalBlock(hidden_channels, hidden_channels, 3, dilation, dropout))

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, d_model: int, num_heads: int = 8, dim_feedforward: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class MultiScalePooling(nn.Module):
    """多尺度池化"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1),
        ])
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        pooled = [pool(x).squeeze(-1) for pool in self.pools]
        out = torch.cat(pooled, dim=-1)
        return self.fc(out)


class HFTModelLarge(nn.Module):
    """大规模高频交易模型"""

    def __init__(self, num_features: int, hidden_dim: int = 384, num_tcn_layers: int = 6,
                 num_transformer_layers: int = 4, num_heads: int = 8, dropout: float = 0.12):
        super().__init__()

        self.input_embed = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.tcn = LargeTCN(hidden_dim, hidden_dim, num_tcn_layers, dropout)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_transformer_layers)
        ])

        self.multi_scale_pool = MultiScalePooling(hidden_dim)

        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2)
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout / 3),
                nn.Linear(64, 3)
            ) for _ in range(5)
        ])

    def forward(self, x):
        x = self.input_embed(x)

        x = x.transpose(1, 2)
        x = self.tcn(x)

        x = x.transpose(1, 2)
        for layer in self.transformer_layers:
            x = layer(x)

        x = x.transpose(1, 2)
        x = self.multi_scale_pool(x)

        x = self.shared_fc(x)

        return tuple(head(x) for head in self.heads)


class Predictor:
    """预测类"""

    def __init__(self):
        import json
        import os

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.json')
        model_path = os.path.join(current_dir, 'best_model.pt')

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.feature_cols = checkpoint.get('feature_cols', self.config['feature'])
        self.hidden_dim = checkpoint.get('hidden_dim', 384)
        self.thresholds = checkpoint.get('thresholds', {})

        self.model = HFTModelLarge(
            num_features=len(self.feature_cols),
            hidden_dim=self.hidden_dim,
            num_tcn_layers=6,
            num_transformer_layers=4,
            num_heads=8,
            dropout=0.12
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        """
        预测函数

        输入: List[pd.DataFrame]，长度为batch
              每个DataFrame为100个tick的数据，列名为feature
        输出: List[List[int]]，长度为batch
              每个内层List长度为label个数，值为0/1/2
        """
        batch_size = len(x)

        features = []
        for df in x:
            df = compute_enhanced_features(df)

            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0.0

            feature_data = df[self.feature_cols].values.astype(np.float32)

            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

            features.append(feature_data)

        features = np.array(features, dtype=np.float32)

        features_tensor = torch.from_numpy(features).to(self.device)

        with torch.no_grad():
            outputs = self.model(features_tensor)

            predictions = []
            for i, output in enumerate(outputs):
                probs = F.softmax(output, dim=1)
                max_probs, preds = probs.max(dim=1)

                threshold = self.thresholds.get(str(i), 0.55)

                for j in range(batch_size):
                    if max_probs[j].item() < threshold:
                        preds[j] = 1

                predictions.append(preds.cpu().numpy())

        results = []
        for i in range(batch_size):
            sample_preds = [int(predictions[j][i]) for j in range(5)]
            results.append(sample_preds)

        return results
