"""
T-KAN + LightGBM 二分类预测器（阈值拒绝版本）

评测规范：
1. 使用 os.path.dirname(__file__) 加载模型（绝对路径）
2. 处理 Polars DataFrame（兼容 Pandas/Polars/PyArrow）
3. 只使用 config.json 中列出的 40 维原始价量特征
4. T-KAN 编码器提取 128 维特征 → LightGBM 二分类 → 阈值决策
5. 返回 List[List[int]] 格式

架构流程：
输入 DataFrame(100, 40) → numpy 提取 → 归一化 → T-KAN 编码器(128维)
→ LightGBM 二分类 x 5 → p_up 概率 → 阈值判断 → List[int] (5个窗口方向)

决策逻辑：
- 若 p_up > τ: 预测上涨 (2)
- 若 (1 - p_up) > τ: 预测下跌 (0)
- 否则: 预测不变 (1) — 拒绝出手

模型加载策略：
优先从检查点内嵌的 lgbm_models 字典加载（避免路径编码问题），
回退到独立 .txt 文件加载。
"""

import os
import json
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import pyarrow as pa
except ImportError:
    pa = None


FEATURE_COLS = [
    'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'bid6', 'bid7', 'bid8', 'bid9', 'bid10',
    'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask6', 'ask7', 'ask8', 'ask9', 'ask10',
    'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'bsize6', 'bsize7', 'bsize8', 'bsize9', 'bsize10',
    'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'asize6', 'asize7', 'asize8', 'asize9', 'asize10',
]

WINDOW_SIZES = [5, 10, 20, 40, 60]


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


class TKANEncoder(nn.Module):
    """T-KAN 编码器"""

    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, num_layers: int = 3,
                 grid_size: int = 8, spline_order: int = 3, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_embedding(x)
        for layer in self.tkan_layers:
            x = layer(x)
        x = x.mean(dim=1)
        return x


def df_to_numpy(df) -> tuple:
    """将 DataFrame 转换为 numpy 数组

    兼容 Pandas/Polars/PyArrow 多种格式。

    参数：
        df: 输入 DataFrame
    返回：
        (numpy_array, column_names)
    """
    if isinstance(df, pd.DataFrame):
        cols = list(df.columns)
        arr = df.to_numpy(dtype=np.float32, copy=False)
        return arr, cols

    if hasattr(df, 'to_pandas'):
        try:
            pdf = df.to_pandas()
            cols = list(pdf.columns)
            arr = pdf.to_numpy(dtype=np.float32, copy=False)
            return arr, cols
        except Exception:
            pass

    if pa is not None and (hasattr(df, 'to_arrow') or hasattr(df, '__arrow_table__')):
        try:
            if hasattr(df, 'to_arrow'):
                table = df.to_arrow()
            else:
                table = df.__arrow_table__()
            pdf = table.to_pandas()
            cols = list(pdf.columns)
            arr = pdf.to_numpy(dtype=np.float32, copy=False)
            return arr, cols
        except Exception:
            pass

    if hasattr(df, 'columns') and hasattr(df, '__len__'):
        cols = list(df.columns)
        data = []
        for col in df.columns:
            col_data = df[col]
            if hasattr(col_data, 'to_numpy'):
                data.append(col_data.to_numpy().astype(np.float32))
            elif hasattr(col_data, 'to_list'):
                data.append(np.array(col_data.to_list(), dtype=np.float32))
            else:
                data.append(np.zeros(len(df), dtype=np.float32))
        arr = np.column_stack(data)
        return arr, cols

    return np.zeros((0, 0), dtype=np.float32), []


def extract_features_from_df(arr: np.ndarray, cols: list,
                             feature_cols: list) -> np.ndarray:
    """从 numpy 数组提取指定特征列

    参数：
        arr: (n_ticks, n_cols) numpy 数组
        cols: 列名列表
        feature_cols: 需要提取的特征列名
    返回：
        (n_ticks, len(feature_cols)) numpy 数组
    """
    col_idx = {c: i for i, c in enumerate(cols)}
    n = arr.shape[0]
    result = np.zeros((n, len(feature_cols)), dtype=np.float32)

    for i, col in enumerate(feature_cols):
        if col in col_idx:
            result[:, i] = arr[:, col_idx[col]]

    return result


def clean_features(features: np.ndarray) -> np.ndarray:
    """特征清洗：处理 NaN/Inf，按列类型裁剪"""
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    for i, col in enumerate(FEATURE_COLS):
        if col.startswith('bid') or col.startswith('ask'):
            features[:, i] = np.clip(features[:, i], -0.3, 0.3)
        elif col.startswith('bsize') or col.startswith('asize'):
            features[:, i] = np.clip(features[:, i], 0, 100)
    return features


def _load_lgbm_from_checkpoint(checkpoint: dict, current_dir: str) -> dict:
    """加载 LightGBM 二分类模型

    优先从检查点内嵌的 lgbm_models 字典加载，
    回退到独立 .txt 文件加载。

    参数：
        checkpoint: PyTorch 检查点字典
        current_dir: 当前文件目录
    返回：
        {window_size: lgb.Booster} 字典
    """
    if lgb is None:
        raise ImportError("lightgbm 未安装，请检查 requirements.txt")

    lgbm_models = {}

    lgbm_model_strs = checkpoint.get('lgbm_models', None)
    if lgbm_model_strs is not None:
        for w in WINDOW_SIZES:
            key = f'w{w}'
            if key in lgbm_model_strs:
                lgbm_models[w] = lgb.Booster(model_str=lgbm_model_strs[key])
        if len(lgbm_models) == len(WINDOW_SIZES):
            return lgbm_models

    for w in WINDOW_SIZES:
        model_path = os.path.join(current_dir, f'lgbm_w{w}.txt')
        if os.path.exists(model_path):
            with open(model_path, 'r', encoding='utf-8') as f:
                model_str = f.read()
            lgbm_models[w] = lgb.Booster(model_str=model_str)

    if len(lgbm_models) < len(WINDOW_SIZES):
        missing = [w for w in WINDOW_SIZES if w not in lgbm_models]
        raise FileNotFoundError(f"LightGBM 模型文件缺失: 窗口 {missing}")

    return lgbm_models


def _load_thresholds(checkpoint: dict) -> dict:
    """加载各窗口的最优出手阈值

    参数：
        checkpoint: PyTorch 检查点字典
    返回：
        {window_size: float} 阈值字典
    """
    thresholds = checkpoint.get('thresholds', None)
    if thresholds is not None:
        result = {}
        for w in WINDOW_SIZES:
            result[w] = thresholds.get(f'{w}', 0.6)
            if isinstance(result[w], str):
                result[w] = float(result[w])
        return result

    return {w: 0.6 for w in WINDOW_SIZES}


class Predictor:
    """T-KAN + LightGBM 二分类预测器（阈值拒绝版本）

    评测平台调用规范：
    - 输入: List[pd.DataFrame]，长度为 batch
    - 每个DataFrame为100个tick的数据，列名为config.json中的feature
    - 输出: List[List[int]]，长度为 batch，每个内层List长度为5

    决策逻辑：
    - 若 p_up > τ: 预测上涨 (2)
    - 若 (1 - p_up) > τ: 预测下跌 (0)
    - 否则: 预测不变 (1) — 拒绝出手
    """

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        encoder_path = os.path.join(current_dir, 'tkan_encoder.pt')
        checkpoint = torch.load(encoder_path, map_location=self.device, weights_only=False)

        self.config = checkpoint.get('config', {})
        self.mean = checkpoint.get('mean', None)
        self.std = checkpoint.get('std', None)
        self.binary_mode = checkpoint.get('binary_mode', False)

        input_dim = self.config.get('input_dim', len(FEATURE_COLS))
        hidden_dim = self.config.get('hidden_dim', 128)
        num_layers = self.config.get('num_layers', 3)
        grid_size = self.config.get('grid_size', 8)
        spline_order = self.config.get('spline_order', 3)

        self.encoder = TKANEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=0.0,
        ).to(self.device)

        if 'ema_encoder_state' in checkpoint:
            ema_state = checkpoint['ema_encoder_state']
            if len(ema_state) > 0:
                self.encoder.load_state_dict(ema_state)
            else:
                self.encoder.load_state_dict(checkpoint['encoder_state'])
        else:
            self.encoder.load_state_dict(checkpoint['encoder_state'])

        self.encoder.eval()
        print(f"T-KAN 编码器加载成功: hidden_dim={hidden_dim}, num_layers={num_layers}")

        self.lgbm_models = _load_lgbm_from_checkpoint(checkpoint, current_dir)
        print(f"LightGBM 模型加载成功: {len(self.lgbm_models)} 个窗口")

        self.thresholds = _load_thresholds(checkpoint)
        print(f"出手阈值: {self.thresholds}")

        config_path = os.path.join(current_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                eval_config = json.load(f)
            self.feature_cols = eval_config.get('feature', FEATURE_COLS)
        else:
            self.feature_cols = FEATURE_COLS

    def preprocess(self, x: List) -> torch.Tensor:
        """预处理输入数据

        参数：
            x: List[DataFrame]，长度为 batch，每个 DataFrame 为 100 tick 数据
        返回：
            torch.Tensor: (batch, 100, feature_dim)
        """
        batch_size = len(x)
        seq_len = 100
        feature_dim = len(self.feature_cols)

        features = np.zeros((batch_size, seq_len, feature_dim), dtype=np.float32)

        for i, df in enumerate(x):
            arr, cols = df_to_numpy(df)
            df_features = extract_features_from_df(arr, cols, self.feature_cols)
            df_features = clean_features(df_features)

            actual_len = min(len(df_features), seq_len)
            features[i, :actual_len, :] = df_features[:actual_len]

        if self.mean is not None and self.std is not None:
            mean = self.mean[:feature_dim]
            std = self.std[:feature_dim]
            features = (features - mean) / std
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(features).to(self.device, dtype=torch.float32)

    def predict(self, x: List) -> List[List[int]]:
        """预测接口

        参数：
            x: List[DataFrame]，长度为 batch
        返回：
            List[List[int]]：长度为 batch，每个内层 List 长度为 5
        """
        features_tensor = self.preprocess(x)

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    encoded = self.encoder(features_tensor)
            else:
                encoded = self.encoder(features_tensor)

        encoded_np = encoded.cpu().numpy()

        batch_size = encoded_np.shape[0]
        results = []

        for i in range(batch_size):
            sample_features = encoded_np[i:i + 1]
            window_preds = []

            for w in WINDOW_SIZES:
                if w in self.lgbm_models:
                    p_up = self.lgbm_models[w].predict(sample_features)[0]
                    tau = self.thresholds.get(w, 0.6)

                    if p_up > tau:
                        pred = 2
                    elif (1 - p_up) > tau:
                        pred = 0
                    else:
                        pred = 1
                else:
                    pred = 1
                window_preds.append(pred)

            results.append(window_preds)

        return results


if __name__ == '__main__':
    predictor = Predictor()

    test_data = []
    for _ in range(4):
        df_dict = {}
        for col in FEATURE_COLS:
            if col.startswith('bid') or col.startswith('ask'):
                df_dict[col] = np.random.randn(100) * 0.01
            else:
                df_dict[col] = np.random.rand(100) * 10
        test_data.append(pd.DataFrame(df_dict))

    predictions = predictor.predict(test_data)
    print(f"预测结果: {predictions}")
    print(f"预测形状: {len(predictions)} x {len(predictions[0])}")
