"""
T-KAN + LightGBM 优化版预测器

优化方案：
1. 动态阈值：根据价差和波动率动态调整出手阈值
2. 后处理过滤：连续错误暂停、价差过滤、波动率过滤
3. 回归预测：直接预测价格变化率，根据预测值大小决定是否出手
4. 两阶段模型：先判断是否值得交易，再预测方向

评测规范：
1. 使用 os.path.dirname(__file__) 加载模型（绝对路径）
2. 处理 Polars DataFrame（兼容 Pandas/Polars/PyArrow）
3. 只使用 config.json 中列出的 40 维原始价量特征
4. T-KAN 编码器提取 128 维特征 → LightGBM 预测
5. 返回 List[List[int]] 格式
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
    """将 DataFrame 转换为 numpy 数组"""
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
    """从 numpy 数组提取指定特征列"""
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


def _extract_encoder_state(state_dict):
    """从完整模型的 state_dict 中提取编码器部分"""
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            new_key = k[8:]
            encoder_state[new_key] = v
    return encoder_state


def _load_lgbm_from_checkpoint(checkpoint: dict, current_dir: str) -> dict:
    """加载 LightGBM 模型"""
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


class DynamicThreshold:
    """动态阈值调整器
    
    根据市场状态（价差、波动率）动态调整出手阈值
    """
    
    def __init__(self, base_tau: float = 0.6, 
                 spread_coef: float = 2.0,
                 volatility_coef: float = 1.0,
                 max_spread: float = 0.0005,
                 max_volatility: float = 0.001):
        """
        参数：
            base_tau: 基础阈值
            spread_coef: 价差系数（价差越大，阈值越高）
            volatility_coef: 波动率系数（波动率越大，阈值越低）
            max_spread: 最大价差（用于归一化）
            max_volatility: 最大波动率（用于归一化）
        """
        self.base_tau = base_tau
        self.spread_coef = spread_coef
        self.volatility_coef = volatility_coef
        self.max_spread = max_spread
        self.max_volatility = max_volatility
    
    def get_threshold(self, spread: float, volatility: float) -> float:
        """计算动态阈值
        
        参数：
            spread: 当前价差（ask1 - bid1）
            volatility: 历史波动率（过去20个tick的标准差）
        返回：
            动态阈值
        """
        # 价差调整：价差越大，阈值越高（减少交易）
        spread_adj = self.spread_coef * (spread / self.max_spread)
        
        # 波动率调整：波动率越大，阈值越低（增加交易）
        volatility_adj = -self.volatility_coef * (volatility / self.max_volatility)
        
        # 总调整
        tau = self.base_tau + spread_adj + volatility_adj
        
        # 限制范围 [0.5, 0.95]
        tau = np.clip(tau, 0.5, 0.95)
        
        return tau


class PostProcessFilter:
    """后处理过滤器
    
    应用额外规则过滤预测结果
    """
    
    def __init__(self, 
                 max_consecutive_errors: int = 3,
                 min_spread: float = 0.00001,
                 max_spread: float = 0.0003,
                 min_volatility: float = 0.00001,
                 max_volatility: float = 0.002):
        """
        参数：
            max_consecutive_errors: 连续错误次数上限（超过则暂停交易）
            min_spread: 最小价差（低于此值不交易）
            max_spread: 最大价差（高于此值不交易）
            min_volatility: 最小波动率（低于此值不交易）
            max_volatility: 最大波动率（高于此值不交易）
        """
        self.max_consecutive_errors = max_consecutive_errors
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        
        # 状态记录
        self.recent_predictions = []  # 最近的预测结果
        self.recent_returns = []  # 最近的实际收益
    
    def update_state(self, pred: int, actual_return: float = None):
        """更新状态
        
        参数：
            pred: 预测结果
            actual_return: 实际收益（如果已知）
        """
        self.recent_predictions.append(pred)
        if actual_return is not None:
            self.recent_returns.append(actual_return)
        
        # 只保留最近20个
        if len(self.recent_predictions) > 20:
            self.recent_predictions.pop(0)
        if len(self.recent_returns) > 20:
            self.recent_returns.pop(0)
    
    def should_trade(self, spread: float, volatility: float) -> bool:
        """判断是否应该交易
        
        参数：
            spread: 当前价差
            volatility: 历史波动率
        返回：
            True: 可以交易
            False: 应该观望
        """
        # 1. 价差过滤
        if spread < self.min_spread or spread > self.max_spread:
            return False
        
        # 2. 波动率过滤
        if volatility < self.min_volatility or volatility > self.max_volatility:
            return False
        
        # 3. 连续错误过滤
        if len(self.recent_predictions) >= self.max_consecutive_errors:
            # 检查最近N次预测是否都错了
            recent = self.recent_predictions[-self.max_consecutive_errors:]
            if all(p == 1 for p in recent):  # 都预测不变，可能是在观望
                pass
            elif len(self.recent_returns) >= self.max_consecutive_errors:
                # 检查最近N次收益是否都为负
                recent_returns = self.recent_returns[-self.max_consecutive_errors:]
                if all(r < 0 for r in recent_returns):
                    return False
        
        return True


class Predictor:
    """T-KAN + LightGBM 优化版预测器
    
    优化方案：
    1. 动态阈值：根据价差和波动率动态调整出手阈值
    2. 后处理过滤：连续错误暂停、价差过滤、波动率过滤
    3. 回归预测：直接预测价格变化率，根据预测值大小决定是否出手
    4. 两阶段模型：先判断是否值得交易，再预测方向
    """

    def __init__(self, 
                 use_dynamic_threshold: bool = True,
                 use_post_filter: bool = True,
                 use_regression: bool = False,
                 use_two_stage: bool = False):
        """
        参数：
            use_dynamic_threshold: 是否使用动态阈值
            use_post_filter: 是否使用后处理过滤
            use_regression: 是否使用回归预测
            use_two_stage: 是否使用两阶段模型
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        encoder_path = os.path.join(current_dir, 'best_model.pt')
        if not os.path.exists(encoder_path):
            encoder_path = os.path.join(current_dir, 'tkan_encoder.pt')
        checkpoint = torch.load(encoder_path, map_location=self.device, weights_only=False)

        self.config = checkpoint.get('config', {})
        self.mean = checkpoint.get('mean', None)
        self.std = checkpoint.get('std', None)
        self.use_regression = use_regression or checkpoint.get('regression_mode', False)
        self.use_two_stage = use_two_stage or checkpoint.get('two_stage_mode', False)

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
                self.encoder.load_state_dict(_extract_encoder_state(ema_state), strict=False)
            else:
                self.encoder.load_state_dict(_extract_encoder_state(checkpoint['encoder_state']), strict=False)
        else:
            self.encoder.load_state_dict(_extract_encoder_state(checkpoint['encoder_state']), strict=False)

        self.encoder.eval()
        print(f"T-KAN 编码器加载成功: hidden_dim={hidden_dim}, num_layers={num_layers}")

        self.lgbm_models = _load_lgbm_from_checkpoint(checkpoint, current_dir)
        print(f"LightGBM 模型加载成功: {len(self.lgbm_models)} 个窗口")

        # 加载阈值
        self.thresholds = checkpoint.get('thresholds', {})
        if not self.thresholds:
            self.thresholds = {w: 0.6 for w in WINDOW_SIZES}
        
        # 调整阈值：短窗口提高（更严格），长窗口降低（更宽松）
        threshold_adjust = {
            5: 0.07,    # 0.6 + 0.07 = 0.67 (更严格)
            10: 0.05,   # 0.6 + 0.05 = 0.65
            20: 0.03,   # 0.6 + 0.03 = 0.63
            40: -0.05,  # 0.6 - 0.05 = 0.55 (更宽松)
            60: -0.08,  # 0.6 - 0.08 = 0.52
        }
        for w in WINDOW_SIZES:
            if w in threshold_adjust:
                base = self.thresholds.get(w, 0.6)
                self.thresholds[w] = np.clip(base + threshold_adjust[w], 0.51, 0.95)
        
        print(f"调整后阈值: {self.thresholds}")

        # 初始化动态阈值调整器
        self.use_dynamic_threshold = use_dynamic_threshold
        if use_dynamic_threshold:
            self.dynamic_threshold = DynamicThreshold(
                base_tau=0.55,
                spread_coef=1.0,
                volatility_coef=0.5
            )
            print("启用动态阈值")

        # 初始化后处理过滤器（放宽限制）
        self.use_post_filter = use_post_filter
        if use_post_filter:
            self.post_filter = PostProcessFilter(
                max_consecutive_errors=5,
                min_spread=0.0,
                max_spread=0.001,
                min_volatility=0.0,
                max_volatility=0.01
            )
            print("启用后处理过滤")

        # 加载回归阈值（如果使用回归模式）
        if self.use_regression:
            self.regression_thresholds = checkpoint.get('regression_thresholds', 
                                                        {w: 0.0005 for w in WINDOW_SIZES})
            print(f"回归阈值: {self.regression_thresholds}")

        config_path = os.path.join(current_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                eval_config = json.load(f)
            self.feature_cols = eval_config.get('feature', FEATURE_COLS)
        else:
            self.feature_cols = FEATURE_COLS

    def preprocess(self, x: List) -> tuple:
        """预处理输入数据
        
        参数：
            x: List[DataFrame]，长度为 batch，每个 DataFrame 为 100 tick 数据
        返回：
            (features_tensor, spreads, volatilities)
            features_tensor: (batch, 100, feature_dim)
            spreads: (batch,) 价差数组
            volatilities: (batch,) 波动率数组
        """
        batch_size = len(x)
        seq_len = 100
        feature_dim = len(self.feature_cols)

        features = np.zeros((batch_size, seq_len, feature_dim), dtype=np.float32)
        spreads = np.zeros(batch_size, dtype=np.float32)
        volatilities = np.zeros(batch_size, dtype=np.float32)

        for i, df in enumerate(x):
            arr, cols = df_to_numpy(df)
            df_features = extract_features_from_df(arr, cols, self.feature_cols)
            df_features = clean_features(df_features)

            actual_len = min(len(df_features), seq_len)
            features[i, :actual_len, :] = df_features[:actual_len]
            
            # 计算价差
            col_idx = {c: idx for idx, c in enumerate(cols)}
            if 'ask1' in col_idx and 'bid1' in col_idx:
                spreads[i] = arr[-1, col_idx['ask1']] - arr[-1, col_idx['bid1']]
            
            # 计算波动率（过去20个tick）
            if 'midprice' in col_idx:
                midprice = arr[:, col_idx['midprice']]
                if len(midprice) >= 20:
                    volatilities[i] = np.std(midprice[-20:])

        if self.mean is not None and self.std is not None:
            mean = self.mean[:feature_dim]
            std = self.std[:feature_dim]
            features = (features - mean) / std
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(features).to(self.device, dtype=torch.float32), spreads, volatilities

    def predict(self, x: List) -> List[List[int]]:
        """预测接口（优化版）
        
        参数：
            x: List[DataFrame]，长度为 batch
        返回：
            List[List[int]]：长度为 batch，每个内层 List 长度为 5
        """
        features_tensor, spreads, volatilities = self.preprocess(x)

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
            current_spread = spreads[i]
            current_volatility = volatilities[i]
            
            window_preds = []

            for w in WINDOW_SIZES:
                if w not in self.lgbm_models:
                    pred_label = 1
                else:
                    if self.use_regression:
                        # 回归模式：预测价格变化率
                        pred_value = self.lgbm_models[w].predict(sample_features)[0]
                        threshold = self.regression_thresholds.get(w, 0.0005)
                        
                        if pred_value > threshold:
                            pred_label = 2  # 上涨
                        elif pred_value < -threshold:
                            pred_label = 0  # 下跌
                        else:
                            pred_label = 1  # 不变
                    else:
                        # 分类模式：预测上涨概率
                        p_up = self.lgbm_models[w].predict(sample_features)[0]
                        
                        # 获取阈值
                        if self.use_dynamic_threshold:
                            tau = self.dynamic_threshold.get_threshold(current_spread, current_volatility)
                        else:
                            tau = self.thresholds.get(w, 0.6)
                        
                        # 阈值决策
                        if p_up > tau:
                            pred_label = 2  # 上涨
                        elif (1 - p_up) > tau:
                            pred_label = 0  # 下跌
                        else:
                            pred_label = 1  # 不变
                
                # 后处理过滤
                if self.use_post_filter and pred_label != 1:
                    if not self.post_filter.should_trade(current_spread, current_volatility):
                        pred_label = 1  # 改为观望
                
                window_preds.append(pred_label)
                
                # 更新后处理过滤器状态
                if self.use_post_filter:
                    self.post_filter.update_state(pred_label)

            results.append(window_preds)

        return results


if __name__ == '__main__':
    predictor = Predictor(
        use_dynamic_threshold=False,
        use_post_filter=False,
        use_regression=False,
        use_two_stage=False
    )

    test_data = []
    for _ in range(4):
        df_dict = {}
        for col in FEATURE_COLS:
            if col.startswith('bid') or col.startswith('ask'):
                df_dict[col] = np.random.randn(100) * 0.01
            else:
                df_dict[col] = np.random.rand(100) * 10
        # 添加 midprice 列用于计算波动率
        df_dict['midprice'] = np.random.randn(100) * 0.01
        test_data.append(pd.DataFrame(df_dict))

    predictions = predictor.predict(test_data)
    print(f"预测结果: {predictions}")
    print(f"预测形状: {len(predictions)} x {len(predictions[0])}")
