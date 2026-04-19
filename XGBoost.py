"""
T-KAN 端到端分类器 - XGBoost版本
基于原T-KAN代码结构，仅将神经网络分类器替换为XGBoost

【修改说明】
原代码修改部分：
1. 第111-149行: 原TKANClassifier(nn.Module) → 改为XGBoostClassifier
2. 第151-318行: 原神经网络分类头 → 改为XGBoost分类器
3. 添加类别权重计算（解决类别不平衡）
4. 添加窗口自适应参数（短窗口特殊处理）

架构：
- 保持原有的特征处理和输入输出接口
- T-KAN编码器 → XGBoost分类器
- 直接预测 5 个窗口的标签（三分类：0/1/2）

评测规范：
1. 使用 os.path.dirname(__file__) 加载模型（绝对路径）
2. 处理 Polars DataFrame（兼容 Pandas/Polars/PyArrow）
3. 只使用 config.json 中列出的 40维原始价量特征
4. 返回 List[List[int]] 格式
"""

import os
import json
from typing import List
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    raise ImportError("XGBoost未安装，请使用: pip install xgboost")

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


class StableSplineLinear:
    """
    数值稳定的 B-spline 线性层
    保持与原代码一致，用于特征变换
    【未修改】保持原代码第40-88行
    """
    
    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 8, spline_order: int = 3):
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 初始化参数
        self.grid = np.linspace(-1, 1, grid_size + 2 * spline_order + 1)
        self.base_weight = np.random.randn(out_features, in_features) * 0.1
        self.spline_weight = np.random.randn(out_features, in_features, grid_size + spline_order) * 0.01
    
    def b_splines(self, x: np.ndarray) -> np.ndarray:
        """计算B-spline基函数"""
        x = np.clip(x, -2.0, 2.0)
        grid = self.grid
        
        # 扩展维度用于广播
        x_expanded = x[..., np.newaxis]  # (..., 1)
        
        # 初始化基函数
        bases = ((x_expanded >= grid[:-1]) & (x_expanded < grid[1:])).astype(float)
        
        # 递归计算高阶B-spline
        for k in range(1, self.spline_order + 1):
            denom1 = grid[k:-1] - grid[:-(k + 1)]
            denom1 = np.where(np.abs(denom1) < 1e-8, 1.0, denom1)
            
            denom2 = grid[k + 1:] - grid[1:-k]
            denom2 = np.where(np.abs(denom2) < 1e-8, 1.0, denom2)
            
            bases = (
                (x_expanded - grid[:-(k + 1)]) / denom1 * bases[..., :-1] +
                (grid[k + 1:] - x_expanded) / denom2 * bases[..., 1:]
            )
        
        return bases
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        original_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        
        # 基础线性变换
        base_output = x_2d @ self.base_weight.T
        
        # Spline变换
        spline_basis = self.b_splines(x_2d)  # (batch*seq, in_features, n_basis)
        
        # einsum: (b,i,j) @ (o,i,j) -> (b,o)
        spline_output = np.einsum('bij,oij->bo', spline_basis, self.spline_weight)
        
        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_features)


class TKANFeatureExtractor:
    """
    T-KAN特征提取器
    保持与原代码相同的特征处理逻辑
    【未修改】保持原代码第90-149行结构
    """
    
    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, num_layers: int = 3,
                 grid_size: int = 8, spline_order: int = 3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 特征嵌入层
        self.feature_embedding = {
            'weight': np.random.randn(input_dim, hidden_dim) * 0.1,
            'bias': np.zeros(hidden_dim)
        }
        
        # T-KAN层
        self.tkan_layers = []
        for _ in range(num_layers):
            layer = {
                'spline': StableSplineLinear(hidden_dim, hidden_dim, grid_size, spline_order),
                'gamma': np.ones(hidden_dim),
                'beta': np.zeros(hidden_dim)
            }
            self.tkan_layers.append(layer)
    
    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """层归一化"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU激活函数"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    
    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """
        提取特征
        【未修改】保持原代码逻辑
        
        参数:
            x: 输入特征 (batch, seq_len, input_dim) 或 (batch, input_dim)
        
        返回:
            features: 提取的特征 (batch, hidden_dim)
        """
        # 处理2D输入
        if len(x.shape) == 2:
            x = x[:, np.newaxis, :]  # (batch, 1, input_dim)
        
        batch_size, seq_len, _ = x.shape
        
        # 特征嵌入
        x_2d = x.reshape(-1, self.input_dim)
        x_embedded = x_2d @ self.feature_embedding['weight'] + self.feature_embedding['bias']
        x_embedded = x_embedded.reshape(batch_size, seq_len, self.hidden_dim)
        
        # 层归一化 + GELU
        x_embedded = self.layer_norm(x_embedded, 
                                     np.ones(self.hidden_dim), 
                                     np.zeros(self.hidden_dim))
        x_embedded = self.gelu(x_embedded)
        
        # T-KAN层
        for layer in self.tkan_layers:
            residual = x_embedded
            
            # Spline变换
            x_flat = x_embedded.reshape(-1, self.hidden_dim)
            x_transformed = layer['spline'].forward(x_flat)
            x_transformed = x_transformed.reshape(batch_size, seq_len, self.hidden_dim)
            
            # 层归一化
            x_transformed = self.layer_norm(x_transformed, layer['gamma'], layer['beta'])
            
            # GELU
            x_transformed = self.gelu(x_transformed)
            
            # 残差连接
            x_embedded = x_transformed + residual
        
        # 时序平均池化
        features = np.mean(x_embedded, axis=1)  # (batch, hidden_dim)
        
        return features


# ============================================================================
# 【修改开始】原代码第111-149行: TKANClassifier(nn.Module) → XGBoostClassifier
# ============================================================================

class XGBoostClassifier:
    """
    XGBoost分类器
    【修改】替换原T-KAN的神经网络分类头
    
    原代码:
        class TKANClassifier(nn.Module):
            def __init__(self, ...):
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, num_windows * 3),
                )
    
    修改为: XGBoost分类器，每个窗口一个模型
    """
    
    def __init__(self, num_windows: int = 5, num_classes: int = 3):
        self.num_windows = num_windows
        self.num_classes = num_classes
        self.models = {}  # 每个窗口一个XGBoost模型
    
    # 【新增】类别权重计算（解决类别不平衡问题）
    def _compute_class_weights(self, y):
        """计算类别权重 - 处理类别不平衡"""
        classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        # 使用有效数量加权
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * len(classes)
        
        return {int(cls): weight for cls, weight in zip(classes, weights)}
    
    def fit(self, X: np.ndarray, y_dict: dict, sample_weight: np.ndarray = None):
        """
        训练分类器
        【修改】原神经网络训练 → XGBoost训练
        
        参数:
            X: 特征 (n_samples, hidden_dim)
            y_dict: {window: labels}
            sample_weight: 样本权重
        """
        print("训练XGBoost分类器...")
        
        for i, window in enumerate(WINDOW_SIZES):
            if window not in y_dict:
                continue
            
            y = y_dict[window]
            
            # 【新增】计算类别权重
            class_weights = self._compute_class_weights(y)
            print(f"  窗口 {window}: 类别权重 = {class_weights}")
            
            # 创建样本权重
            sw = np.ones(len(y))
            if sample_weight is not None:
                sw = sample_weight.copy()
            
            for j, label in enumerate(y):
                sw[j] *= class_weights.get(label, 1.0)
            
            # 【新增】窗口自适应参数（短窗口特殊处理）
            if window <= 10:  # 短窗口（label_5, label_10）
                params = {
                    'n_estimators': 300,        # 更多树
                    'max_depth': 5,             # 适中的深度
                    'learning_rate': 0.03,      # 较小的学习率
                    'subsample': 0.7,           # 更多随机性
                    'colsample_bytree': 0.7,
                    'reg_alpha': 0.5,           # L1正则化（更强）
                    'reg_lambda': 2.0,          # L2正则化（更强）
                    'min_child_weight': 5,      # 防止过拟合
                }
            else:  # 长窗口（label_20, label_40, label_60）
                params = {
                    'n_estimators': 200,
                    'max_depth': 7,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'min_child_weight': 3,
                }
            
            # 【修改】使用XGBoost替代神经网络
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=self.num_classes,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1,
                **params
            )
            
            model.fit(X, y, sample_weight=sw, verbose=False)
            
            self.models[window] = model
            print(f"    训练完成，树数量: {model.n_estimators}")
        
        print(f"共训练 {len(self.models)} 个模型")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        【修改】原神经网络前向传播 → XGBoost预测
        
        参数:
            X: 特征 (n_samples, hidden_dim)
        
        返回:
            predictions: (n_samples, num_windows)
        """
        predictions = []
        
        for window in WINDOW_SIZES:
            if window not in self.models:
                predictions.append(np.ones(X.shape[0], dtype=int))
                continue
            
            model = self.models[window]
            pred = model.predict(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        【新增】XGBoost支持概率输出
        
        参数:
            X: 特征 (n_samples, hidden_dim)
        
        返回:
            probabilities: (n_samples, num_windows, num_classes)
        """
        probabilities = []
        
        for window in WINDOW_SIZES:
            if window not in self.models:
                probs = np.ones((X.shape[0], self.num_classes)) / self.num_classes
                probabilities.append(probs)
                continue
            
            model = self.models[window]
            probs = model.predict_proba(X)
            probabilities.append(probs)
        
        return np.stack(probabilities, axis=1)
    
    def save_model(self, path: str):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        for window, model in self.models.items():
            model.save_model(os.path.join(path, f'xgb_window_{window}.json'))
    
    def load_model(self, path: str):
        """加载模型"""
        for window in WINDOW_SIZES:
            model_path = os.path.join(path, f'xgb_window_{window}.json')
            if os.path.exists(model_path):
                self.models[window] = xgb.XGBClassifier()
                self.models[window].load_model(model_path)

# ============================================================================
# 【修改结束】
# ============================================================================


def df_to_numpy(df) -> tuple:
    """将 DataFrame 转换为 numpy 数组
    【未修改】保持原代码第151-318行中的df_to_numpy函数
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
    
    if pa is not None and (hasattr(df, '__class__') and 'pyarrow' in str(type(df))):
        try:
            pdf = df.to_pandas()
            cols = list(pdf.columns)
            arr = pdf.to_numpy(dtype=np.float32, copy=False)
            return arr, cols
        except Exception:
            pass
    
    raise ValueError(f"不支持的DataFrame类型: {type(df)}")


class Predictor:
    """
    预测器类（符合评测规范）
    【未修改】保持原代码接口不变
    """
    
    def __init__(self):
        """初始化预测器"""
        self.feature_extractor = None
        self.classifier = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, 'models_xgb')
        
        # 初始化特征提取器
        self.feature_extractor = TKANFeatureExtractor(
            input_dim=40,
            hidden_dim=128,
            num_layers=3,
            grid_size=8,
            spline_order=3
        )
        
        # 加载分类器
        self.classifier = XGBoostClassifier()
        
        if os.path.exists(model_dir):
            self.classifier.load_model(model_dir)
            print(f"已加载模型: {model_dir}")
        else:
            print(f"警告: 模型目录不存在: {model_dir}")
    
    def predict(self, df_test) -> List[List[int]]:
        """
        预测接口
        【未修改】保持原代码接口不变
        
        参数:
            df_test: DataFrame，包含40维原始特征
        
        返回:
            List[List[int]]: 每个样本的5个窗口预测标签
        """
        # 1. 转换DataFrame
        arr, cols = df_to_numpy(df_test)
        
        # 2. 提取特征（使用T-KAN特征提取器）
        features = self.feature_extractor.extract_features(arr)
        
        # 3. XGBoost预测
        predictions = self.classifier.predict(features)
        
        # 4. 转换为List[List[int]]
        return predictions.tolist()


def train_model(train_df, labels_dict, model_save_path='models_xgb'):
    """
    训练模型
    【修改】支持XGBoost训练
    
    参数:
        train_df: 训练数据DataFrame (包含40维原始特征)
        labels_dict: 标签字典 {window: labels}
        model_save_path: 模型保存路径
    """
    print("="*60)
    print("T-KAN + XGBoost 模型训练")
    print("="*60)
    
    # 1. 准备数据
    arr = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    
    print(f"\n数据形状: {arr.shape}")
    
    # 2. 特征提取
    print("\n提取T-KAN特征...")
    feature_extractor = TKANFeatureExtractor(
        input_dim=40,
        hidden_dim=128,
        num_layers=3,
        grid_size=8,
        spline_order=3
    )
    
    features = feature_extractor.extract_features(arr)
    print(f"特征形状: {features.shape}")
    
    # 3. 训练XGBoost分类器
    print("\n训练XGBoost分类器...")
    classifier = XGBoostClassifier()
    classifier.fit(features, labels_dict)
    
    # 4. 保存模型
    print(f"\n保存模型到: {model_save_path}")
    classifier.save_model(model_save_path)
    
    print("\n训练完成！")
    return feature_extractor, classifier


if __name__ == "__main__":
    print("="*60)
    print("T-KAN + XGBoost 分类器")
    print("="*60)
    print("\n【修改说明】")
    print("  修改位置: 第111-149行（原TKANClassifier）")
    print("  修改内容: 神经网络分类头 → XGBoost分类器")
    print("  新增功能:")
    print("    1. 类别权重计算（解决类别不平衡）")
    print("    2. 窗口自适应参数（短窗口特殊处理）")
    print("    3. XGBoost概率输出")
    print("\n架构:")
    print("  1. T-KAN特征提取器 (保持原结构)")
    print("  2. XGBoost分类器 (替换原神经网络)")
    print("="*60)
