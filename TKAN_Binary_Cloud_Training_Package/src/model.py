"""
T-KAN 编码器 + 回归预训练模型

架构设计：
1. 输入：40 维原始价量特征（bid1~10, ask1~10, bsize1~10, asize1~10）
2. T-KAN 编码器：3 层 TKANLayer（B-spline + LayerNorm + Dropout + 残差）
3. 时序平均池化 → 128 维特征向量
4. 回归头：预测 5 个窗口的中间价变化率（收益加权 Huber Loss 预训练用）

形状：输入 (batch, 100, 40) → 编码器 → (batch, 128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableSplineLinear(nn.Module):
    """数值稳定的 B-spline 线性层

    参数：
        in_features: 输入维度
        out_features: 输出维度
        grid_size: B-spline 网格大小
        spline_order: B-spline 阶数
    """

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
        """计算 B-spline 基函数（数值稳定版本）"""
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
    """T-KAN 单层（B-spline + LayerNorm + GELU + Dropout + 残差连接）

    参数：
        features: 特征维度
        grid_size: B-spline 网格大小
        spline_order: B-spline 阶数
        dropout: Dropout 概率
    """

    def __init__(self, features: int, grid_size: int = 8,
                 spline_order: int = 3, dropout: float = 0.15):
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
    """T-KAN 编码器

    架构流程：
    1. 特征嵌入: Linear(input_dim, hidden_dim) + LayerNorm + GELU + Dropout
    2. T-KAN 层 x num_layers: 带残差连接
    3. 时序平均池化: mean(dim=1) → 固定长度特征向量

    参数：
        input_dim: 输入特征维度（默认 40）
        hidden_dim: 隐藏层维度（默认 128）
        num_layers: T-KAN 层数（默认 3）
        grid_size: B-spline 网格大小
        spline_order: B-spline 阶数
        dropout: Dropout 概率
    """

    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, num_layers: int = 3,
                 grid_size: int = 8, spline_order: int = 3, dropout: float = 0.15):
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
        """
        参数：
            x: (batch, seq_len, input_dim)
        返回：
            features: (batch, hidden_dim)
        """
        x = self.feature_embedding(x)
        for layer in self.tkan_layers:
            x = layer(x)
        x = x.mean(dim=1)
        return x


class RegressionHead(nn.Module):
    """回归预测头

    预测 5 个窗口的中间价变化率，用于 T-KAN 编码器预训练。

    参数：
        input_dim: 输入维度（编码器输出维度）
        hidden_dim: 隐藏层维度
        num_windows: 预测窗口数
        dropout: Dropout 概率
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64,
                 num_windows: int = 5, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_windows),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TKANRegressionModel(nn.Module):
    """T-KAN 回归预训练模型

    用于预训练 T-KAN 编码器，预测 5 个窗口的中间价变化率。

    参数：
        input_dim: 输入特征维度
        hidden_dim: 编码器隐藏维度
        num_layers: T-KAN 层数
        grid_size: B-spline 网格大小
        spline_order: B-spline 阶数
        dropout: Dropout 概率
        num_windows: 预测窗口数
    """

    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, num_layers: int = 3,
                 grid_size: int = 8, spline_order: int = 3, dropout: float = 0.15,
                 num_windows: int = 5):
        super().__init__()
        self.encoder = TKANEncoder(input_dim, hidden_dim, num_layers,
                                   grid_size, spline_order, dropout)
        self.regression_head = RegressionHead(hidden_dim, 64, num_windows, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: (batch, seq_len, input_dim)
        返回：
            predictions: (batch, num_windows)
        """
        features = self.encoder(x)
        predictions = self.regression_head(features)
        return predictions

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量（用于 LightGBM 输入）"""
        return self.encoder(x)


class ProfitWeightedHuberLoss(nn.Module):
    """收益加权 Huber 损失

    对每个样本的损失乘以目标变化率的绝对值（放大后截断），
    使编码器更关注大幅价格变动的样本。

    参数：
        delta: Huber 损失的 delta 参数
        scale: 收益放大系数
        min_weight: 最小权重（防止小波动样本权重为零）
    """

    def __init__(self, delta: float = 1.0, scale: float = 10.0, min_weight: float = 0.5):
        super().__init__()
        self.delta = delta
        self.scale = scale
        self.min_weight = min_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        参数：
            predictions: (batch, num_windows)
            targets: (batch, num_windows)
        返回：
            标量损失值
        """
        loss = F.huber_loss(predictions, targets, reduction='none', delta=self.delta)
        abs_targets = targets.abs()
        weights = torch.clamp(abs_targets * self.scale, min=self.min_weight)
        weighted_loss = (loss * weights).mean()
        return weighted_loss


def create_encoder(input_dim: int = 40, hidden_dim: int = 128, num_layers: int = 3,
                   **kwargs) -> TKANEncoder:
    """创建 T-KAN 编码器"""
    defaults = {'grid_size': 8, 'spline_order': 3, 'dropout': 0.15}
    defaults.update(kwargs)
    return TKANEncoder(input_dim, hidden_dim, num_layers, **defaults)


def create_regression_model(input_dim: int = 40, hidden_dim: int = 128,
                            num_layers: int = 3, num_windows: int = 5,
                            **kwargs) -> TKANRegressionModel:
    """创建 T-KAN 回归预训练模型"""
    defaults = {'grid_size': 8, 'spline_order': 3, 'dropout': 0.15}
    defaults.update(kwargs)
    return TKANRegressionModel(input_dim, hidden_dim, num_layers,
                               num_windows=num_windows, **defaults)


def count_parameters(model: nn.Module) -> int:
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = create_regression_model(input_dim=40, hidden_dim=128, num_layers=3)
    print(f"模型参数量: {count_parameters(model):,}")

    x = torch.randn(4, 100, 40)
    predictions = model(x)
    print(f"输入形状: {x.shape}")
    print(f"回归输出形状: {predictions.shape}")

    features = model.encode(x)
    print(f"编码器输出形状: {features.shape}")

    encoder = create_encoder(input_dim=40, hidden_dim=128, num_layers=3)
    print(f"编码器参数量: {count_parameters(encoder):,}")

    loss_fn = ProfitWeightedHuberLoss(scale=10.0, min_weight=0.5)
    targets = torch.randn(4, 5) * 0.01
    loss = loss_fn(predictions, targets)
    print(f"收益加权 Huber 损失: {loss.item():.6f}")
