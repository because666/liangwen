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


def create_encoder(input_dim: int = 40, hidden_dim: int = 128, num_layers: int = 3,
                   **kwargs) -> TKANEncoder:
    """创建 T-KAN 编码器"""
    defaults = {'grid_size': 8, 'spline_order': 3, 'dropout': 0.15}
    defaults.update(kwargs)
    return TKANEncoder(input_dim, hidden_dim, num_layers, **defaults)


def count_parameters(model: nn.Module) -> int:
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    encoder = create_encoder(input_dim=40, hidden_dim=128, num_layers=3)
    print(f"编码器参数量: {count_parameters(encoder):,}")

    x = torch.randn(4, 100, 40)
    features = encoder(x)
    print(f"输入形状: {x.shape}")
    print(f"编码器输出形状: {features.shape}")


