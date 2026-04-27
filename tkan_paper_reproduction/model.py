"""
TKAN: Temporal Kolmogorov-Arnold Networks - PyTorch 实现

论文: TKAN: Temporal Kolmogorov-Arnold Networks (arXiv:2405.07344v4)
作者: Rémi Genet, Hugo Inzirillo

核心架构复现:
1. KANLinear - Kolmogorov-Arnold 线性层 (B-Spline 可学习激活函数)
2. RKANCell - 循环 KAN 单元 (短时记忆管理)
3. TKANCell - 时序 KAN 单元 (RKAN + LSTM 门控机制)
4. TKAN - 完整 TKAN 模型 (多层 TKANCell 堆叠)

适配良文杯竞赛:
- 输入: (batch, 100, input_dim) 100 tick 数据
- 输出: (batch, 5, 3) 5 个窗口的三分类预测
- 收益导向预测头: 分类 + 收益预测双分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List


class KANLinear(nn.Module):
    """Kolmogorov-Arnold 线性层

    论文公式 (2)-(6) 的 PyTorch 实现:
    - 基础线性变换: y_base = W_base @ x
    - B-Spline 变换: y_spline = sum(phi_{q,p}(x_p) * spline_weight)
    - 总输出: y = y_base + y_spline

    B-Spline 基函数使用 Cox-de Boor 递推公式计算
    grid 扩展: 在原始 grid 两端各扩展 spline_order 个节点
    基函数数量: grid_size + spline_order
    """

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3,
                 grid_range: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float64) * h + grid_range[0]
        self.register_buffer('grid', grid.float())

        num_basis = grid_size + spline_order
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, num_basis))
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.base_weight, gain=0.5)
        with torch.no_grad():
            self.spline_weight.data.normal_(0, 0.01)
        nn.init.xavier_uniform_(self.spline_scaler, gain=0.5)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """计算 B-Spline 基函数 (Cox-de Boor 递推)

        Args:
            x: (..., in_features)
        Returns:
            bases: (..., in_features, num_basis) 其中 num_basis = grid_size + spline_order
        """
        grid = self.grid
        x = x.unsqueeze(-1)

        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()

        for k in range(1, self.spline_order + 1):
            left = grid[k:-1]
            right = grid[k + 1:]

            x_left = x - left
            x_right = right - x

            denom_left = left - grid[:-(k + 1)]
            denom_right = grid[k + 1:] - grid[1:-k]

            denom_left = torch.where(denom_left.abs() < 1e-8, torch.ones_like(denom_left), denom_left)
            denom_right = torch.where(denom_right.abs() < 1e-8, torch.ones_like(denom_right), denom_right)

            bases = x_left / denom_left * bases[..., :-1] + x_right / denom_right * bases[..., 1:]

        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(x, self.base_weight)

        spline_basis = self.b_splines(x)
        spline_output = torch.einsum(
            'bij,oij->bo', spline_basis,
            self.spline_weight * self.spline_scaler.unsqueeze(-1))

        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_features)


class RKANCell(nn.Module):
    """循环 Kolmogorov-Arnold 网络单元 (论文 Section III-A)

    论文公式 (8)-(9):
    h_{l,i}(t) = W_hh * h_{l,i}(t-1) + W_hz * x_{l,i}(t)
    x_{l+1,j}(t) = sum_i phi_{l,j,i,t}(x_{l,i}(t), h_{l,i}(t))

    核心思想: 在每个 KAN 层中嵌入短时记忆管理
    """

    def __init__(self, input_size: int, hidden_size: int,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_kan = KANLinear(input_size, hidden_size,
                                   grid_size, spline_order)
        self.recurrent_kan = KANLinear(hidden_size, hidden_size,
                                       grid_size, spline_order)

    def forward(self, x: torch.Tensor,
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_size) 当前时间步输入
            h_prev: (batch, hidden_size) 前一时间步隐藏状态
        Returns:
            h: (batch, hidden_size) 当前隐藏状态
        """
        input_part = self.input_kan(x)
        recurrent_part = self.recurrent_kan(h_prev)
        h = input_part + recurrent_part
        return h


class TKANCell(nn.Module):
    """时序 Kolmogorov-Arnold 网络单元 (论文 Section III-B)

    论文公式 (11)-(16):
    f_t = sigma(W_f @ x_t + U_f @ h_{t-1} + b_f)  遗忘门
    i_t = sigma(W_i @ x_t + U_i @ h_{t-1} + b_i)  输入门
    o_t = sigma(KAN(x_t, t))                        输出门 (使用KAN)
    c_t = f_t * c_{t-1} + i_t * tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)
    h_t = o_t * tanh(c_t)

    核心创新: 输出门使用 KAN 替代标准线性变换
    """

    def __init__(self, input_size: int, hidden_size: int,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)

        self.output_kan = KANLinear(input_size + hidden_size, hidden_size,
                                    grid_size, spline_order)

    def forward(self, x: torch.Tensor,
                state: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_size)
            state: (h_prev, c_prev) 各为 (batch, hidden_size)
        Returns:
            h_t: (batch, hidden_size)
            c_t: (batch, hidden_size)
        """
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=-1)

        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined))
        c_tilde = torch.tanh(self.cell_gate(combined))
        o_t = torch.sigmoid(self.output_kan(combined))

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class TKANLayer(nn.Module):
    """TKAN 层: 包装 TKANCell 处理完整序列

    逐时间步处理序列数据, 维护隐藏状态和细胞状态
    """

    def __init__(self, input_size: int, hidden_size: int,
                 grid_size: int = 5, spline_order: int = 3,
                 return_sequences: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        self.cell = TKANCell(input_size, hidden_size, grid_size, spline_order)

    def forward(self, x: torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, input_size)
            state: 可选初始状态
        Returns:
            output: (batch, seq_len, hidden_size) 或 (batch, hidden_size)
            state: (h_t, c_t)
        """
        batch_size = x.size(0)
        device = x.device

        if state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
            c = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h, c = state

        outputs = []
        for t in range(x.size(1)):
            h, c = self.cell(x[:, t, :], (h, c))
            outputs.append(h)

        if self.return_sequences:
            output = torch.stack(outputs, dim=1)
        else:
            output = h

        return output, (h, c)


class FeatureEmbedding(nn.Module):
    """特征嵌入层: 将原始特征映射到隐藏空间"""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class ProfitGuidedHead(nn.Module):
    """收益导向预测头 (双分支设计)

    分类分支: 预测方向 (0=下跌, 1=不变, 2=上涨)
    收益分支: 预测期望收益 (辅助训练)
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64,
                 num_windows: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_windows = num_windows

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 3) for _ in range(num_windows)
        ])

        self.return_predictors = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_windows)
        ])

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (batch, num_windows, 3)
            return_pred: (batch, num_windows)
        """
        h = self.shared(x)
        logits = torch.stack(
            [cls(h) for cls in self.classifiers], dim=1)
        return_pred = torch.stack(
            [pred(h).squeeze(-1) for pred in self.return_predictors], dim=1)
        return logits, return_pred


class TKANModel(nn.Module):
    """TKAN 完整模型 (论文核心架构复现)

    架构流程 (论文 Figure 1-2):
    输入: (batch, 100, input_dim)
      -> FeatureEmbedding: Linear(input_dim, hidden_dim) + LayerNorm + GELU
      -> TKANLayer 1: return_sequences=True  (完整序列输出)
      -> TKANLayer 2: return_sequences=False (最后隐藏状态)
      -> Dense: Linear(hidden_dim, output_dim)

    适配良文杯:
      -> ProfitGuidedHead: 分类 + 收益预测双分支
      -> 输出: (batch, 5, 3) 分类 + (batch, 5) 收益
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_tkan_layers: int = 2, num_windows: int = 5,
                 grid_size: int = 5, spline_order: int = 3,
                 dropout: float = 0.15):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_windows = num_windows

        self.feature_embedding = FeatureEmbedding(
            input_dim, hidden_dim, dropout=0.1)

        self.tkan_layers = nn.ModuleList()
        for i in range(num_tkan_layers):
            return_seq = (i < num_tkan_layers - 1)
            self.tkan_layers.append(TKANLayer(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                return_sequences=return_seq,
            ))

        self.prediction_head = ProfitGuidedHead(
            hidden_dim, hidden_dim // 2, num_windows, dropout)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, num_windows, 3)
            return_pred: (batch, num_windows)
        """
        x = self.feature_embedding(x)

        state = None
        for layer in self.tkan_layers:
            x, state = layer(x, state)

        if x.dim() == 3:
            x = x[:, -1, :]

        logits, return_pred = self.prediction_head(x)
        return logits, return_pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推理时只返回分类结果"""
        logits, _ = self.forward(x)
        return logits.argmax(dim=-1)


def create_model(input_dim: int, num_windows: int = 5, **kwargs) -> TKANModel:
    """创建 TKAN 模型"""
    defaults = {
        'hidden_dim': 128,
        'num_tkan_layers': 2,
        'grid_size': 5,
        'spline_order': 3,
        'dropout': 0.15,
    }
    defaults.update(kwargs)
    return TKANModel(input_dim, num_windows=num_windows, **defaults)


def count_parameters(model: nn.Module) -> int:
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = create_model(input_dim=53, num_windows=5)
    print(f"模型参数量: {count_parameters(model):,}")

    x = torch.randn(2, 100, 53)
    logits, return_pred = model(x)
    print(f"输入形状: {x.shape}")
    print(f"分类输出形状: {logits.shape}")
    print(f"收益预测形状: {return_pred.shape}")
    print(f"预测结果: {model.predict(x)}")
