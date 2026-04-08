"""
深度收益导向分层集成网络 - 多专家模型定义

核心改进（相比V2）：
1. 直接三分类预测（0/1/2），不再使用动作门
2. 多专家协同：T-KAN + 档位注意力 + Patch Transformer + 跨资产OFI
3. 动态门控融合
4. 收益加权交叉熵损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SplineLinear(nn.Module):
    """B样条线性层（T-KAN核心组件）"""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 8,
                 spline_order: int = 3):
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
        nn.init.kaiming_uniform_(self.base_weight)
        nn.init.zeros_(self.spline_weight)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            # 数值稳定性保护：防止除零
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
    """T-KAN 单层"""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 8,
                 spline_order: int = 3, dropout: float = 0.1):
        super().__init__()
        self.spline_linear = SplineLinear(in_features, out_features, grid_size, spline_order)
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spline_linear(x)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class ExpertA_TKAN(nn.Module):
    """专家A：T-KAN 时序专家 - 捕捉非线性时序演化"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64,
                 num_layers: int = 2, grid_size: int = 8, spline_order: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layers = []
        dims = [hidden_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            layers.append(TKANLayer(dims[i], dims[i + 1], grid_size, spline_order, dropout))
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.final_norm(x)


class ExpertB_LevelAttention(nn.Module):
    """专家B：档位注意力专家 - 捕捉订单簿档位间依赖"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        attn_out, _ = self.attention(h, h, h)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        h = h.mean(dim=1)
        return self.final_norm(self.output_proj(h))


class ExpertC_PatchTransformer(nn.Module):
    """专家C：Patch Transformer专家 - 并行处理时空依赖"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64,
                 patch_size: int = 5, num_heads: int = 4, num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(input_dim * patch_size, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        num_patches = T // self.patch_size
        if num_patches == 0:
            num_patches = 1
        x_trimmed = x[:, :num_patches * self.patch_size, :]
        x_patches = x_trimmed.reshape(B, num_patches, -1)
        h = self.patch_proj(x_patches)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.transformer(h)
        h = h[:, 0]
        return self.final_norm(self.output_proj(h))


class ExpertD_CrossAssetOFI(nn.Module):
    """专家D：跨资产OFI专家 - 学习多股票订单流关联"""

    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)
        h = h.mean(dim=1)
        return self.final_norm(self.output_proj(h))


class DynamicGatingFusion(nn.Module):
    """动态门控融合 - 根据市场状态动态分配专家权重"""

    def __init__(self, num_experts: int, expert_dim: int, market_state_dim: int,
                 hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(market_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        # expert_dim 应该是填充后的统一维度（即 max_dim）
        self.expert_proj = nn.ModuleList([
            nn.Linear(expert_dim, expert_dim) for _ in range(num_experts)
        ])

    def forward(self, expert_outputs: list, market_state: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate_network(market_state)
        gate_weights = F.softmax(gate_logits, dim=-1)
        projected = [proj(out) for proj, out in zip(self.expert_proj, expert_outputs)]
        stacked = torch.stack(projected, dim=1)
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)
        return fused


class ProfitGuidedHead(nn.Module):
    """收益导向预测头 - 直接三分类，不再使用动作门"""

    def __init__(self, input_dim: int, num_windows: int = 5, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 3) for _ in range(num_windows)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        logits = torch.stack([cls(h) for cls in self.classifiers], dim=1)
        return logits


class MarketStateEncoder(nn.Module):
    """市场状态编码器 - 提取波动率/价差/成交量等市场状态"""

    def __init__(self, input_dim: int, output_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 3, output_dim * 2),  # 修复：input_dim * 3 因为统计量拼接了3个
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = torch.cat([
            x.std(dim=1),
            x.max(dim=1).values - x.min(dim=1).values,
            x.mean(dim=1),
        ], dim=-1)
        return self.encoder(stats)


class DeepProfitNet(nn.Module):
    """深度收益导向分层集成网络"""

    def __init__(self, input_dim: int, num_windows: int = 5,
                 expert_a_dim: int = 64, expert_b_dim: int = 64,
                 expert_c_dim: int = 64, expert_d_dim: int = 32,
                 tkan_hidden: int = 128, tkan_layers: int = 2,
                 grid_size: int = 8, spline_order: int = 3,
                 num_heads: int = 4, patch_size: int = 5,
                 head_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_windows = num_windows

        self.expert_a = ExpertA_TKAN(
            input_dim, tkan_hidden, expert_a_dim, tkan_layers,
            grid_size, spline_order, dropout
        )
        self.expert_b = ExpertB_LevelAttention(
            input_dim, 64, expert_b_dim, num_heads, dropout
        )
        self.expert_c = ExpertC_PatchTransformer(
            input_dim, 64, expert_c_dim, patch_size, num_heads, 1, dropout
        )
        self.expert_d = ExpertD_CrossAssetOFI(
            input_dim, 32, expert_d_dim, dropout
        )

        total_expert_dim = expert_a_dim + expert_b_dim + expert_c_dim + expert_d_dim
        # 计算填充后的统一维度（所有专家输出中的最大值）
        max_expert_dim = max(expert_a_dim, expert_b_dim, expert_c_dim, expert_d_dim)
        market_state_dim = input_dim * 3

        self.market_encoder = MarketStateEncoder(input_dim, 32)
        # 使用 max_expert_dim 作为门控的 expert_dim
        self.gating = DynamicGatingFusion(
            4, max_expert_dim, 32, 64, dropout
        )

        self.fusion_proj = nn.Linear(total_expert_dim, total_expert_dim)
        self.fusion_norm = nn.LayerNorm(total_expert_dim)

        self.prediction_head = ProfitGuidedHead(
            total_expert_dim, num_windows, head_hidden, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_a = self.expert_a(x)
        h_b = self.expert_b(x)
        h_c = self.expert_c(x)
        h_d = self.expert_d(x)

        market_state = self.market_encoder(x)

        expert_outputs = [h_a, h_b, h_c, h_d]
        expert_dims = [h.size(-1) for h in expert_outputs]
        max_dim = max(expert_dims)
        padded = []
        for h in expert_outputs:
            if h.size(-1) < max_dim:
                pad_size = max_dim - h.size(-1)
                h = F.pad(h, (0, pad_size))
            padded.append(h)

        fused = self.gating(padded, market_state)
        fused = torch.cat(expert_outputs, dim=-1)
        fused = self.fusion_norm(self.fusion_proj(fused))

        logits = self.prediction_head(fused)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return logits.argmax(dim=-1)


def create_model(input_dim: int, num_windows: int = 5, **kwargs) -> DeepProfitNet:
    defaults = {
        'expert_a_dim': 64, 'expert_b_dim': 64,
        'expert_c_dim': 64, 'expert_d_dim': 32,
        'tkan_hidden': 128, 'tkan_layers': 2,
        'grid_size': 8, 'spline_order': 3,
        'num_heads': 4, 'patch_size': 5,
        'head_hidden': 64, 'dropout': 0.1,
    }
    defaults.update(kwargs)
    return DeepProfitNet(input_dim, num_windows, **defaults)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
