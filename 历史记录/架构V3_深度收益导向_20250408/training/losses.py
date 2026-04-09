"""
深度收益导向损失函数

核心设计：
1. 收益加权交叉熵 - 让模型对高收益交易更敏感
2. 夏普比率损失 - 平衡收益与风险
3. 手续费敏感 - 直接扣除0.02%双边手续费
4. 稀疏正则 - 鼓励模型减少交易次数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProfitWeightedCELoss(nn.Module):
    """收益加权交叉熵损失

    对预测正确的高收益交易给予更大权重，
    对预测"不变"的交易给予更高权重，鼓励模型学会观望。
    """

    def __init__(self, fee: float = 0.0002, hold_weight: float = 1.5,
                 min_trade_weight: float = 0.5, max_trade_weight: float = 2.0):
        super().__init__()
        self.fee = fee
        self.hold_weight = hold_weight  # 提高 hold_weight，鼓励观望
        self.min_trade_weight = min_trade_weight
        self.max_trade_weight = max_trade_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                true_returns: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_windows, 3) 模型输出
            labels: (batch, num_windows) 真实标签 0/1/2
            true_returns: (batch, num_windows) 真实收益率（可选）
        """
        batch_size, num_windows, _ = logits.shape

        total_loss = torch.tensor(0.0, device=logits.device)
        for w in range(num_windows):
            w_logits = logits[:, w, :]
            w_labels = labels[:, w]

            ce_loss = F.cross_entropy(w_logits, w_labels, reduction='none')

            if true_returns is not None:
                w_returns = true_returns[:, w].abs()
                trade_mask = (w_labels != 1).float()
                hold_mask = (w_labels == 1).float()

                # 限制权重范围，防止数值过大
                normalized_returns = torch.sqrt(torch.clamp(w_returns / self.fee, 0.0, 100.0))
                trade_weights = torch.clamp(
                    normalized_returns,
                    min=self.min_trade_weight,
                    max=self.max_trade_weight
                )

                sample_weights = (
                    trade_mask * trade_weights +
                    hold_mask * self.hold_weight
                )
            else:
                sample_weights = torch.ones(batch_size, device=logits.device)

            total_loss = total_loss + (ce_loss * sample_weights).mean()

        return total_loss / num_windows


class SharpeRatioLoss(nn.Module):
    """夏普比率损失 - 最大化风险调整后收益

    修复：
    1. 当 std_profit 过小时返回 0，避免除零
    2. 添加 hold_bonus 鼓励观望
    """

    def __init__(self, fee: float = 0.0002, eps: float = 1e-6, 
                 min_std: float = 1e-4, hold_bonus_threshold: float = 0.001):
        super().__init__()
        self.fee = fee
        self.eps = eps
        self.min_std = min_std
        self.hold_bonus_threshold = hold_bonus_threshold

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                true_returns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_windows, 3)
            labels: (batch, num_windows)
            true_returns: (batch, num_windows) 真实收益率
        """
        probs = F.softmax(logits, dim=-1)
        batch_size, num_windows, _ = probs.shape

        all_profits = []
        for w in range(num_windows):
            w_probs = probs[:, w, :]
            w_returns = true_returns[:, w]

            down_prob = w_probs[:, 0]
            up_prob = w_probs[:, 2]
            hold_prob = w_probs[:, 1]

            expected_direction = up_prob - down_prob
            trade_prob = down_prob + up_prob

            # 期望收益（带手续费）
            expected_profit = trade_prob * (expected_direction * w_returns - self.fee)

            # 添加观望奖励：当波动很小时，观望是正确的选择
            normalized_volatility = torch.clamp(w_returns.abs() / self.hold_bonus_threshold, 0.0, 1.0)
            hold_bonus = hold_prob * (1.0 - normalized_volatility) * 0.1

            all_profits.append(expected_profit + hold_bonus)

        profits = torch.stack(all_profits, dim=-1)
        mean_profit = profits.mean()
        std_profit = profits.std(unbiased=False)

        # 关键修复：当标准差过小时返回 0，避免除零
        if std_profit < self.min_std:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        sharpe = mean_profit / (std_profit + self.eps)
        sharpe = torch.clamp(sharpe, -10.0, 10.0)
        return -sharpe


class CompositeLoss(nn.Module):
    """复合损失函数

    Loss = λ1 * ProfitWeightedCE + λ2 * (-SharpeRatio) + λ3 * L2Reg + λ4 * SparseReg
    """

    def __init__(self, fee: float = 0.0002,
                 lambda_ce: float = 1.0,
                 lambda_sharpe: float = 0.05,
                 lambda_l2: float = 1e-5,
                 lambda_sparse: float = 0.01,
                 hold_weight: float = 1.5):
        super().__init__()
        self.profit_ce = ProfitWeightedCELoss(fee, hold_weight)
        self.sharpe_loss = SharpeRatioLoss(fee)
        self.lambda_ce = lambda_ce
        self.lambda_sharpe = lambda_sharpe
        self.lambda_l2 = lambda_l2
        self.lambda_sparse = lambda_sparse

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                true_returns: torch.Tensor = None,
                model_params: list = None) -> dict:
        """
        Returns:
            dict with 'loss', 'ce_loss', 'sharpe_loss', 'l2_loss', 'sparse_loss'
        """
        ce_loss = self.profit_ce(logits, labels, true_returns)

        sharpe_loss = torch.tensor(0.0, device=logits.device)
        if true_returns is not None and self.lambda_sharpe > 0:
            sharpe_loss = self.sharpe_loss(logits, labels, true_returns)

        l2_loss = torch.tensor(0.0, device=logits.device)
        if model_params is not None and self.lambda_l2 > 0:
            l2_loss = sum(p.pow(2.0).sum() for p in model_params)

        # 稀疏正则：惩罚交易率，鼓励观望
        sparse_loss = torch.tensor(0.0, device=logits.device)
        if self.lambda_sparse > 0:
            probs = F.softmax(logits, dim=-1)
            trade_rate = (probs[:, :, 0] + probs[:, :, 2]).mean()
            sparse_loss = trade_rate

        total_loss = (
            self.lambda_ce * ce_loss +
            self.lambda_sharpe * sharpe_loss +
            self.lambda_l2 * l2_loss +
            self.lambda_sparse * sparse_loss
        )

        return {
            'loss': total_loss,
            'ce_loss': ce_loss.item(),
            'sharpe_loss': sharpe_loss.item() if isinstance(sharpe_loss, torch.Tensor) else sharpe_loss,
            'l2_loss': l2_loss.item() if isinstance(l2_loss, torch.Tensor) else l2_loss,
            'sparse_loss': sparse_loss.item() if isinstance(sparse_loss, torch.Tensor) else sparse_loss,
        }


def compute_trading_metrics(logits: torch.Tensor, labels: torch.Tensor,
                            true_returns: torch.Tensor, fee: float = 0.0002) -> dict:
    """计算交易指标"""
    preds = logits.argmax(dim=-1)
    batch_size, num_windows = preds.shape

    total_profit = 0.0
    total_trades = 0
    correct_trades = 0

    for w in range(num_windows):
        w_preds = preds[:, w]
        w_labels = labels[:, w]
        w_returns = true_returns[:, w]

        trade_mask = w_preds != 1
        num_trades = trade_mask.sum().item()

        if num_trades > 0:
            direction = torch.where(
                w_preds == 2, 1.0,
                torch.where(w_preds == 0, -1.0, 0.0)
            )
            profit = (direction[trade_mask] * w_returns[trade_mask] - fee).sum().item()
            total_profit += profit
            total_trades += num_trades

            correct = ((w_preds[trade_mask] == w_labels[trade_mask]) &
                       (w_labels[trade_mask] != 1)).sum().item()
            correct_trades += correct

    avg_profit = total_profit / total_trades if total_trades > 0 else 0.0
    accuracy = correct_trades / total_trades if total_trades > 0 else 0.0
    trade_rate = total_trades / (batch_size * num_windows)

    return {
        'cumulative_return': total_profit,
        'single_return': avg_profit,
        'total_trades': total_trades,
        'trade_accuracy': accuracy,
        'trade_rate': trade_rate,
    }
