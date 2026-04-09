"""
收益导向损失函数

核心设计：
1. 收益加权交叉熵 - 让模型对高收益交易更敏感
2. 收益预测损失 - 直接优化收益预测
3. 交易惩罚 - 鼓励模型减少过度交易
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ProfitGuidedLoss(nn.Module):
    """收益导向损失函数
    
    核心思想：
    1. 分类损失：收益加权的交叉熵
    2. 收益损失：预测收益与真实收益的 MSE
    3. 交易惩罚：过度交易的惩罚项
    
    关键：避免 NaN，数值稳定
    """
    
    def __init__(self, fee: float = 0.0002, lambda_return: float = 0.5, 
                 lambda_trade: float = 0.1):
        """
        Args:
            fee: 手续费率（双边）
            lambda_return: 收益损失权重
            lambda_trade: 交易惩罚权重
        """
        super().__init__()
        self.fee = fee
        self.lambda_return = lambda_return
        self.lambda_trade = lambda_trade
    
    def forward(self, logits: torch.Tensor, return_pred: torch.Tensor,
                labels: torch.Tensor, true_returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (batch, 5, 3) 分类预测
            return_pred: (batch, 5) 收益预测
            labels: (batch, 5) 真实标签
            true_returns: (batch, 5) 真实收益
        
        Returns:
            dict with 'loss', 'ce_loss', 'return_loss', 'trade_rate'
        """
        batch_size, num_windows = labels.shape
        
        ce_loss = torch.tensor(0.0, device=logits.device)
        for w in range(num_windows):
            w_logits = logits[:, w, :]
            w_labels = labels[:, w]
            w_returns = true_returns[:, w].abs()
            
            weights = torch.clamp(w_returns / self.fee, 0.5, 3.0)
            
            hold_mask = (w_labels == 1).float()
            weights = weights * (1 - hold_mask) + 1.0 * hold_mask
            
            ce = F.cross_entropy(w_logits, w_labels, reduction='none')
            ce_loss = ce_loss + (ce * weights).mean()
        
        ce_loss = ce_loss / num_windows
        
        preds = logits.argmax(dim=-1)
        trade_mask = (preds != 1).float()
        
        direction = torch.where(
            preds == 2, 1.0,
            torch.where(preds == 0, -1.0, 0.0)
        )
        
        expected_return = direction * true_returns - self.fee
        
        return_loss = F.mse_loss(
            return_pred * trade_mask,
            expected_return * trade_mask,
            reduction='sum'
        ) / (trade_mask.sum() + 1e-8)
        
        trade_rate = trade_mask.mean()
        trade_penalty = torch.relu(trade_rate - 0.5)
        
        total_loss = ce_loss + self.lambda_return * return_loss + self.lambda_trade * trade_penalty
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss.item(),
            'return_loss': return_loss.item(),
            'trade_rate': trade_rate.item(),
            'trade_penalty': trade_penalty.item(),
        }


class CompositeProfitLoss(nn.Module):
    """复合收益损失函数
    
    Loss = λ1 * ProfitGuidedLoss + λ2 * L2正则
    """
    
    def __init__(self, fee: float = 0.0002,
                 lambda_return: float = 0.5,
                 lambda_trade: float = 0.1,
                 lambda_l2: float = 1e-4):
        super().__init__()
        self.profit_loss = ProfitGuidedLoss(fee, lambda_return, lambda_trade)
        self.lambda_l2 = lambda_l2
    
    def forward(self, logits: torch.Tensor, return_pred: torch.Tensor,
                labels: torch.Tensor, true_returns: torch.Tensor,
                model_params: Optional[list] = None) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with 'loss', 'ce_loss', 'return_loss', 'trade_rate', 'l2_loss'
        """
        loss_dict = self.profit_loss(logits, return_pred, labels, true_returns)
        
        l2_loss = torch.tensor(0.0, device=logits.device)
        if model_params is not None and self.lambda_l2 > 0:
            l2_loss = sum(p.pow(2.0).sum() for p in model_params)
            loss_dict['loss'] = loss_dict['loss'] + self.lambda_l2 * l2_loss
        
        loss_dict['l2_loss'] = l2_loss.item() if isinstance(l2_loss, torch.Tensor) else l2_loss
        
        return loss_dict


def compute_trading_metrics(logits: torch.Tensor, labels: torch.Tensor,
                            true_returns: torch.Tensor, fee: float = 0.0002) -> Dict[str, float]:
    """计算交易指标
    
    Args:
        logits: (batch, num_windows, 3)
        labels: (batch, num_windows)
        true_returns: (batch, num_windows)
        fee: 手续费率
    
    Returns:
        dict with 'cumulative_return', 'single_return', 'total_trades', 
             'trade_accuracy', 'trade_rate'
    """
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


def compute_window_metrics(logits: torch.Tensor, labels: torch.Tensor,
                           true_returns: torch.Tensor, fee: float = 0.0002,
                           window_names: list = None) -> Dict[str, Dict[str, float]]:
    """计算每个窗口的独立指标
    
    用于选择最优预测窗口
    """
    if window_names is None:
        window_names = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
    
    preds = logits.argmax(dim=-1)
    num_windows = preds.shape[1]
    
    window_metrics = {}
    
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
            avg_profit = profit / num_trades
            
            correct = ((w_preds[trade_mask] == w_labels[trade_mask]) &
                       (w_labels[trade_mask] != 1)).sum().item()
            accuracy = correct / num_trades
        else:
            profit = 0.0
            avg_profit = 0.0
            accuracy = 0.0
        
        window_name = window_names[w] if w < len(window_names) else f'window_{w}'
        window_metrics[window_name] = {
            'cumulative_return': profit,
            'single_return': avg_profit,
            'total_trades': num_trades,
            'trade_accuracy': accuracy,
            'trade_rate': num_trades / len(w_preds),
        }
    
    return window_metrics


if __name__ == '__main__':
    loss_fn = ProfitGuidedLoss()
    
    batch_size = 4
    num_windows = 5
    
    logits = torch.randn(batch_size, num_windows, 3)
    return_pred = torch.randn(batch_size, num_windows)
    labels = torch.randint(0, 3, (batch_size, num_windows))
    true_returns = torch.randn(batch_size, num_windows) * 0.01
    
    loss_dict = loss_fn(logits, return_pred, labels, true_returns)
    print(f"总损失: {loss_dict['loss'].item():.4f}")
    print(f"交叉熵损失: {loss_dict['ce_loss']:.4f}")
    print(f"收益损失: {loss_dict['return_loss']:.4f}")
    print(f"交易率: {loss_dict['trade_rate']:.3f}")
