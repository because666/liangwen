"""
损失函数模块

包含：
- Huber Loss：预训练阶段使用
- Profit-Aware Loss：收益感知损失
- 组合损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class HuberLoss(nn.Module):
    """Huber Loss（平滑L1损失）"""
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch, num_windows) 预测值
            target: (batch, num_windows) 目标值
        Returns:
            loss: 标量
        """
        diff = torch.abs(pred - target)
        quadratic = torch.min(diff, torch.tensor(self.delta, device=diff.device))
        linear = diff - quadratic
        
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()


class ProfitAwareLoss(nn.Module):
    """收益感知损失函数"""
    
    def __init__(self, 
                 threshold: float = 0.5,
                 gamma: float = 0.03,
                 fee_rate: float = 0.0002,
                 use_soft_threshold: bool = False,
                 temperature: float = 0.1):
        super().__init__()
        self.threshold = threshold
        self.gamma = gamma
        self.fee_rate = fee_rate
        self.use_soft_threshold = use_soft_threshold
        self.temperature = temperature
        
    def forward(self, 
                pred_delta: torch.Tensor,
                true_delta: torch.Tensor,
                action_prob: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred_delta: (batch, num_windows) 预测的价格变化率
            true_delta: (batch, num_windows) 真实的价格变化率
            action_prob: (batch, num_windows) 出手概率
            
        Returns:
            loss: 标量损失
            info: 字典，包含统计信息
        """
        if self.use_soft_threshold:
            action = torch.sigmoid((action_prob - self.threshold) / self.temperature)
        else:
            action = (action_prob > self.threshold).float()
        
        trade_dir = torch.sign(pred_delta)
        
        actual_return = trade_dir * true_delta - self.fee_rate
        
        realized_return = action * actual_return
        
        profit_loss = -realized_return.mean()
        
        sparsity_loss = self.gamma * action.mean()
        
        loss = profit_loss + sparsity_loss
        
        with torch.no_grad():
            info = {
                'profit_loss': profit_loss.item(),
                'sparsity_loss': sparsity_loss.item(),
                'action_rate': action.mean().item(),
                'mean_return': realized_return.mean().item(),
                'positive_return_rate': (realized_return > 0).float().mean().item(),
            }
        
        return loss, info


class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self, 
                 huber_weight: float = 1.0,
                 profit_weight: float = 1.0,
                 action_weight: float = 0.1,
                 threshold: float = 0.5,
                 gamma: float = 0.03,
                 fee_rate: float = 0.0002):
        super().__init__()
        
        self.huber_weight = huber_weight
        self.profit_weight = profit_weight
        self.action_weight = action_weight
        
        self.huber_loss = HuberLoss()
        self.profit_loss = ProfitAwareLoss(threshold, gamma, fee_rate)
        
    def forward(self,
                pred_delta: torch.Tensor,
                true_delta: torch.Tensor,
                action_prob: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred_delta: (batch, num_windows) 预测的价格变化率
            true_delta: (batch, num_windows) 真实的价格变化率
            action_prob: (batch, num_windows) 出手概率
            
        Returns:
            loss: 标量损失
            info: 字典，包含统计信息
        """
        huber = self.huber_loss(pred_delta, true_delta)
        
        profit, profit_info = self.profit_loss(pred_delta, true_delta, action_prob)
        
        # 使用 binary_cross_entropy_with_logits 替代，更安全
        # 将 action_prob 转换为 logits
        action_prob_clipped = torch.clamp(action_prob, min=1e-7, max=1-1e-7)
        action_logits = torch.log(action_prob_clipped / (1 - action_prob_clipped))
        action_bce = F.binary_cross_entropy_with_logits(
            action_logits,
            (torch.abs(true_delta) > 0.0005).float(),
            reduction='mean'
        )
        
        loss = (self.huber_weight * huber + 
                self.profit_weight * profit + 
                self.action_weight * action_bce)
        
        info = {
            'huber_loss': huber.item(),
            'profit_loss': profit.item(),
            'action_bce': action_bce.item(),
            **profit_info
        }
        
        return loss, info


class DirectionalLoss(nn.Module):
    """方向感知损失函数"""
    
    def __init__(self, 
                 direction_weight: float = 1.0,
                 magnitude_weight: float = 0.5,
                 threshold: float = 0.0005):
        super().__init__()
        
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.threshold = threshold
        
    def forward(self,
                pred_delta: torch.Tensor,
                true_delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_delta: (batch, num_windows) 预测的价格变化率
            true_delta: (batch, num_windows) 真实的价格变化率
            
        Returns:
            loss: 标量损失
        """
        pred_sign = torch.sign(pred_delta)
        true_sign = torch.sign(true_delta)
        
        direction_correct = (pred_sign == true_sign).float()
        direction_loss = 1 - direction_correct.mean()
        
        magnitude_loss = F.smooth_l1_loss(pred_delta, true_delta)
        
        loss = self.direction_weight * direction_loss + self.magnitude_weight * magnitude_loss
        
        return loss


class FocalRegressionLoss(nn.Module):
    """焦点回归损失：关注难以预测的样本"""
    
    def __init__(self, alpha: float = 2.0, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self,
                pred_delta: torch.Tensor,
                true_delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_delta: (batch, num_windows) 预测的价格变化率
            true_delta: (batch, num_windows) 真实的价格变化率
            
        Returns:
            loss: 标量损失
        """
        diff = torch.abs(pred_delta - true_delta)
        
        weights = torch.pow(diff + 1e-8, self.alpha)
        
        loss = weights * F.smooth_l1_loss(pred_delta, true_delta, reduction='none')
        
        loss = loss.mean()
        
        return loss


def compute_cumulative_return(predictions: torch.Tensor,
                               true_delta: torch.Tensor,
                               fee_rate: float = 0.0002) -> float:
    """
    计算累计收益率
    
    Args:
        predictions: (batch, num_windows) 预测标签 (0=下跌, 1=不变, 2=上涨)
        true_delta: (batch, num_windows) 真实价格变化率
        fee_rate: 单边手续费率
        
    Returns:
        cumulative_return: 累计收益率
    """
    total_return = 0.0
    num_trades = 0
    
    for w in range(predictions.size(1)):
        for i in range(predictions.size(0)):
            pred = predictions[i, w].item()
            delta = true_delta[i, w].item()
            
            if pred == 2:
                total_return += delta - fee_rate * 2
                num_trades += 1
            elif pred == 0:
                total_return -= delta - fee_rate * 2
                num_trades += 1
                
    return total_return


def compute_single_return(predictions: torch.Tensor,
                          true_delta: torch.Tensor,
                          fee_rate: float = 0.0002) -> float:
    """
    计算单次收益率
    
    Args:
        predictions: (batch, num_windows) 预测标签
        true_delta: (batch, num_windows) 真实价格变化率
        fee_rate: 单边手续费率
        
    Returns:
        single_return: 单次收益率
    """
    cumulative = compute_cumulative_return(predictions, true_delta, fee_rate)
    
    num_trades = 0
    for w in range(predictions.size(1)):
        for i in range(predictions.size(0)):
            pred = predictions[i, w].item()
            if pred in [0, 2]:
                num_trades += 1
                
    if num_trades == 0:
        return 0.0
        
    return cumulative / num_trades


if __name__ == '__main__':
    batch_size = 32
    num_windows = 5
    
    pred_delta = torch.randn(batch_size, num_windows)
    true_delta = torch.randn(batch_size, num_windows)
    action_prob = torch.sigmoid(torch.randn(batch_size, num_windows))
    
    huber = HuberLoss()
    print(f"Huber Loss: {huber(pred_delta, true_delta).item():.4f}")
    
    profit_loss = ProfitAwareLoss(threshold=0.5, gamma=0.03)
    loss, info = profit_loss(pred_delta, true_delta, action_prob)
    print(f"Profit-Aware Loss: {loss.item():.4f}")
    print(f"Info: {info}")
    
    combined = CombinedLoss()
    loss, info = combined(pred_delta, true_delta, action_prob)
    print(f"Combined Loss: {loss.item():.4f}")
    print(f"Info: {info}")
