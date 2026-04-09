"""测试脚本 - 验证模型和损失函数"""

import torch
from model import create_model, count_parameters
from losses import ProfitGuidedLoss, compute_trading_metrics

print("=" * 50)
print("测试 T-KAN Pro 模型")
print("=" * 50)

model = create_model(input_dim=54, num_windows=5)
print(f"模型参数量: {count_parameters(model):,}")

x = torch.randn(2, 100, 54)
logits, return_pred = model(x)
print(f"输入形状: {x.shape}")
print(f"分类输出形状: {logits.shape}")
print(f"收益预测形状: {return_pred.shape}")

print("\n" + "=" * 50)
print("测试损失函数")
print("=" * 50)

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

print("\n" + "=" * 50)
print("测试交易指标计算")
print("=" * 50)

metrics = compute_trading_metrics(logits, labels, true_returns)
print(f"累计收益: {metrics['cumulative_return']:.6f}")
print(f"单次收益: {metrics['single_return']:.6f}")
print(f"交易次数: {metrics['total_trades']}")
print(f"交易准确率: {metrics['trade_accuracy']:.3f}")
print(f"交易率: {metrics['trade_rate']:.3f}")

print("\n所有测试通过！")
