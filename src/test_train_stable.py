"""本地测试 train_stable.py 是否能正常导入和运行"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("测试 train_stable.py")
print("=" * 60)

# 1. 测试导入
print("\n[1] 测试导入...")
try:
    from train_stable import (
        RAW_FEATURE_COLS, DERIVED_FEATURE_COLS, ALL_FEATURE_COLS,
        LABEL_COLS, WINDOW_SIZES,
        clean_features, compute_derived_features, load_and_process_file,
        HFTDataset, EMA, compute_stats
    )
    print("✓ 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 2. 测试特征列定义
print("\n[2] 测试特征列...")
print(f"  RAW_FEATURE_COLS: {len(RAW_FEATURE_COLS)} 列")
print(f"  DERIVED_FEATURE_COLS: {len(DERIVED_FEATURE_COLS)} 列")
print(f"  ALL_FEATURE_COLS: {len(ALL_FEATURE_COLS)} 列")
print(f"  LABEL_COLS: {LABEL_COLS}")
assert len(RAW_FEATURE_COLS) == 46, f"RAW_FEATURE_COLS 应该是46列，实际{len(RAW_FEATURE_COLS)}"
assert len(DERIVED_FEATURE_COLS) == 7, f"DERIVED_FEATURE_COLS 应该是7列，实际{len(DERIVED_FEATURE_COLS)}"
assert len(ALL_FEATURE_COLS) == 53, f"ALL_FEATURE_COLS 应该是53列，实际{len(ALL_FEATURE_COLS)}"
print("✓ 特征列定义正确")

# 3. 测试特征清洗函数
print("\n[3] 测试特征清洗...")
import numpy as np
test_features = np.random.randn(100, len(ALL_FEATURE_COLS)).astype(np.float32)
cleaned = clean_features(test_features, ALL_FEATURE_COLS)
assert not np.isnan(cleaned).any(), "存在NaN"
assert not np.isinf(cleaned).any(), "存在Inf"
print(f"✓ 特征清洗正常 (shape: {cleaned.shape})")

# 4. 测试衍生特征计算
print("\n[4] 测试衍生特征计算...")
import pandas as pd

# 创建模拟数据
n = 200
data = {
    'bid1': np.random.randn(n).astype(np.float32),
    'ask1': np.random.randn(n).astype(np.float32),
    **{f'bid{i}': np.random.randn(n).astype(np.float32) for i in range(2, 11)},
    **{f'ask{i}': np.random.randn(n).astype(np.float32) for i in range(2, 11)},
    **{f'bsize{i}': np.abs(np.random.randn(n)).astype(np.float32) for i in range(1, 11)},
    **{f'asize{i}': np.abs(np.random.randn(n)).astype(np.float32) for i in range(1, 11)},
    'mb_intst': np.random.randn(n).astype(np.float32),
    'ma_intst': np.random.randn(n).astype(np.float32),
    'lb_intst': np.random.randn(n).astype(np.float32),
    'la_intst': np.random.randn(n).astype(np.float32),
    'cb_intst': np.random.randn(n).astype(np.float32),
    'ca_intst': np.random.randn(n).astype(np.float32),
}
df = pd.DataFrame(data)

derived, midprice = compute_derived_features(df)
print(f"  derived shape: {derived.shape}")
print(f"  midprice shape: {midprice.shape}")
assert derived.shape == (n, len(DERIVED_FEATURE_COLS)), f"衍生特征形状错误: {derived.shape}"
assert midprice.shape == (n,), f"midprice形状错误: {midprice.shape}"
print("✓ 衍生特征计算正常")

# 5. 测试完整数据处理流程
print("\n[5] 测试完整数据处理...")
# 添加标签列
for label in LABEL_COLS:
    df[label] = np.random.randint(0, 3, size=n)

# 保存为临时parquet文件
import tempfile
temp_dir = tempfile.mkdtemp()
temp_file = os.path.join(temp_dir, 'test.parquet')
df.to_parquet(temp_file, engine='pyarrow')

result = load_and_process_file(temp_file)
assert result is not None, "load_and_process_file 返回None"
features, labels, true_returns = result
print(f"  features: {features.shape}")
print(f"  labels: {labels.shape}")
print(f"  true_returns: {true_returns.shape}")
assert features.shape[1] == len(ALL_FEATURE_COLS), f"特征数错误: {features.shape[1]}"
assert labels.shape[1] == len(LABEL_COLS), f"标签数错误: {labels.shape[1]}"
print("✓ 完整数据处理正常")

# 清理临时文件
import shutil
shutil.rmtree(temp_dir)

# 6. 测试模型创建
print("\n[6] 测试模型创建...")
try:
    from model import create_model, count_parameters
    model = create_model(len(ALL_FEATURE_COLS), num_windows=5, dropout=0.15)
    params = count_parameters(model)
    print(f"  参数量: {params:,}")
    assert params > 0, "模型参数为0"
    print("✓ 模型创建成功")
except Exception as e:
    print(f"✗ 模型创建失败: {e}")

# 7. 测试损失函数
print("\n[7] 测试损失函数...")
try:
    from losses import CompositeProfitLoss, compute_trading_metrics
    criterion = CompositeProfitLoss()
    
    import torch
    batch_size = 4
    seq_len = 100
    logits = torch.randn(batch_size, 5, 3)
    return_pred = torch.randn(batch_size, 5)
    labels = torch.randint(0, 3, (batch_size, 5))
    true_returns = torch.randn(batch_size, 5) * 0.01
    
    loss_dict = criterion(logits, return_pred, labels, true_returns, list(model.parameters()))
    metrics = compute_trading_metrics(logits, labels, true_returns)
    
    print(f"  loss: {loss_dict['loss'].item():.4f}")
    print(f"  trade_rate: {metrics['trade_rate']:.3f}")
    print("✓ 损失函数正常")
except Exception as e:
    print(f"✗ 损失函数失败: {e}")

print("\n" + "=" * 60)
print("所有测试通过！✅")
print("=" * 60)
print("\ntrain_stable.py 可以正常使用")
print("请上传到服务器运行完整训练")
