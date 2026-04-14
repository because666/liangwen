"""
T-KAN + LightGBM 优化方案本地测试脚本

测试所有优化方案：
1. 动态阈值 + 后处理过滤
2. 回归预测
3. 两阶段模型
4. 代价敏感学习

使用方法：
python test_optimized.py --mode [dynamic|regression|two_stage|cost_sensitive|all]
"""

import os
import sys
import argparse
import json
import zipfile
import numpy as np
import pandas as pd
from typing import List

FEATURE_COLS = [
    'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'bid6', 'bid7', 'bid8', 'bid9', 'bid10',
    'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask6', 'ask7', 'ask8', 'ask9', 'ask10',
    'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'bsize6', 'bsize7', 'bsize8', 'bsize9', 'bsize10',
    'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'asize6', 'asize7', 'asize8', 'asize9', 'asize10',
]

WINDOW_SIZES = [5, 10, 20, 40, 60]
LABEL_COLS = [f'label_{w}' for w in WINDOW_SIZES]
FEE = 0.0001


def load_test_data(data_path: str, max_samples: int = None):
    """加载测试数据"""
    print(f"加载测试数据: {data_path}")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"数据形状: {df.shape}")
    print(f"数据列: {list(df.columns)[:10]}...")
    
    return df


def prepare_batch_data(df, batch_size: int = 4):
    """准备批量测试数据
    
    返回 List[DataFrame]，每个 DataFrame 为 100 个 tick 的数据
    """
    max_window = max(WINDOW_SIZES)
    
    if len(df) < 100 + max_window:
        print(f"警告: 数据长度 {len(df)} 不足，需要至少 {100 + max_window} 行")
        return [], []
    
    all_batches = []
    all_labels = []
    
    for t in range(100, len(df) - max_window, 50):
        batch_df = df.iloc[t - 100:t].copy()
        all_batches.append(batch_df)
        
        labels = [df.iloc[t + w]['label_5'] if t + w < len(df) else 1 for w in range(5)]
        all_labels.append(labels)
        
        if len(all_batches) >= batch_size:
            break
    
    return all_batches, all_labels


def compute_metrics(preds: List[List[int]], labels: List[List[int]], 
                    returns: np.ndarray = None):
    """计算评估指标"""
    preds = np.array(preds)
    labels = np.array(labels)
    
    results = {}
    
    for w_idx, w in enumerate(WINDOW_SIZES):
        pred_w = preds[:, w_idx]
        label_w = labels[:, w_idx]
        
        accuracy = (pred_w == label_w).mean()
        
        trade_mask = pred_w != 1
        num_trades = trade_mask.sum()
        trade_rate = num_trades / len(pred_w)
        
        if returns is not None and len(returns) > w_idx:
            ret = returns[:, w_idx]
            direction = np.where(pred_w == 2, 1.0, np.where(pred_w == 0, -1.0, 0.0))
            profit = (direction[trade_mask] * ret[trade_mask] - FEE).sum()
            avg_profit = profit / max(num_trades, 1)
        else:
            profit = 0.0
            avg_profit = 0.0
        
        results[f'label_{w}'] = {
            'accuracy': float(accuracy),
            'trade_count': int(num_trades),
            'trade_rate': float(trade_rate),
            'cumulative_return': float(profit),
            'single_return': float(avg_profit),
        }
    
    return results


def test_dynamic_threshold_mode(test_data_path: str):
    """测试动态阈值 + 后处理过滤模式"""
    print("\n" + "=" * 80)
    print("测试优化方案: 动态阈值 + 后处理过滤")
    print("=" * 80)
    
    from Predictor_optimized import Predictor
    
    predictor = Predictor(
        use_dynamic_threshold=True,
        use_post_filter=True,
        use_regression=False,
        use_two_stage=False
    )
    
    df = load_test_data(test_data_path)
    batches, labels = prepare_batch_data(df)
    
    if not batches:
        print("错误: 没有有效的测试数据")
        return
    
    print(f"测试样本数: {len(batches)}")
    
    predictions = predictor.predict(batches)
    
    results = compute_metrics(predictions, labels)
    
    print("\n--- 测试结果 ---")
    for window, metrics in results.items():
        print(f"\n{window}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  交易次数: {metrics['trade_count']}")
        print(f"  交易率: {metrics['trade_rate']:.4f}")
        print(f"  累计收益: {metrics['cumulative_return']:.6f}")
        print(f"  单次收益: {metrics['single_return']:.6f}")
    
    return results


def test_regression_mode(test_data_path: str):
    """测试回归预测模式"""
    print("\n" + "=" * 80)
    print("测试优化方案: 回归预测")
    print("=" * 80)
    
    from Predictor_optimized import Predictor
    
    predictor = Predictor(
        use_dynamic_threshold=False,
        use_post_filter=False,
        use_regression=True,
        use_two_stage=False
    )
    
    df = load_test_data(test_data_path)
    batches, labels = prepare_batch_data(df)
    
    if not batches:
        print("错误: 没有有效的测试数据")
        return
    
    print(f"测试样本数: {len(batches)}")
    
    predictions = predictor.predict(batches)
    
    results = compute_metrics(predictions, labels)
    
    print("\n--- 测试结果 ---")
    for window, metrics in results.items():
        print(f"\n{window}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  交易次数: {metrics['trade_count']}")
        print(f"  交易率: {metrics['trade_rate']:.4f}")
        print(f"  累计收益: {metrics['cumulative_return']:.6f}")
        print(f"  单次收益: {metrics['single_return']:.6f}")
    
    return results


def test_two_stage_mode(test_data_path: str):
    """测试两阶段模型模式"""
    print("\n" + "=" * 80)
    print("测试优化方案: 两阶段模型")
    print("=" * 80)
    
    from Predictor_two_stage import Predictor
    
    predictor = Predictor()
    
    df = load_test_data(test_data_path)
    batches, labels = prepare_batch_data(df)
    
    if not batches:
        print("错误: 没有有效的测试数据")
        return
    
    print(f"测试样本数: {len(batches)}")
    
    predictions = predictor.predict(batches)
    
    results = compute_metrics(predictions, labels)
    
    print("\n--- 测试结果 ---")
    for window, metrics in results.items():
        print(f"\n{window}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  交易次数: {metrics['trade_count']}")
        print(f"  交易率: {metrics['trade_rate']:.4f}")
        print(f"  累计收益: {metrics['cumulative_return']:.6f}")
        print(f"  单次收益: {metrics['single_return']:.6f}")
    
    return results


def test_cost_sensitive_mode(test_data_path: str):
    """测试代价敏感学习模式"""
    print("\n" + "=" * 80)
    print("测试优化方案: 代价敏感学习")
    print("=" * 80)
    
    from Predictor_cost_sensitive import Predictor
    
    predictor = Predictor()
    
    df = load_test_data(test_data_path)
    batches, labels = prepare_batch_data(df)
    
    if not batches:
        print("错误: 没有有效的测试数据")
        return
    
    print(f"测试样本数: {len(batches)}")
    
    predictions = predictor.predict(batches)
    
    results = compute_metrics(predictions, labels)
    
    print("\n--- 测试结果 ---")
    for window, metrics in results.items():
        print(f"\n{window}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  交易次数: {metrics['trade_count']}")
        print(f"  交易率: {metrics['trade_rate']:.4f}")
        print(f"  累计收益: {metrics['cumulative_return']:.6f}")
        print(f"  单次收益: {metrics['single_return']:.6f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='T-KAN + LightGBM 优化方案本地测试')
    parser.add_argument('--mode', type=str, default='dynamic',
                       choices=['dynamic', 'regression', 'two_stage', 'cost_sensitive', 'all'],
                       help='测试模式')
    parser.add_argument('--data', type=str, 
                       default='../../example_本地测试代码/example/snapshot_sym0_date0_am.parquet',
                       help='测试数据路径')
    
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, args.data)
    
    if not os.path.exists(data_path):
        print(f"错误: 测试数据不存在: {data_path}")
        return
    
    if args.mode == 'dynamic' or args.mode == 'all':
        try:
            test_dynamic_threshold_mode(data_path)
        except Exception as e:
            print(f"动态阈值模式测试失败: {e}")
    
    if args.mode == 'regression' or args.mode == 'all':
        try:
            test_regression_mode(data_path)
        except Exception as e:
            print(f"回归预测模式测试失败: {e}")
    
    if args.mode == 'two_stage' or args.mode == 'all':
        try:
            test_two_stage_mode(data_path)
        except Exception as e:
            print(f"两阶段模型测试失败: {e}")
    
    if args.mode == 'cost_sensitive' or args.mode == 'all':
        try:
            test_cost_sensitive_mode(data_path)
        except Exception as e:
            print(f"代价敏感学习测试失败: {e}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
