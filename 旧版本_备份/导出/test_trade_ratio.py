"""
测试交易比例 - 验证优化版预测器的交易频率
"""
import os
import sys
import numpy as np
import pandas as pd

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Predictor_optimized import Predictor

def test_trade_ratio():
    """测试交易比例"""
    predictor = Predictor()
    
    # 生成更多测试数据来统计交易比例
    np.random.seed(42)
    test_data = [pd.DataFrame(np.random.randn(100, 35)) for _ in range(100)]
    
    results = predictor.predict(test_data)
    
    # 统计每个时间窗口的交易比例
    labels = ["label_5", "label_10", "label_20", "label_40", "label_60"]
    directions = ["下跌", "不变", "上涨"]
    
    print("\n" + "="*60)
    print("交易比例统计")
    print("="*60)
    
    for label_idx, label_name in enumerate(labels):
        up_count = sum(1 for r in results if r[label_idx] == 2)
        down_count = sum(1 for r in results if r[label_idx] == 0)
        neutral_count = sum(1 for r in results if r[label_idx] == 1)
        
        total = len(results)
        trade_count = up_count + down_count
        trade_ratio = trade_count / total
        
        print(f"\n{label_name}:")
        print(f"  上涨: {up_count} ({up_count/total*100:.1f}%)")
        print(f"  不变: {neutral_count} ({neutral_count/total*100:.1f}%)")
        print(f"  下跌: {down_count} ({down_count/total*100:.1f}%)")
        print(f"  交易比例: {trade_ratio*100:.1f}% (目标: ≥40%)")
    
    # 统计整体交易比例
    all_preds = [pred for result in results for pred in result]
    up_total = all_preds.count(2)
    down_total = all_preds.count(0)
    neutral_total = all_preds.count(1)
    total_preds = len(all_preds)
    
    print("\n" + "="*60)
    print("总体统计")
    print("="*60)
    print(f"总预测数: {total_preds}")
    print(f"上涨: {up_total} ({up_total/total_preds*100:.1f}%)")
    print(f"不变: {neutral_total} ({neutral_total/total_preds*100:.1f}%)")
    print(f"下跌: {down_total} ({down_total/total_preds*100:.1f}%)")
    print(f"总交易比例: {(up_total+down_total)/total_preds*100:.1f}%")
    
    # 验证最小交易比例
    min_ratio_met = all(
        (sum(1 for r in results if r[i] != 1) / len(results)) >= predictor.min_trade_ratio
        for i in range(5)
    )
    print(f"\n最小交易比例要求(40%): {'✓ 满足' if min_ratio_met else '✗ 不满足'}")

if __name__ == "__main__":
    test_trade_ratio()
