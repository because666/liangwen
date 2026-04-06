"""
收益导向模型本地测试脚本

功能：
1. 加载训练好的收益导向模型
2. 在测试数据上运行预测
3. 计算详细的交易统计（重点关注收益而非准确率）
4. 验证是否解决了"准确率高但收益低"的问题
"""

from __future__ import annotations

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Predictor_profit import Predictor, compute_ofi_features


FEE_RATE = 0.0002
LABEL_COLS = ["label_5", "label_10", "label_20", "label_40", "label_60"]


def load_test_data(data_dir: str, max_files: int = 50) -> Tuple[List[pd.DataFrame], List[np.ndarray]]:
    """加载测试数据"""
    print(f"\n加载测试数据从: {data_dir}")
    
    files = sorted(glob.glob(os.path.join(data_dir, "snapshot_sym*.parquet")))[:max_files]
    print(f"找到 {len(files)} 个文件")
    
    test_data = []
    test_labels = []
    
    for f in files:
        try:
            df = pd.read_parquet(f)
            df = df.fillna(0).replace([np.inf, -np.inf], 0)
            
            if len(df) < 100:
                continue
            
            for i in range(0, len(df) - 99, 10):
                sample_df = df.iloc[i:i+100].copy()
                label = df.iloc[i+99][LABEL_COLS].values.astype(int)
                
                test_data.append(sample_df)
                test_labels.append(label)
                
        except Exception as e:
            continue
    
    print(f"加载了 {len(test_data)} 个测试样本")
    return test_data, np.array(test_labels)


def evaluate_predictions(predictions: List[List[int]], labels: np.ndarray) -> Dict:
    """
    评估预测结果 - 收益导向评估
    
    重点指标：
    1. 累计收益率（扣除手续费）
    2. 单次平均收益
    3. 胜率
    4. 出手次数和出手率
    """
    predictions = np.array(predictions)
    
    results = {
        'window_stats': [],
        'total_profit': 0,
        'total_trades': 0,
        'total_wins': 0
    }
    
    print(f"\n{'='*80}")
    print("收益导向评估结果")
    print(f"{'='*80}")
    print(f"\n{'窗口':<12} {'准确率':>8} {'出手率':>8} {'出手数':>8} {'胜率':>8} {'累计收益':>12} {'单次收益':>12}")
    print(f"{'-'*80}")
    
    for i, label_name in enumerate(LABEL_COLS):
        preds = predictions[:, i]
        true_labels = labels[:, i]
        
        accuracy = (preds == true_labels).mean()
        
        trade_mask = preds != 1
        trade_preds = preds[trade_mask]
        trade_labels = true_labels[trade_mask]
        
        trade_count = len(trade_preds)
        trade_rate = trade_count / len(preds)
        
        if trade_count > 0:
            trade_accuracy = (trade_preds == trade_labels).mean()
            
            profits = []
            for j in range(len(trade_preds)):
                if trade_preds[j] == 2 and trade_labels[j] == 2:
                    profits.append(0.001 - FEE_RATE)
                elif trade_preds[j] == 0 and trade_labels[j] == 0:
                    profits.append(0.001 - FEE_RATE)
                elif trade_preds[j] == 2 and trade_labels[j] == 0:
                    profits.append(-0.002 - FEE_RATE)
                elif trade_preds[j] == 0 and trade_labels[j] == 2:
                    profits.append(-0.002 - FEE_RATE)
                else:
                    profits.append(-FEE_RATE)
            
            total_profit = sum(profits)
            avg_profit = total_profit / trade_count
            win_rate = sum(1 for p in profits if p > 0) / len(profits)
            
            results['total_profit'] += total_profit
            results['total_trades'] += trade_count
            results['total_wins'] += sum(1 for p in profits if p > 0)
        else:
            trade_accuracy = 0
            total_profit = 0
            avg_profit = 0
            win_rate = 0
        
        window_stat = {
            'window': label_name,
            'accuracy': accuracy,
            'trade_rate': trade_rate,
            'trade_count': trade_count,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit
        }
        results['window_stats'].append(window_stat)
        
        print(f"{label_name:<12} {accuracy:>8.4f} {trade_rate:>8.4f} {trade_count:>8} {win_rate:>8.4f} {total_profit:>12.6f} {avg_profit:>12.6f}")
    
    print(f"{'-'*80}")
    
    if results['total_trades'] > 0:
        overall_avg_profit = results['total_profit'] / results['total_trades']
        overall_win_rate = results['total_wins'] / results['total_trades']
        
        results['overall_avg_profit'] = overall_avg_profit
        results['overall_win_rate'] = overall_win_rate
        
        print(f"\n★ 总体统计:")
        print(f"  总出手次数: {results['total_trades']}")
        print(f"  总胜率: {overall_win_rate:.4f}")
        print(f"  累计收益: {results['total_profit']:.6f}")
        print(f"  单次平均收益: {overall_avg_profit:.6f}")
        
        if overall_avg_profit > 0:
            print(f"\n✓ 模型具有正期望收益！适合提交")
        else:
            print(f"\n✗ 模型仍为负收益，需要继续优化")
    else:
        results['overall_avg_profit'] = 0
        results['overall_win_rate'] = 0
        print(f"\n⚠ 模型没有进行任何交易！阈值可能过高")
    
    print(f"{'='*80}\n")
    
    return results


def check_prediction_distribution(predictions: List[List[int]]) -> None:
    """检查预测分布，验证是否解决了"高准确率低收益"问题"""
    predictions = np.array(predictions)
    
    print(f"\n{'='*60}")
    print("预测分布分析")
    print(f"{'='*60}")
    
    labels = ["label_5", "label_10", "label_20", "label_40", "label_60"]
    direction_names = ["下跌", "不变", "上涨"]
    
    print(f"\n{'窗口':<12} {'下跌%':>8} {'不变%':>8} {'上涨%':>8} {'问题检测':>15}")
    print(f"{'-'*60}")
    
    for i, label in enumerate(labels):
        preds = predictions[:, i]
        total = len(preds)
        
        down_pct = (preds == 0).sum() / total * 100
        hold_pct = (preds == 1).sum() / total * 100
        up_pct = (preds == 2).sum() / total * 100
        
        if hold_pct > 70:
            warning = "⚠ 过多'不变'预测"
        elif hold_pct < 30:
            warning = "⚠ 出手过于激进"
        else:
            warning = "✓ 分布合理"
        
        print(f"{label:<12} {down_pct:>8.2f}% {hold_pct:>8.2f}% {up_pct:>8.2f}% {warning:>15}")
    
    avg_hold = np.mean([(predictions[:, i] == 1).mean() for i in range(5)])
    print(f"\n平均'不变'比例: {avg_hold:.2%}")
    
    if avg_hold > 0.7:
        print("\n❌ 检测到问题：模型倾向于预测'不变'")
        print("   这会导致：准确率高但收益极低")
        print("   建议：降低阈值或调整类别权重")
    elif avg_hold < 0.3:
        print("\n⚠ 警告：模型出手过于激进")
        print("   这可能导致：手续费侵蚀利润")
        print("   建议：适当提高阈值")
    else:
        print("\n✓ 预测分布健康，平衡了准确率和收益")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="收益导向模型本地测试")
    parser.add_argument('--data_dir', type=str, default='../2026train_set/2026train_set')
    parser.add_argument('--max_samples', type=int, default=500)
    args = parser.parse_args()
    
    print("=" * 70)
    print("收益导向模型 - 本地测试脚本")
    print("=" * 70)
    print("\n目的：验证是否解决了'准确率高但收益低'的问题")
    print("=" * 70)
    
    try:
        predictor = Predictor()
    except Exception as e:
        print(f"\n✗ 无法加载模型: {e}")
        print("\n请确保已经完成训练并生成best_model.pt")
        return
    
    test_data, test_labels = load_test_data(args.data_dir, max_files=args.max_samples // 4)
    
    if len(test_data) == 0:
        print("\n✗ 没有找到测试数据")
        return
    
    print(f"\n开始预测...")
    
    batch_size = 256
    all_predictions = []
    
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        preds = predictor.predict(batch)
        all_predictions.extend(preds)
        
        if (i // batch_size + 1) % 5 == 0:
            print(f"  已处理 {min(i+batch_size, len(test_data))}/{len(test_data)} 样本")
    
    print(f"\n预测完成，共 {len(all_predictions)} 个样本")
    
    check_prediction_distribution(all_predictions)
    
    results = evaluate_predictions(all_predictions, test_labels[:len(all_predictions)])
    
    output_file = os.path.join(current_dir, "test_results_profit.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'overall_avg_profit': results.get('overall_avg_profit', 0),
            'overall_win_rate': results.get('overall_win_rate', 0),
            'total_trades': results['total_trades'],
            'window_stats': [{
                'window': ws['window'],
                'accuracy': ws['accuracy'],
                'trade_rate': ws['trade_rate'],
                'avg_profit': ws['avg_profit']
            } for ws in results['window_stats']]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"详细结果已保存到: {output_file}")


if __name__ == "__main__":
    import argparse
    main()
