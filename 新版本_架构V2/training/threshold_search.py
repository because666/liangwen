"""
阈值搜索脚本

在验证集上搜索每个窗口的最优出手阈值
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import TKANOFIModel, create_model
from data_processing.data_loader import HFTDataset
from data_processing.ofi_features import compute_ofi_features, get_ofi_column_names


def evaluate_with_threshold(model, dataloader, device, thresholds, windows=[5, 10, 20, 40, 60]):
    """使用指定阈值评估模型"""
    model.eval()
    
    all_predictions = []
    all_deltas = []
    
    with torch.no_grad():
        for features, labels, deltas in dataloader:
            features = features.to(device)
            
            pred_delta, action_prob = model(features)
            
            batch_size = features.size(0)
            predictions = torch.ones(batch_size, len(windows), dtype=torch.long, device=device)
            
            for w in range(len(windows)):
                should_act = action_prob[:, w] > thresholds[w]
                predictions[should_act, w] = torch.where(
                    pred_delta[should_act, w] > 0,
                    torch.tensor(2, device=device),
                    torch.tensor(0, device=device)
                )
            
            all_predictions.append(predictions.cpu())
            all_deltas.append(deltas)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_deltas = torch.cat(all_deltas, dim=0)
    
    cumulative_return = 0.0
    num_trades = 0
    
    for w in range(len(windows)):
        for i in range(all_predictions.size(0)):
            pred = all_predictions[i, w].item()
            delta = all_deltas[i, w].item()
            
            if pred == 2:
                cumulative_return += delta - 0.0002 * 2
                num_trades += 1
            elif pred == 0:
                cumulative_return -= delta - 0.0002 * 2
                num_trades += 1
    
    single_return = cumulative_return / num_trades if num_trades > 0 else 0.0
    
    return cumulative_return, single_return, num_trades


def search_thresholds(model, dataloader, device, windows=[5, 10, 20, 40, 60], 
                     threshold_range=(0.3, 0.7, 0.05)):
    """搜索最优阈值"""
    thresholds = [0.5] * len(windows)
    
    print("开始阈值搜索...")
    
    for w in range(len(windows)):
        best_thr = 0.5
        best_return = float('-inf')
        
        print(f"\n搜索窗口 {w} ({windows[w]} tick)...")
        
        for thr in np.arange(threshold_range[0], threshold_range[1], threshold_range[2]):
            test_thresholds = thresholds.copy()
            test_thresholds[w] = thr
            
            cumulative, single, num_trades = evaluate_with_threshold(
                model, dataloader, device, test_thresholds, windows
            )
            
            if cumulative > best_return:
                best_return = cumulative
                best_thr = thr
        
        thresholds[w] = best_thr
        print(f"窗口 {w} 最优阈值: {best_thr:.2f}, 累计收益: {best_return:.6f}")
    
    return thresholds


def main():
    parser = argparse.ArgumentParser(description='阈值搜索')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_dir', type=str, default=r'd:\量化\良文杯\2026train_set\2026train_set',
                        help='数据目录')
    parser.add_argument('--output_path', type=str, default='best_thresholds.json',
                        help='输出路径')
    parser.add_argument('--batch_size', type=int, default=256, help='批大小')
    parser.add_argument('--val_samples', type=int, default=5000, help='验证样本数')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', {})
    mean = checkpoint.get('mean', None)
    std = checkpoint.get('std', None)
    feature_cols = checkpoint.get('feature_cols', [])
    
    if not feature_cols:
        feature_cols = [
            'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'bid6', 'bid7', 'bid8', 'bid9', 'bid10',
            'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask6', 'ask7', 'ask8', 'ask9', 'ask10',
            'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'bsize6', 'bsize7', 'bsize8', 'bsize9', 'bsize10',
            'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'asize6', 'asize7', 'asize8', 'asize9', 'asize10',
            'midprice', 'imbalance', 'cumspread',
            'mb_intst', 'ma_intst', 'lb_intst', 'la_intst', 'cb_intst', 'ca_intst'
        ]
        ofi_cols = ['ofi_raw', 'ofi_ewm', 'ofi_multilevel', 'ofi_velocity', 'ofi_volatility']
        feature_cols = feature_cols + ofi_cols
    
    model = TKANOFIModel(
        input_dim=len(feature_cols),
        hidden_dim=config.get('hidden_dim', 256),
        output_dim=config.get('output_dim', 128),
        num_windows=5,
        num_layers=config.get('num_encoder_layers', 2),
        grid_size=config.get('grid_size', 8),
        spline_order=config.get('spline_order', 3),
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print("加载验证数据集...")
    
    label_cols = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
    
    val_dataset = HFTDataset(
        data_dir=args.data_dir,
        feature_cols=feature_cols[:43],
        label_cols=label_cols,
        seq_len=100,
        max_samples=args.val_samples,
        compute_ofi=False,
        normalize=True,
        mean=mean,
        std=std
    )
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"验证集样本数: {len(val_dataset)}")
    
    windows = [5, 10, 20, 40, 60]
    best_thresholds = search_thresholds(model, val_loader, device, windows)
    
    print("\n" + "="*50)
    print("最优阈值:")
    for i, (w, t) in enumerate(zip(windows, best_thresholds)):
        print(f"  {w} tick: {t:.2f}")
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump({'thresholds': best_thresholds}, f, indent=2, ensure_ascii=False)
    
    print(f"\n阈值已保存至: {args.output_path}")


if __name__ == '__main__':
    main()
