"""
收益诊断脚本

目的：分析预测结果，定位亏损原因

输出统计：
1. 按置信度分组统计收益
2. 按实际波动大小分组统计收益
3. 按预测方向统计收益
4. 预测方向 vs 实际收益的详细统计
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from model import create_encoder
import lightgbm as lgb

FEATURE_COLS = [
    'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'bid6', 'bid7', 'bid8', 'bid9', 'bid10',
    'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask6', 'ask7', 'ask8', 'ask9', 'ask10',
    'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'bsize6', 'bsize7', 'bsize8', 'bsize9', 'bsize10',
    'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'asize6', 'asize7', 'asize8', 'asize9', 'asize10',
]

WINDOW_SIZES = [5, 10, 20, 40, 60]
LABEL_COLS = [f'label_{w}' for w in WINDOW_SIZES]
FEE = 0.0001


def clean_features(features: np.ndarray) -> np.ndarray:
    """特征清洗"""
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    for i, col in enumerate(FEATURE_COLS):
        if col.startswith('bid') or col.startswith('ask'):
            features[:, i] = np.clip(features[:, i], -0.3, 0.3)
        elif col.startswith('bsize') or col.startswith('asize'):
            features[:, i] = np.clip(features[:, i], 0, 100)
    return features


def load_data_with_returns(data_dir, max_files=None, sample_interval=5):
    """加载数据并计算实际收益率"""
    parquet_files = sorted([
        f for f in os.listdir(data_dir) if f.endswith('.parquet')
    ])
    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    print(f"找到 {len(parquet_files)} 个数据文件")

    all_sequences = []
    all_labels = []
    all_returns = []
    all_spreads = []
    all_volatility = []
    max_window = max(WINDOW_SIZES)

    for i, fname in enumerate(parquet_files):
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            print(f"  跳过文件 {fname}: {e}")
            continue

        if len(df) < 100 + max_window:
            continue

        feature_data = df[FEATURE_COLS].values.astype(np.float32)
        feature_data = clean_features(feature_data)

        midprice = df['midprice'].values.astype(np.float32)
        label_data = df[LABEL_COLS].values.astype(np.int32)
        
        # 计算价差（ask1 - bid1）
        if 'ask1' in df.columns and 'bid1' in df.columns:
            spread = (df['ask1'].values - df['bid1'].values).astype(np.float32)
        else:
            spread = np.zeros(len(df), dtype=np.float32)

        for t in range(100, len(df) - max_window, sample_interval):
            seq = feature_data[t - 100:t]
            lbl = label_data[t]
            ret = np.array([
                midprice[t + w] - midprice[t] for w in WINDOW_SIZES
            ], dtype=np.float32)
            
            # 计算当前价差
            current_spread = spread[t]
            
            # 计算历史波动率（过去20个tick的标准差）
            if t >= 20:
                volatility = np.std(midprice[t-20:t])
            else:
                volatility = 0.0
            
            all_sequences.append(seq)
            all_labels.append(lbl)
            all_returns.append(ret)
            all_spreads.append(current_spread)
            all_volatility.append(volatility)

        if (i + 1) % 20 == 0:
            print(f"  已加载 {i + 1}/{len(parquet_files)} 个文件, "
                  f"累计 {len(all_sequences)} 个样本")

    sequences = np.stack(all_sequences)
    labels = np.stack(all_labels)
    returns = np.stack(all_returns)
    spreads = np.array(all_spreads)
    volatility = np.array(all_volatility)

    feature_mean = sequences.reshape(-1, sequences.shape[-1]).mean(axis=0)
    feature_std = sequences.reshape(-1, sequences.shape[-1]).std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    print(f"数据加载完成: {sequences.shape[0]} 个样本")

    return sequences, labels, returns, spreads, volatility, feature_mean, feature_std


def extract_encoder_state(state_dict):
    """从完整模型的 state_dict 中提取编码器部分"""
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            new_key = k[8:]
            encoder_state[new_key] = v
    return encoder_state


def diagnose_predictions(predictions, labels, returns, spreads, volatility, probabilities, window_idx):
    """诊断预测结果
    
    输出详细的统计信息，帮助定位亏损原因
    """
    w = WINDOW_SIZES[window_idx]
    preds = predictions[:, window_idx]
    lbls = labels[:, window_idx]
    rets = returns[:, window_idx]
    probs = probabilities[:, window_idx]
    
    print(f"\n{'='*80}")
    print(f"窗口 label_{w} 诊断报告")
    print(f"{'='*80}")
    
    # 1. 整体统计
    print(f"\n【整体统计】")
    total_samples = len(preds)
    correct = (preds == lbls).sum()
    accuracy = correct / total_samples
    
    # 计算收益
    trade_mask = preds != 1
    num_trades = trade_mask.sum()
    direction = np.where(preds == 2, 1.0, np.where(preds == 0, -1.0, 0.0))
    profit = (direction[trade_mask] * rets[trade_mask] - FEE).sum()
    avg_profit = profit / num_trades if num_trades > 0 else 0
    
    print(f"总样本数: {total_samples}")
    print(f"预测准确率: {accuracy:.4f} ({correct}/{total_samples})")
    print(f"交易次数: {num_trades} ({num_trades/total_samples:.2%})")
    print(f"累计收益: {profit:.6f}")
    print(f"单次收益: {avg_profit:.6f}")
    
    # 2. 按预测方向统计
    print(f"\n【按预测方向统计】")
    for pred_label, pred_name in [(0, '下跌'), (1, '不变'), (2, '上涨')]:
        mask = preds == pred_label
        count = mask.sum()
        if count == 0:
            continue
        
        # 计算该预测方向的收益
        if pred_label == 1:
            # 预测不变，不交易
            profit_pred = 0
        else:
            # 预测涨跌，计算收益
            direction_pred = 1.0 if pred_label == 2 else -1.0
            profit_pred = (direction_pred * rets[mask] - FEE).sum()
        
        avg_profit_pred = profit_pred / count if count > 0 else 0
        
        # 计算准确率
        correct_pred = (lbls[mask] == pred_label).sum()
        acc_pred = correct_pred / count if count > 0 else 0
        
        print(f"  预测{pred_name}: {count} 次 ({count/total_samples:.2%}), "
              f"准确率={acc_pred:.4f}, 累计收益={profit_pred:.6f}, 单次收益={avg_profit_pred:.6f}")
    
    # 3. 按置信度分组统计
    print(f"\n【按置信度分组统计】")
    confidence = np.maximum(probs, 1 - probs)
    
    for low, high in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        mask = (confidence >= low) & (confidence < high)
        count = mask.sum()
        if count == 0:
            continue
        
        # 计算该置信度区间的收益
        trade_mask_conf = (preds[mask] != 1)
        if trade_mask_conf.sum() > 0:
            direction_conf = np.where(preds[mask][trade_mask_conf] == 2, 1.0, -1.0)
            profit_conf = (direction_conf * rets[mask][trade_mask_conf] - FEE).sum()
            avg_profit_conf = profit_conf / trade_mask_conf.sum()
        else:
            profit_conf = 0
            avg_profit_conf = 0
        
        # 计算准确率
        correct_conf = (preds[mask] == lbls[mask]).sum()
        acc_conf = correct_conf / count if count > 0 else 0
        
        print(f"  置信度 [{low:.1f}, {high:.1f}): {count} 次 ({count/total_samples:.2%}), "
              f"准确率={acc_conf:.4f}, 累计收益={profit_conf:.6f}, 单次收益={avg_profit_conf:.6f}")
    
    # 4. 按实际波动大小分组统计
    print(f"\n【按实际波动大小分组统计】")
    abs_returns = np.abs(rets)
    
    for low, high, name in [(0, 0.0001, '极小波动'), 
                            (0.0001, 0.0005, '小波动'),
                            (0.0005, 0.001, '中等波动'),
                            (0.001, float('inf'), '大波动')]:
        mask = (abs_returns >= low) & (abs_returns < high)
        count = mask.sum()
        if count == 0:
            continue
        
        # 计算该波动区间的收益
        trade_mask_vol = (preds[mask] != 1)
        if trade_mask_vol.sum() > 0:
            direction_vol = np.where(preds[mask][trade_mask_vol] == 2, 1.0, -1.0)
            profit_vol = (direction_vol * rets[mask][trade_mask_vol] - FEE).sum()
            avg_profit_vol = profit_vol / trade_mask_vol.sum()
        else:
            profit_vol = 0
            avg_profit_vol = 0
        
        # 计算准确率
        correct_vol = (preds[mask] == lbls[mask]).sum()
        acc_vol = correct_vol / count if count > 0 else 0
        
        print(f"  {name} [{low:.4f}, {high:.4f}): {count} 次 ({count/total_samples:.2%}), "
              f"准确率={acc_vol:.4f}, 累计收益={profit_vol:.6f}, 单次收益={avg_profit_vol:.6f}")
    
    # 5. 按价差分组统计
    print(f"\n【按价差分组统计】")
    for low, high, name in [(0, 0.0001, '小价差'),
                            (0.0001, 0.0002, '中等价差'),
                            (0.0002, float('inf'), '大价差')]:
        mask = (spreads >= low) & (spreads < high)
        count = mask.sum()
        if count == 0:
            continue
        
        # 计算该价差区间的收益
        trade_mask_spread = (preds[mask] != 1)
        if trade_mask_spread.sum() > 0:
            direction_spread = np.where(preds[mask][trade_mask_spread] == 2, 1.0, -1.0)
            profit_spread = (direction_spread * rets[mask][trade_mask_spread] - FEE).sum()
            avg_profit_spread = profit_spread / trade_mask_spread.sum()
        else:
            profit_spread = 0
            avg_profit_spread = 0
        
        # 计算准确率
        correct_spread = (preds[mask] == lbls[mask]).sum()
        acc_spread = correct_spread / count if count > 0 else 0
        
        print(f"  {name} [{low:.4f}, {high:.4f}): {count} 次 ({count/total_samples:.2%}), "
              f"准确率={acc_spread:.4f}, 累计收益={profit_spread:.6f}, 单次收益={avg_profit_spread:.6f}")
    
    # 6. 错误预测分析
    print(f"\n【错误预测分析】")
    wrong_mask = preds != lbls
    wrong_count = wrong_mask.sum()
    
    if wrong_count > 0:
        # 分析错误类型
        for true_label, true_name in [(0, '下跌'), (1, '不变'), (2, '上涨')]:
            for pred_label, pred_name in [(0, '下跌'), (1, '不变'), (2, '上涨')]:
                if true_label == pred_label:
                    continue
                
                mask = (lbls == true_label) & (preds == pred_label)
                count = mask.sum()
                if count == 0:
                    continue
                
                # 计算该错误类型的损失
                if pred_label == 1:
                    # 预测不变，不交易，损失为0
                    loss = 0
                else:
                    # 预测涨跌，但方向错误
                    direction_wrong = 1.0 if pred_label == 2 else -1.0
                    loss = (direction_wrong * rets[mask] - FEE).sum()
                
                avg_loss = loss / count if count > 0 else 0
                
                print(f"  真实{true_name} → 预测{pred_name}: {count} 次, "
                      f"累计损失={loss:.6f}, 单次损失={avg_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='收益诊断脚本')
    parser.add_argument('--data_dir', type=str, default='../../2026train_set/2026train_set')
    parser.add_argument('--encoder_path', type=str, default='output/tkan_encoder.pt')
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=512)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载编码器
    print("\n" + "="*80)
    print("步骤 1: 加载 T-KAN 编码器")
    print("="*80)
    
    checkpoint = torch.load(args.encoder_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    encoder = create_encoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        grid_size=config.get('grid_size', 8),
        spline_order=config.get('spline_order', 3),
        dropout=0.0,
    ).to(device)
    
    if 'ema_encoder_state' in checkpoint:
        ema_state = checkpoint['ema_encoder_state']
        if len(ema_state) > 0:
            encoder.load_state_dict(extract_encoder_state(ema_state), strict=False)
        else:
            encoder.load_state_dict(extract_encoder_state(checkpoint['encoder_state']), strict=False)
    else:
        encoder.load_state_dict(extract_encoder_state(checkpoint['encoder_state']), strict=False)
    
    encoder.eval()
    
    # 加载 LightGBM 模型
    lgbm_models = {}
    lgbm_model_strs = checkpoint.get('lgbm_models', None)
    if lgbm_model_strs is not None:
        for w in WINDOW_SIZES:
            key = f'w{w}'
            if key in lgbm_model_strs:
                lgbm_models[w] = lgb.Booster(model_str=lgbm_model_strs[key])
    
    thresholds = checkpoint.get('thresholds', {w: 0.6 for w in WINDOW_SIZES})
    
    print(f"编码器加载成功: hidden_dim={config['hidden_dim']}, num_layers={config['num_layers']}")
    print(f"LightGBM 模型加载成功: {len(lgbm_models)} 个窗口")
    print(f"阈值: {thresholds}")
    
    # 加载数据
    print("\n" + "="*80)
    print("步骤 2: 加载数据并提取特征")
    print("="*80)
    
    sequences, labels, returns, spreads, volatility, data_mean, data_std = load_data_with_returns(
        args.data_dir, args.max_files, args.sample_interval
    )
    
    sequences = (sequences - data_mean) / data_std
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 提取特征
    print("\n提取 T-KAN 特征...")
    all_features = []
    for start in tqdm(range(0, len(sequences), args.batch_size), desc="提取特征"):
        end = min(start + args.batch_size, len(sequences))
        batch = torch.from_numpy(sequences[start:end]).float().to(device)
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    features = encoder(batch)
            else:
                features = encoder(batch)
        all_features.append(features.cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    print(f"特征维度: {features.shape[1]}")
    
    # 预测
    print("\n" + "="*80)
    print("步骤 3: 预测并诊断")
    print("="*80)
    
    predictions = np.zeros((len(features), len(WINDOW_SIZES)), dtype=np.int32)
    probabilities = np.zeros((len(features), len(WINDOW_SIZES)), dtype=np.float32)
    
    for w_idx, w in enumerate(WINDOW_SIZES):
        if w not in lgbm_models:
            predictions[:, w_idx] = 1
            continue
        
        # 预测上涨概率
        p_up = lgbm_models[w].predict(features)
        probabilities[:, w_idx] = p_up
        
        # 阈值决策
        tau = thresholds.get(w, 0.6)
        predictions[:, w_idx] = np.where(
            p_up > tau, 2,
            np.where((1 - p_up) > tau, 0, 1)
        )
    
    # 诊断每个窗口
    for w_idx in range(len(WINDOW_SIZES)):
        diagnose_predictions(predictions, labels, returns, spreads, volatility, probabilities, w_idx)
    
    print("\n" + "="*80)
    print("诊断完成！")
    print("="*80)


if __name__ == '__main__':
    main()
