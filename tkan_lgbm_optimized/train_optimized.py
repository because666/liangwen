"""
T-KAN + LightGBM 综合优化训练脚本

包含所有优化方案：
1. 回归预测：直接预测价格变化率，根据预测值大小决定是否出手
2. 两阶段模型：先判断是否值得交易，再预测方向
3. 样本重加权优化：指数加权，更激进地关注大波动样本
4. 代价敏感学习：不同错误类型设置不同惩罚

使用方法：
python train_optimized.py --mode [regression|two_stage|cost_sensitive|all]
"""

import os
import argparse
import json
import tempfile
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from model import create_encoder

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


def compute_cumulative_return(preds, labels, returns, fee=FEE):
    """计算累计收益率"""
    trade_mask = preds != 1
    num_trades = trade_mask.sum()

    if num_trades == 0:
        return {
            'cumulative_return': 0.0,
            'single_return': 0.0,
            'total_trades': 0,
            'trade_rate': 0.0,
        }

    direction = np.where(preds == 2, 1.0, np.where(preds == 0, -1.0, 0.0))
    profit = (direction[trade_mask] * returns[trade_mask] - fee).sum()
    avg_profit = profit / num_trades

    return {
        'cumulative_return': float(profit),
        'single_return': float(avg_profit),
        'total_trades': int(num_trades),
        'trade_rate': float(num_trades / len(preds)),
    }


def search_optimal_threshold(proba, labels, returns, fee=FEE):
    """搜索最优阈值"""
    best_tau = 0.5
    best_cum_return = -float('inf')
    best_metrics = None

    for tau_int in range(50, 99):
        tau = tau_int / 100.0

        preds = np.where(
            proba > tau, 2,
            np.where((1 - proba) > tau, 0, 1)
        )

        metrics = compute_cumulative_return(preds, labels, returns, fee)
        cum_return = metrics['cumulative_return']

        if cum_return > best_cum_return:
            best_cum_return = cum_return
            best_tau = tau
            best_metrics = metrics

    return best_tau, best_cum_return, best_metrics


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

        for t in range(100, len(df) - max_window, sample_interval):
            seq = feature_data[t - 100:t]
            lbl = label_data[t]
            ret = np.array([
                midprice[t + w] - midprice[t] for w in WINDOW_SIZES
            ], dtype=np.float32)
            all_sequences.append(seq)
            all_labels.append(lbl)
            all_returns.append(ret)

        if (i + 1) % 20 == 0:
            print(f"  已加载 {i + 1}/{len(parquet_files)} 个文件, "
                  f"累计 {len(all_sequences)} 个样本")

    sequences = np.stack(all_sequences)
    labels = np.stack(all_labels)
    returns = np.stack(all_returns)

    feature_mean = sequences.reshape(-1, sequences.shape[-1]).mean(axis=0)
    feature_std = sequences.reshape(-1, sequences.shape[-1]).std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    print(f"数据加载完成: {sequences.shape[0]} 个样本")

    return sequences, labels, returns, feature_mean, feature_std


def _extract_encoder_state(state_dict):
    """提取编码器权重"""
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            new_key = k[8:]
            encoder_state[new_key] = v
    return encoder_state


def train_regression_mode(args, device, encoder, sequences, labels, returns, feature_mean, feature_std):
    """训练回归预测版本
    
    直接预测价格变化率，根据预测值大小决定是否出手
    """
    print("\n" + "=" * 80)
    print("优化方案 1: 回归预测模式")
    print("=" * 80)
    
    import lightgbm as lgb
    
    sequences = (sequences - feature_mean) / feature_std
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples = len(sequences)
    n_val = min(int(n_samples * 0.1), 50000)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    print(f"训练集: {n_train} 样本, 验证集: {n_val} 样本")

    print("\n提取 T-KAN 特征...")
    
    def extract_features(data_sequences, batch_size=512, desc="提取特征"):
        encoder.eval()
        all_features = []
        n_batches = (len(data_sequences) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(data_sequences), batch_size), desc=desc, total=n_batches, leave=False)
        for start in pbar:
            end = min(start + batch_size, len(data_sequences))
            batch = torch.from_numpy(data_sequences[start:end]).float().to(device)
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        features = encoder(batch)
                else:
                    features = encoder(batch)
            all_features.append(features.cpu().numpy())
            pbar.set_postfix({'batch': f'{start//batch_size + 1}/{n_batches}'})
        return np.concatenate(all_features, axis=0)

    train_features = extract_features(sequences[train_idx], desc="提取训练特征")
    val_features = extract_features(sequences[val_idx], desc="提取验证特征")

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    train_returns = returns[train_idx]
    val_returns = returns[val_idx]

    print(f"特征维度: {train_features.shape[1]}")

    lgbm_models = {}
    regression_thresholds = {}

    for w_idx, w in tqdm(enumerate(WINDOW_SIZES), desc="训练回归模型", total=len(WINDOW_SIZES)):
        print(f"\n--- 窗口 label_{w} ---")

        train_ret = train_returns[:, w_idx]
        val_ret = val_returns[:, w_idx]
        val_lbl = val_labels[:, w_idx]

        # 训练回归模型
        train_data = lgb.Dataset(train_features, label=train_ret)
        val_data = lgb.Dataset(val_features, label=val_ret)

        params = {
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 5,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': args.seed,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        lgbm_models[w] = model
        print(f"  回归模型训练完成, 迭代次数: {model.best_iteration}")

        # 搜索最优回归阈值
        val_pred = model.predict(val_features)
        
        best_threshold = 0.0005
        best_cum_return = -float('inf')
        
        for threshold_int in range(1, 20):
            threshold = threshold_int * 0.0001  # 0.0001 ~ 0.0019
            
            preds = np.where(
                val_pred > threshold, 2,
                np.where(val_pred < -threshold, 0, 1)
            )
            
            metrics = compute_cumulative_return(preds, val_lbl, val_ret)
            if metrics['cumulative_return'] > best_cum_return:
                best_cum_return = metrics['cumulative_return']
                best_threshold = threshold
                best_metrics = metrics
        
        regression_thresholds[w] = best_threshold
        print(f"  最优回归阈值: {best_threshold:.4f}")
        print(f"  验证集累计收益: {best_metrics['cumulative_return']:.6f}")
        print(f"  单次收益: {best_metrics['single_return']:.6f}")
        print(f"  交易次数: {best_metrics['total_trades']}")

    return lgbm_models, regression_thresholds


def train_two_stage_mode(args, device, encoder, sequences, labels, returns, feature_mean, feature_std):
    """训练两阶段模型
    
    阶段1: 判断是否值得交易（波动是否够大）
    阶段2: 只对值得交易的样本预测方向
    """
    print("\n" + "=" * 80)
    print("优化方案 2: 两阶段模型")
    print("=" * 80)
    
    import lightgbm as lgb
    
    sequences = (sequences - feature_mean) / feature_std
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples = len(sequences)
    n_val = min(int(n_samples * 0.1), 50000)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    print(f"训练集: {n_train} 样本, 验证集: {n_val} 样本")

    print("\n提取 T-KAN 特征...")
    
    def extract_features(data_sequences, batch_size=512, desc="提取特征"):
        encoder.eval()
        all_features = []
        n_batches = (len(data_sequences) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(data_sequences), batch_size), desc=desc, total=n_batches, leave=False)
        for start in pbar:
            end = min(start + batch_size, len(data_sequences))
            batch = torch.from_numpy(data_sequences[start:end]).float().to(device)
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        features = encoder(batch)
                else:
                    features = encoder(batch)
            all_features.append(features.cpu().numpy())
            pbar.set_postfix({'batch': f'{start//batch_size + 1}/{n_batches}'})
        return np.concatenate(all_features, axis=0)

    train_features = extract_features(sequences[train_idx], desc="提取训练特征")
    val_features = extract_features(sequences[val_idx], desc="提取验证特征")

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    train_returns = returns[train_idx]
    val_returns = returns[val_idx]

    print(f"特征维度: {train_features.shape[1]}")

    # 定义"值得交易"的阈值（波动大于某个值）
    WORTH_TRADING_THRESHOLD = 0.0005
    
    stage1_models = {}
    stage2_models = {}
    thresholds = {}

    for w_idx, w in tqdm(enumerate(WINDOW_SIZES), desc="训练两阶段模型", total=len(WINDOW_SIZES)):
        print(f"\n--- 窗口 label_{w} ---")

        train_lbl = train_labels[:, w_idx]
        val_lbl = val_labels[:, w_idx]
        train_ret = train_returns[:, w_idx]
        val_ret = val_returns[:, w_idx]

        # 阶段1: 判断是否值得交易（二分类：波动是否够大）
        train_worth_trading = (np.abs(train_ret) > WORTH_TRADING_THRESHOLD).astype(np.int32)
        val_worth_trading = (np.abs(val_ret) > WORTH_TRADING_THRESHOLD).astype(np.int32)

        train_data_stage1 = lgb.Dataset(train_features, label=train_worth_trading)
        val_data_stage1 = lgb.Dataset(val_features, label=val_worth_trading)

        params_stage1 = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 5,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': args.seed,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ]

        stage1_model = lgb.train(
            params_stage1,
            train_data_stage1,
            num_boost_round=500,
            valid_sets=[val_data_stage1],
            callbacks=callbacks,
        )

        stage1_models[w] = stage1_model
        print(f"  阶段1模型训练完成（判断是否值得交易）")

        # 阶段2: 只对值得交易的样本预测方向
        train_mask = train_lbl != 1
        train_X = train_features[train_mask]
        train_y_raw = train_lbl[train_mask]
        train_y = (train_y_raw == 2).astype(np.int32)  # 0=下跌, 1=上涨
        train_ret_filtered = train_ret[train_mask]

        # 样本权重（指数加权）
        sample_weights = np.exp(np.abs(train_ret_filtered) * 100)
        sample_weights = np.clip(sample_weights, 0.5, 10.0)

        train_data_stage2 = lgb.Dataset(train_X, label=train_y, weight=sample_weights)
        val_data_stage2 = lgb.Dataset(val_features, label=(val_lbl == 2).astype(np.int32))

        params_stage2 = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 5,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': args.seed,
        }

        stage2_model = lgb.train(
            params_stage2,
            train_data_stage2,
            num_boost_round=1000,
            valid_sets=[val_data_stage2],
            callbacks=callbacks,
        )

        stage2_models[w] = stage2_model
        print(f"  阶段2模型训练完成（预测方向）")

        # 搜索最优阈值
        val_proba_stage1 = stage1_model.predict(val_features)
        val_proba_stage2 = stage2_model.predict(val_features)

        best_tau = 0.6
        best_cum_return = -float('inf')

        for tau_int in range(50, 99):
            tau = tau_int / 100.0

            # 阶段1：判断是否值得交易
            worth_trading = val_proba_stage1 > 0.5

            # 阶段2：预测方向
            preds = np.where(
                worth_trading & (val_proba_stage2 > tau), 2,
                np.where(worth_trading & ((1 - val_proba_stage2) > tau), 0, 1)
            )

            metrics = compute_cumulative_return(preds, val_lbl, val_ret)
            if metrics['cumulative_return'] > best_cum_return:
                best_cum_return = metrics['cumulative_return']
                best_tau = tau
                best_metrics = metrics

        thresholds[w] = best_tau
        print(f"  最优阈值: {best_tau:.2f}")
        print(f"  验证集累计收益: {best_metrics['cumulative_return']:.6f}")
        print(f"  单次收益: {best_metrics['single_return']:.6f}")
        print(f"  交易次数: {best_metrics['total_trades']}")

    return stage1_models, stage2_models, thresholds


def train_cost_sensitive_mode(args, device, encoder, sequences, labels, returns, feature_mean, feature_std):
    """训练代价敏感学习版本
    
    对不同错误类型设置不同惩罚
    """
    print("\n" + "=" * 80)
    print("优化方案 3: 代价敏感学习")
    print("=" * 80)
    
    import lightgbm as lgb
    
    sequences = (sequences - feature_mean) / feature_std
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples = len(sequences)
    n_val = min(int(n_samples * 0.1), 50000)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    print(f"训练集: {n_train} 样本, 验证集: {n_val} 样本")

    print("\n提取 T-KAN 特征...")
    
    def extract_features(data_sequences, batch_size=512, desc="提取特征"):
        encoder.eval()
        all_features = []
        n_batches = (len(data_sequences) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(data_sequences), batch_size), desc=desc, total=n_batches, leave=False)
        for start in pbar:
            end = min(start + batch_size, len(data_sequences))
            batch = torch.from_numpy(data_sequences[start:end]).float().to(device)
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        features = encoder(batch)
                else:
                    features = encoder(batch)
            all_features.append(features.cpu().numpy())
            pbar.set_postfix({'batch': f'{start//batch_size + 1}/{n_batches}'})
        return np.concatenate(all_features, axis=0)

    train_features = extract_features(sequences[train_idx], desc="提取训练特征")
    val_features = extract_features(sequences[val_idx], desc="提取验证特征")

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    train_returns = returns[train_idx]
    val_returns = returns[val_idx]

    print(f"特征维度: {train_features.shape[1]}")

    lgbm_models = {}
    thresholds = {}

    for w_idx, w in tqdm(enumerate(WINDOW_SIZES), desc="训练代价敏感模型", total=len(WINDOW_SIZES)):
        print(f"\n--- 窗口 label_{w} ---")

        train_lbl = train_labels[:, w_idx]
        val_lbl = val_labels[:, w_idx]
        train_ret = train_returns[:, w_idx]
        val_ret = val_returns[:, w_idx]

        # 过滤 label=1（不变）的样本
        train_mask = train_lbl != 1
        train_X = train_features[train_mask]
        train_y_raw = train_lbl[train_mask]
        train_y = (train_y_raw == 2).astype(np.int32)  # 0=下跌, 1=上涨
        train_ret_filtered = train_ret[train_mask]

        # 代价敏感权重
        # 预测涨但实际跌：损失大（权重高）
        # 预测跌但实际涨：损失大（权重高）
        # 预测涨且实际涨：收益大（权重高）
        # 预测跌且实际跌：收益大（权重高）
        
        sample_weights = np.ones(len(train_y), dtype=np.float32)
        
        for i in range(len(train_y)):
            actual_direction = train_y[i]  # 0=下跌, 1=上涨
            actual_return = train_ret_filtered[i]
            
            # 根据实际收益设置权重
            # 大波动样本权重高
            weight = np.exp(np.abs(actual_return) * 100)
            
            # 如果是上涨样本，预测错的代价更大
            if actual_direction == 1:  # 实际上涨
                # 预测跌的代价是预测涨的2倍
                weight *= 1.5
            else:  # 实际下跌
                # 预测涨的代价是预测跌的2倍
                weight *= 1.5
            
            sample_weights[i] = np.clip(weight, 0.5, 20.0)

        train_data = lgb.Dataset(train_X, label=train_y, weight=sample_weights)
        val_data = lgb.Dataset(val_features, label=(val_lbl == 2).astype(np.int32))

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 5,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': args.seed,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        lgbm_models[w] = model
        print(f"  代价敏感模型训练完成, 迭代次数: {model.best_iteration}")

        # 阈值搜索
        val_proba = model.predict(val_features)

        best_tau, best_cum_return, best_metrics = search_optimal_threshold(
            val_proba, val_lbl, val_ret
        )
        thresholds[w] = best_tau

        print(f"  最优阈值: {best_tau:.2f}")
        print(f"  验证集累计收益: {best_metrics['cumulative_return']:.6f}")
        print(f"  单次收益: {best_metrics['single_return']:.6f}")
        print(f"  交易次数: {best_metrics['total_trades']}")

    return lgbm_models, thresholds


def main():
    parser = argparse.ArgumentParser(description='T-KAN + LightGBM 综合优化训练')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['regression', 'two_stage', 'cost_sensitive', 'all'],
                       help='优化模式')
    parser.add_argument('--data_dir', type=str, default='../../2026train_set/2026train_set')
    parser.add_argument('--encoder_path', type=str, default='output/tkan_encoder.pt')
    parser.add_argument('--output_dir', type=str, default='output_optimized')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("步骤 1: 加载 T-KAN 编码器")
    print("=" * 80)

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
            encoder.load_state_dict(_extract_encoder_state(ema_state), strict=False)
        else:
            encoder.load_state_dict(_extract_encoder_state(checkpoint['encoder_state']), strict=False)
    else:
        encoder.load_state_dict(_extract_encoder_state(checkpoint['encoder_state']), strict=False)

    encoder.eval()
    print(f"编码器: input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}, "
          f"num_layers={config['num_layers']}")

    print("\n" + "=" * 80)
    print("步骤 2: 加载数据并提取特征")
    print("=" * 80)

    sequences, labels, returns, data_mean, data_std = load_data_with_returns(
        args.data_dir, args.max_files, args.sample_interval
    )

    # 根据模式训练
    if args.mode == 'regression' or args.mode == 'all':
        lgbm_models, regression_thresholds = train_regression_mode(
            args, device, encoder, sequences, labels, returns, data_mean, data_std
        )
        
        # 保存回归模型
        if args.mode == 'regression':
            save_models(args.output_dir, checkpoint, lgbm_models, regression_thresholds, 
                       data_mean, data_std, regression_mode=True)

    if args.mode == 'two_stage' or args.mode == 'all':
        stage1_models, stage2_models, thresholds = train_two_stage_mode(
            args, device, encoder, sequences, labels, returns, data_mean, data_std
        )
        
        # 保存两阶段模型
        if args.mode == 'two_stage':
            save_two_stage_models(args.output_dir, checkpoint, stage1_models, stage2_models, 
                                 thresholds, data_mean, data_std)

    if args.mode == 'cost_sensitive' or args.mode == 'all':
        lgbm_models, thresholds = train_cost_sensitive_mode(
            args, device, encoder, sequences, labels, returns, data_mean, data_std
        )
        
        # 保存代价敏感模型
        if args.mode == 'cost_sensitive':
            save_models(args.output_dir, checkpoint, lgbm_models, thresholds, 
                       data_mean, data_std, cost_sensitive_mode=True)

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)


def save_models(output_dir, checkpoint, lgbm_models, thresholds, data_mean, data_std, 
               regression_mode=False, cost_sensitive_mode=False):
    """保存模型"""
    import tempfile
    
    submission_dir = os.path.join(output_dir, 'submission')
    os.makedirs(submission_dir, exist_ok=True)

    lgbm_model_strs = {}
    for w in WINDOW_SIZES:
        model = lgbm_models[w]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as tmp:
            tmp_path = tmp.name
        model.save_model(tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            lgbm_model_strs[f'w{w}'] = f.read()
        os.unlink(tmp_path)

    base_checkpoint = {
        'encoder_state': checkpoint['encoder_state'],
        'ema_encoder_state': checkpoint.get('ema_encoder_state', {}),
        'config': checkpoint['config'],
        'mean': data_mean,
        'std': data_std,
        'lgbm_models': lgbm_model_strs,
        'thresholds': thresholds,
        'regression_mode': regression_mode,
        'cost_sensitive_mode': cost_sensitive_mode,
    }
    
    if regression_mode:
        base_checkpoint['regression_thresholds'] = thresholds

    torch.save(base_checkpoint, os.path.join(submission_dir, 'tkan_encoder.pt'))
    print(f"模型已保存到: {submission_dir}")


def save_two_stage_models(output_dir, checkpoint, stage1_models, stage2_models, thresholds, data_mean, data_std):
    """保存两阶段模型"""
    import tempfile
    
    submission_dir = os.path.join(output_dir, 'submission')
    os.makedirs(submission_dir, exist_ok=True)

    stage1_model_strs = {}
    stage2_model_strs = {}
    
    for w in WINDOW_SIZES:
        # 保存阶段1模型
        model = stage1_models[w]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as tmp:
            tmp_path = tmp.name
        model.save_model(tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            stage1_model_strs[f'w{w}'] = f.read()
        os.unlink(tmp_path)
        
        # 保存阶段2模型
        model = stage2_models[w]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as tmp:
            tmp_path = tmp.name
        model.save_model(tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            stage2_model_strs[f'w{w}'] = f.read()
        os.unlink(tmp_path)

    base_checkpoint = {
        'encoder_state': checkpoint['encoder_state'],
        'ema_encoder_state': checkpoint.get('ema_encoder_state', {}),
        'config': checkpoint['config'],
        'mean': data_mean,
        'std': data_std,
        'stage1_models': stage1_model_strs,
        'stage2_models': stage2_model_strs,
        'thresholds': thresholds,
        'two_stage_mode': True,
    }

    torch.save(base_checkpoint, os.path.join(submission_dir, 'tkan_encoder.pt'))
    print(f"两阶段模型已保存到: {submission_dir}")


if __name__ == '__main__':
    main()
