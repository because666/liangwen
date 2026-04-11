"""
LightGBM 二分类训练脚本（收益加权 + 阈值搜索版本）

核心改进：
1. 二分类：只训练上涨/下跌样本（过滤"不变"类），模型专注方向判断
2. 收益加权样本权重：高收益率样本获得更高权重
3. 阈值搜索：在验证集上搜索最优出手阈值 τ，最大化累计收益
4. 推理决策：若 max(p_up, p_down) > τ 则出手，否则预测"不变"

训练流程：
1. 加载 T-KAN 编码器 → 提取 128 维特征
2. 过滤 label=1 样本 → 二分类标签（0=下跌, 1=上涨）
3. 计算样本权重 = max(0.5, |实际收益率| * 10)
4. 训练 LightGBM 二分类器
5. 验证集搜索最优阈值 τ
6. 保存模型 + 阈值到检查点
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
    """特征清洗：处理 NaN/Inf，按列类型裁剪"""
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    for i, col in enumerate(FEATURE_COLS):
        if col.startswith('bid') or col.startswith('ask'):
            features[:, i] = np.clip(features[:, i], -0.3, 0.3)
        elif col.startswith('bsize') or col.startswith('asize'):
            features[:, i] = np.clip(features[:, i], 0, 100)
    return features


def compute_cumulative_return(preds, labels, returns, fee=FEE):
    """计算累计收益率（竞赛评测标准）

    参数：
        preds: 预测方向 (0=下跌, 1=不变, 2=上涨)
        labels: 真实标签 (0/1/2)
        returns: 实际中间价变化率
        fee: 手续费（0.01%）
    返回：
        包含累计收益、单次收益、交易次数等指标的字典
    """
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
    """搜索最优出手阈值 τ

    在验证集上遍历不同阈值，找到最大化累计收益的 τ 值。

    参数：
        proba: 上涨概率 (n_samples,)
        labels: 真实标签 (0/1/2)
        returns: 实际中间价变化率
        fee: 手续费
    返回：
        (best_tau, best_cum_return, best_metrics)
    """
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
    """加载数据并计算实际收益率（带采样）

    参数：
        data_dir: 数据目录路径
        max_files: 最大加载文件数
        sample_interval: 采样间隔，每 N 个 tick 取 1 个样本
    返回：
        (sequences, labels, returns, feature_mean, feature_std)
        sequences: (n_samples, 100, 40)
        labels: (n_samples, 5) 原始标签 0/1/2
        returns: (n_samples, 5) 实际中间价变化率
    """
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

        # 采样：每 sample_interval 个 tick 取 1 个样本
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


def main():
    parser = argparse.ArgumentParser(description='LightGBM 二分类训练（收益加权 + 阈值搜索）')
    parser.add_argument('--data_dir', type=str, default='../../2026train_set/2026train_set')
    parser.add_argument('--encoder_path', type=str, default='output/tkan_encoder.pt')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lgbm_lr', type=float, default=0.05)
    parser.add_argument('--lgbm_num_leaves', type=int, default=31)
    parser.add_argument('--lgbm_max_depth', type=int, default=5)
    parser.add_argument('--lgbm_n_estimators', type=int, default=1000)
    parser.add_argument('--lgbm_early_stopping', type=int, default=50)
    parser.add_argument('--lgbm_feature_fraction', type=float, default=0.8)
    parser.add_argument('--lgbm_bagging_fraction', type=float, default=0.8)
    parser.add_argument('--lgbm_lambda_l1', type=float, default=0.0)
    parser.add_argument('--lgbm_lambda_l2', type=float, default=0.0)
    parser.add_argument('--lgbm_min_data_in_leaf', type=int, default=20)
    parser.add_argument('--profit_scale', type=float, default=10.0)
    parser.add_argument('--profit_min_weight', type=float, default=0.5)

    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        print("错误：未安装 lightgbm，请运行 pip install lightgbm")
        return

    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    print("\n" + "=" * 60)
    print("步骤 1: 加载 T-KAN 编码器")
    print("=" * 60)

    if not os.path.exists(args.encoder_path):
        print(f"错误：编码器文件不存在: {args.encoder_path}")
        print("请先运行 train_tkan.py 完成编码器预训练")
        return

    checkpoint = torch.load(args.encoder_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    mean = checkpoint['mean']
    std = checkpoint['std']

    encoder = create_encoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        grid_size=config.get('grid_size', 8),
        spline_order=config.get('spline_order', 3),
        dropout=0.0,
    ).to(device)

    # 处理 state_dict：只保留编码器部分，移除 'encoder.' 前缀
    def _extract_encoder_state(state_dict):
        """从完整模型的 state_dict 中提取编码器部分
        
        参数：
            state_dict: 完整模型（TKANRegressionModel）的 state_dict
        返回：
            仅包含编码器权重的 state_dict
        """
        encoder_state = {}
        for k, v in state_dict.items():
            # 只保留以 'encoder.' 开头的键
            if k.startswith('encoder.'):
                # 移除 'encoder.' 前缀
                new_key = k[8:]
                encoder_state[new_key] = v
        return encoder_state

    # 加载编码器权重（strict=False 忽略缺失的 buffer，如 grid）
    if 'ema_encoder_state' in checkpoint:
        ema_state = checkpoint['ema_encoder_state']
        if len(ema_state) > 0:
            encoder.load_state_dict(_extract_encoder_state(ema_state), strict=False)
            print("加载 EMA 编码器权重")
        else:
            encoder.load_state_dict(_extract_encoder_state(checkpoint['encoder_state']), strict=False)
            print("加载标准编码器权重")
    else:
        encoder.load_state_dict(_extract_encoder_state(checkpoint['encoder_state']), strict=False)
        print("加载标准编码器权重")

    encoder.eval()
    print(f"编码器: input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}, "
          f"num_layers={config['num_layers']}")

    print("\n" + "=" * 60)
    print("步骤 2: 加载数据并提取特征")
    print("=" * 60)

    sequences, labels, returns, data_mean, data_std = load_data_with_returns(
        args.data_dir, args.max_files, args.sample_interval
    )

    sequences = (sequences - data_mean) / data_std
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

    print("\n" + "=" * 60)
    print("步骤 3: 训练 LightGBM 二分类器")
    print("=" * 60)

    lgbm_models = {}
    optimal_thresholds = {}

    for w_idx, w in tqdm(enumerate(WINDOW_SIZES), desc="训练 LightGBM", total=len(WINDOW_SIZES)):
        print(f"\n--- 窗口 label_{w} ---")

        train_lbl = train_labels[:, w_idx]
        val_lbl = val_labels[:, w_idx]
        train_ret = train_returns[:, w_idx]
        val_ret = val_returns[:, w_idx]

        # 过滤 label=1（不变）的样本，只保留上涨/下跌
        train_mask = train_lbl != 1
        train_X = train_features[train_mask]
        train_y_raw = train_lbl[train_mask]
        train_y = (train_y_raw == 2).astype(np.int32)  # 0=下跌, 1=上涨
        train_ret_filtered = train_ret[train_mask]

        # 收益加权样本权重
        sample_weights = np.maximum(
            args.profit_min_weight,
            np.abs(train_ret_filtered) * args.profit_scale
        )

        n_up = (train_y == 1).sum()
        n_down = (train_y == 0).sum()
        print(f"  训练样本: {len(train_y)} (上涨={n_up}, 下跌={n_down}), "
              f"过滤不变样本 {train_mask.shape[0] - train_mask.sum()} 个")
        print(f"  样本权重: min={sample_weights.min():.3f}, "
              f"max={sample_weights.max():.3f}, mean={sample_weights.mean():.3f}")

        # 验证集（保留全部样本用于阈值搜索）
        val_X = val_features
        val_y_raw = val_lbl

        # 训练 LightGBM 二分类
        train_data = lgb.Dataset(
            train_X, label=train_y, weight=sample_weights
        )
        val_data = lgb.Dataset(val_X, label=(val_y_raw == 2).astype(np.int32))

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': args.lgbm_lr,
            'num_leaves': args.lgbm_num_leaves,
            'max_depth': args.lgbm_max_depth,
            'feature_fraction': args.lgbm_feature_fraction,
            'bagging_fraction': args.lgbm_bagging_fraction,
            'bagging_freq': 5,
            'lambda_l1': args.lgbm_lambda_l1,
            'lambda_l2': args.lgbm_lambda_l2,
            'min_data_in_leaf': args.lgbm_min_data_in_leaf,
            'verbose': -1,
            'seed': args.seed,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=args.lgbm_early_stopping),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=args.lgbm_n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        lgbm_models[w] = model
        print(f"  LightGBM 训练完成, 迭代次数: {model.best_iteration}")

        # 阈值搜索
        val_proba = model.predict(val_X)

        best_tau, best_cum_return, best_metrics = search_optimal_threshold(
            val_proba, val_y_raw, val_ret
        )
        optimal_thresholds[w] = best_tau

        print(f"  最优阈值 τ={best_tau:.2f}")
        print(f"  验证集累计收益: {best_metrics['cumulative_return']:.6f}")
        print(f"  单次收益: {best_metrics['single_return']:.6f}")
        print(f"  交易次数: {best_metrics['total_trades']}")
        print(f"  交易率: {best_metrics['trade_rate']:.4f}")

        # 对比固定阈值 0.5
        preds_05 = np.where(
            val_proba > 0.5, 2,
            np.where((1 - val_proba) > 0.5, 0, 1)
        )
        metrics_05 = compute_cumulative_return(preds_05, val_y_raw, val_ret)
        print(f"  [对比] τ=0.50: 累计收益={metrics_05['cumulative_return']:.6f}, "
              f"交易={metrics_05['total_trades']}")

    print("\n" + "=" * 60)
    print("步骤 4: 保存提交文件")
    print("=" * 60)

    submission_dir = os.path.join(args.output_dir, 'submission')
    os.makedirs(submission_dir, exist_ok=True)

    lgbm_model_strs = {}
    for w_idx, w in enumerate(WINDOW_SIZES):
        model = lgbm_models[w]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as tmp:
            tmp_path = tmp.name
        model.save_model(tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            lgbm_model_strs[f'w{w}'] = f.read()
        os.unlink(tmp_path)

    src_encoder = os.path.join(args.output_dir, 'tkan_encoder.pt')
    if os.path.exists(src_encoder):
        base_checkpoint = torch.load(src_encoder, map_location='cpu', weights_only=False)
        base_checkpoint['lgbm_models'] = lgbm_model_strs
        base_checkpoint['thresholds'] = optimal_thresholds
        base_checkpoint['binary_mode'] = True
        torch.save(base_checkpoint, os.path.join(submission_dir, 'tkan_encoder.pt'))
        print(f"检查点已更新（含 LightGBM 二分类模型 + 阈值）")
    else:
        print(f"警告: 未找到编码器文件 {src_encoder}")

    config_data = {
        'python_version': '3.10',
        'batch': 256,
        'feature': FEATURE_COLS,
        'label': LABEL_COLS,
    }
    with open(os.path.join(submission_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(submission_dir, 'requirements.txt'), 'w', encoding='utf-8') as f:
        f.write('torch>=2.0.0\npandas>=2.0.0\nnumpy>=1.24.0\npyarrow>=10.0.0\nlightgbm>=4.0.0\n')

    print(f"提交文件已保存到: {submission_dir}")
    print("文件列表:")
    for f_name in os.listdir(submission_dir):
        f_path = os.path.join(submission_dir, f_name)
        f_size = os.path.getsize(f_path) / 1024 / 1024
        print(f"  {f_name} ({f_size:.2f} MB)")

    print("\n" + "=" * 60)
    print("各窗口最优阈值汇总:")
    print("=" * 60)
    for w in WINDOW_SIZES:
        tau = optimal_thresholds[w]
        print(f"  label_{w}: τ = {tau:.2f}")

    print("\nLightGBM 二分类训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
