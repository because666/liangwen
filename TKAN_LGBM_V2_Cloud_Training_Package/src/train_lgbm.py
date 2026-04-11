"""
LightGBM 分类器训练脚本

训练流程：
1. 加载预训练的 T-KAN 编码器
2. 提取训练集/验证集的特征向量（128 维）
3. 训练 5 个独立的 LightGBM 三分类器
4. 评估并保存模型

前置条件：需先运行 train_tkan.py 完成编码器预训练

使用方式：
    python train_lgbm.py --data_dir ../../2026train_set/2026train_set --encoder_path output/tkan_encoder.pt
"""

import os
import sys
import json
import argparse
import gc
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import TKANEncoder, create_encoder

FEATURE_COLS = [
    'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'bid6', 'bid7', 'bid8', 'bid9', 'bid10',
    'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask6', 'ask7', 'ask8', 'ask9', 'ask10',
    'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'bsize6', 'bsize7', 'bsize8', 'bsize9', 'bsize10',
    'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'asize6', 'asize7', 'asize8', 'asize9', 'asize10',
    'mb_intst', 'ma_intst', 'lb_intst', 'la_intst', 'cb_intst', 'ca_intst',
]

LABEL_COLS = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']

WINDOW_SIZES = [5, 10, 20, 40, 60]

INPUT_DIM = len(FEATURE_COLS)

WINDOW_LABEL_MAP = {
    'label_5': 5, 'label_10': 10, 'label_20': 20, 'label_40': 40, 'label_60': 60,
}


def compute_midprice(bid1: np.ndarray, ask1: np.ndarray) -> np.ndarray:
    """计算中间价"""
    n = len(bid1)
    midprice = np.zeros(n, dtype=np.float32)
    both = (bid1 != 0) & (ask1 != 0)
    bid0 = (bid1 == 0) & (ask1 != 0)
    ask0 = (ask1 == 0) & (bid1 != 0)
    midprice[both] = (bid1[both] + ask1[both]) / 2
    midprice[bid0] = ask1[bid0]
    midprice[ask0] = bid1[ask0]
    return midprice


def clean_features(features: np.ndarray) -> np.ndarray:
    """特征清洗"""
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    for i, col in enumerate(FEATURE_COLS):
        if col.startswith('bid') or col.startswith('ask'):
            features[:, i] = np.clip(features[:, i], -0.3, 0.3)
        elif col.startswith('bsize') or col.startswith('asize'):
            features[:, i] = np.clip(features[:, i], 0, 100)
        elif col.endswith('_intst'):
            features[:, i] = np.clip(features[:, i], -10, 10)
    return features


def load_and_process_file(file_path: str) -> tuple:
    """加载并处理单个 parquet 文件

    返回：
        (features, labels, return_targets) 或 None
    """
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')

        missing = [c for c in FEATURE_COLS + LABEL_COLS if c not in df.columns]
        if missing:
            return None

        feature_list = []
        for col in FEATURE_COLS:
            feature_list.append(df[col].values.astype(np.float32))
        features = np.column_stack(feature_list)
        features = clean_features(features)

        labels = df[LABEL_COLS].values.astype(np.int64)

        bid1 = df['bid1'].values.astype(np.float32)
        ask1 = df['ask1'].values.astype(np.float32)
        midprice = compute_midprice(bid1, ask1)
        midprice = np.nan_to_num(midprice, nan=0.0, posinf=0.0, neginf=0.0)
        midprice = np.clip(midprice, -0.3, 0.3)

        return_targets = np.zeros((len(df), len(WINDOW_SIZES)), dtype=np.float32)
        for i, w in enumerate(WINDOW_SIZES):
            if w < len(df):
                return_targets[:-w, i] = midprice[w:] - midprice[:-w]

        return_targets = np.nan_to_num(return_targets, nan=0.0, posinf=0.0, neginf=0.0)
        return_targets = np.clip(return_targets, -0.1, 0.1)

        return features, labels, return_targets

    except Exception as e:
        print(f"加载失败 {os.path.basename(file_path)}: {e}")
        return None


def extract_features_from_file(file_path: str, encoder: TKANEncoder,
                               mean: np.ndarray, std: np.ndarray,
                               device: torch.device, batch_size: int = 512,
                               seq_len: int = 100) -> tuple:
    """从单个文件提取 T-KAN 特征

    参数：
        file_path: parquet 文件路径
        encoder: T-KAN 编码器
        mean: 归一化均值
        std: 归一化标准差
        device: 计算设备
        batch_size: 批量大小
        seq_len: 序列长度
    返回：
        (tkan_features, labels, return_targets) 或 None
        - tkan_features: (n_samples, hidden_dim)
        - labels: (n_samples, 5)
        - return_targets: (n_samples, 5)
    """
    result = load_and_process_file(file_path)
    if result is None:
        return None

    features, labels, return_targets = result

    if mean is not None and std is not None:
        features = (features - mean) / std
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    max_window = max(WINDOW_SIZES)
    valid_end = len(features) - max_window

    if valid_end <= seq_len:
        return None

    windows = []
    window_labels = []
    window_returns = []

    for j in range(seq_len, valid_end):
        windows.append(features[j - seq_len:j])
        window_labels.append(labels[j])
        window_returns.append(return_targets[j])

    windows = np.array(windows, dtype=np.float32)
    window_labels = np.array(window_labels, dtype=np.int64)
    window_returns = np.array(window_returns, dtype=np.float32)

    encoder.eval()
    all_features = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            end = min(start + batch_size, len(windows))
            batch = torch.from_numpy(windows[start:end]).to(device)
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    feat = encoder(batch)
            else:
                feat = encoder(batch)
            all_features.append(feat.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)

    return all_features, window_labels, window_returns


def compute_trading_metrics(preds: np.ndarray, labels: np.ndarray,
                            true_returns: np.ndarray, fee: float = 0.0002) -> dict:
    """计算交易指标

    参数：
        preds: (n_samples,) 预测标签 0/1/2
        labels: (n_samples,) 真实标签
        true_returns: (n_samples,) 真实收益
        fee: 手续费率
    返回：
        交易指标字典
    """
    trade_mask = preds != 1
    num_trades = trade_mask.sum()

    if num_trades == 0:
        return {
            'cumulative_return': 0.0,
            'single_return': 0.0,
            'total_trades': 0,
            'trade_accuracy': 0.0,
            'trade_rate': 0.0,
        }

    direction = np.where(preds == 2, 1.0, np.where(preds == 0, -1.0, 0.0))
    profit = (direction[trade_mask] * true_returns[trade_mask] - fee).sum()
    avg_profit = profit / num_trades

    correct = ((preds[trade_mask] == labels[trade_mask]) & (labels[trade_mask] != 1)).sum()
    accuracy = correct / num_trades

    return {
        'cumulative_return': float(profit),
        'single_return': float(avg_profit),
        'total_trades': int(num_trades),
        'trade_accuracy': float(accuracy),
        'trade_rate': float(num_trades / len(preds)),
    }


def main():
    parser = argparse.ArgumentParser(description='LightGBM 分类器训练')
    parser.add_argument('--data_dir', type=str, default='../../2026train_set/2026train_set')
    parser.add_argument('--encoder_path', type=str, default='output/tkan_encoder.pt')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lgbm_lr', type=float, default=0.1)
    parser.add_argument('--lgbm_num_leaves', type=int, default=63)
    parser.add_argument('--lgbm_max_depth', type=int, default=6)
    parser.add_argument('--lgbm_n_estimators', type=int, default=2000)
    parser.add_argument('--lgbm_early_stopping', type=int, default=100)
    parser.add_argument('--lgbm_feature_fraction', type=float, default=0.9)
    parser.add_argument('--lgbm_bagging_fraction', type=float, default=0.9)
    parser.add_argument('--lgbm_lambda_l1', type=float, default=0.01)
    parser.add_argument('--lgbm_lambda_l2', type=float, default=0.01)
    parser.add_argument('--lgbm_min_data_in_leaf', type=int, default=10)

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

    if 'ema_encoder_state' in checkpoint:
        ema_state = checkpoint['ema_encoder_state']
        if len(ema_state) > 0:
            encoder.load_state_dict(ema_state)
            print("加载 EMA 编码器权重")
        else:
            encoder.load_state_dict(checkpoint['encoder_state'])
            print("加载标准编码器权重")
    else:
        encoder.load_state_dict(checkpoint['encoder_state'])
        print("加载标准编码器权重")

    encoder.eval()
    print(f"编码器: input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}, "
          f"num_layers={config['num_layers']}")

    print("\n" + "=" * 60)
    print("步骤 2: 提取特征")
    print("=" * 60)

    all_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.parquet')])
    if args.max_files:
        all_files = all_files[:args.max_files]

    total = len(all_files)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    print(f"训练: {len(train_files)}, 验证: {len(val_files)}, 测试: {len(test_files)}")

    def extract_from_files(file_list, split_name):
        all_features = []
        all_labels = []
        all_returns = []

        for i, file in enumerate(tqdm(file_list, desc=f"提取{split_name}特征")):
            file_path = os.path.join(args.data_dir, file)
            result = extract_features_from_file(
                file_path, encoder, mean, std, device, args.batch_size
            )
            if result is not None:
                feat, lab, ret = result
                all_features.append(feat)
                all_labels.append(lab)
                all_returns.append(ret)

            if (i + 1) % 100 == 0:
                gc.collect()

        if len(all_features) == 0:
            return None, None, None

        return (
            np.concatenate(all_features, axis=0),
            np.concatenate(all_labels, axis=0),
            np.concatenate(all_returns, axis=0),
        )

    train_features, train_labels, train_returns = extract_from_files(train_files, "训练")
    val_features, val_labels, val_returns = extract_from_files(val_files, "验证")
    test_features, test_labels, test_returns = extract_from_files(test_files, "测试")

    if train_features is None:
        print("错误：未能提取训练特征！")
        return

    print(f"\n训练特征: {train_features.shape}")
    print(f"验证特征: {val_features.shape if val_features is not None else 'None'}")
    print(f"测试特征: {test_features.shape if test_features is not None else 'None'}")

    print("\n" + "=" * 60)
    print("步骤 3: 训练 LightGBM 分类器")
    print("=" * 60)

    lgbm_models = {}
    best_window = None
    best_cumulative_return = float('-inf')

    for w_idx, w_name in enumerate(LABEL_COLS):
        print(f"\n--- 训练 {w_name} ---")

        y_train = train_labels[:, w_idx]
        y_val = val_labels[:, w_idx] if val_labels is not None else None

        train_data = lgb.Dataset(train_features, label=y_train)

        valid_sets = [train_data]
        valid_names = ['train']
        if y_val is not None:
            val_data = lgb.Dataset(val_features, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
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
            'num_threads': 4,
        }

        callbacks = [
            lgb.log_evaluation(period=100),
        ]
        if y_val is not None:
            callbacks.append(lgb.early_stopping(args.lgbm_early_stopping, verbose=True))

        model = lgb.train(
            params,
            train_data,
            num_boost_round=args.lgbm_n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        model_path = os.path.join(args.output_dir, f'lgbm_w{WINDOW_SIZES[w_idx]}.txt')
        model.save_model(model_path)
        lgbm_models[w_name] = model
        print(f"模型已保存: {model_path}")

        if y_val is not None:
            val_pred_proba = model.predict(val_features)
            val_preds = val_pred_proba.argmax(axis=1)
            val_true_returns = val_returns[:, w_idx]

            metrics = compute_trading_metrics(val_preds, y_val, val_true_returns)
            print(f"验证指标 - 累计收益: {metrics['cumulative_return']:.6f}, "
                  f"单次收益: {metrics['single_return']:.6f}, "
                  f"交易率: {metrics['trade_rate']:.3f}, "
                  f"准确率: {metrics['trade_accuracy']:.3f}")

            if metrics['cumulative_return'] > best_cumulative_return:
                best_cumulative_return = metrics['cumulative_return']
                best_window = w_name

    print(f"\n最佳窗口: {best_window}, 累计收益: {best_cumulative_return:.6f}")

    print("\n" + "=" * 60)
    print("步骤 4: 测试评估")
    print("=" * 60)

    if test_features is not None:
        for w_idx, w_name in enumerate(LABEL_COLS):
            model = lgbm_models[w_name]
            test_pred_proba = model.predict(test_features)
            test_preds = test_pred_proba.argmax(axis=1)
            y_test = test_labels[:, w_idx]
            test_true_returns = test_returns[:, w_idx]

            metrics = compute_trading_metrics(test_preds, y_test, test_true_returns)
            print(f"{w_name} - 累计收益: {metrics['cumulative_return']:.6f}, "
                  f"单次收益: {metrics['single_return']:.6f}, "
                  f"交易率: {metrics['trade_rate']:.3f}")

    print("\n" + "=" * 60)
    print("步骤 5: 保存提交文件")
    print("=" * 60)

    submission_dir = os.path.join(args.output_dir, 'submission')
    os.makedirs(submission_dir, exist_ok=True)

    lgbm_model_strs = {}
    for w_idx, w_name in enumerate(LABEL_COLS):
        model = lgbm_models[w_name]
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp_path = tmp.name
        model.save_model(tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            lgbm_model_strs[f'w{WINDOW_SIZES[w_idx]}'] = f.read()
        os.unlink(tmp_path)

    src_encoder = os.path.join(args.output_dir, 'tkan_encoder.pt')
    if os.path.exists(src_encoder):
        base_checkpoint = torch.load(src_encoder, map_location='cpu', weights_only=False)
        base_checkpoint['lgbm_models'] = lgbm_model_strs
        torch.save(base_checkpoint, os.path.join(submission_dir, 'tkan_encoder.pt'))
        print(f"检查点已更新（含 LightGBM 模型）: tkan_encoder.pt")

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
    print("LightGBM 训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
