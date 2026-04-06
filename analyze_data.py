"""
数据分析脚本 - 检测数据异常和质量问题
"""
import pandas as pd
import numpy as np
import os

data_dir = 'd:/量化/良文杯/2026train_set/2026train_set'
files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

print(f"总文件数: {len(files)}")

# 读取第一个文件分析
sample_file = os.path.join(data_dir, files[0])
df = pd.read_parquet(sample_file)

print(f"\n=== 数据基本信息 ===")
print(f"数据形状: {df.shape}")
print(f"列数: {len(df.columns)}")
print(f"前10列: {list(df.columns[:10])}")
label_cols = [c for c in df.columns if 'label' in c]
print(f"标签列: {label_cols}")

print(f"\n=== 数据统计 ===")
print(df.describe().T.head(20))

print(f"\n=== 检查异常值 ===")
nan_counts = df.isna().sum()
nan_cols = nan_counts[nan_counts > 0]
if len(nan_cols) > 0:
    print(f"NaN数量:\n{nan_cols}")
else:
    print("没有NaN值")

numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_counts = (np.isinf(df[numeric_cols])).sum()
inf_cols = inf_counts[inf_counts > 0]
if len(inf_cols) > 0:
    print(f"\nInf数量:\n{inf_cols}")
else:
    print("没有Inf值")

print(f"\n=== 涨跌停检测 ===")
print(f"bid1=0的数量: {(df['bid1'] == 0).sum()}")
print(f"ask1=0的数量: {(df['ask1'] == 0).sum()}")

print(f"\n=== 标签分布 ===")
for label in label_cols:
    counts = df[label].value_counts().sort_index()
    print(f"{label}: {dict(counts)}")

# 分析多个文件
print(f"\n=== 多文件异常检测 ===")
anomaly_stats = {
    'nan_count': 0,
    'inf_count': 0,
    'limit_up_count': 0,
    'limit_down_count': 0,
    'total_rows': 0
}

for i, f in enumerate(files[:20]):
    df = pd.read_parquet(os.path.join(data_dir, f))
    anomaly_stats['nan_count'] += df.isna().sum().sum()
    anomaly_stats['inf_count'] += np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    anomaly_stats['limit_up_count'] += (df['ask1'] == 0).sum()
    anomaly_stats['limit_down_count'] += (df['bid1'] == 0).sum()
    anomaly_stats['total_rows'] += len(df)

print(f"分析了前20个文件:")
print(f"总行数: {anomaly_stats['total_rows']}")
print(f"NaN总数: {anomaly_stats['nan_count']}")
print(f"Inf总数: {anomaly_stats['inf_count']}")
print(f"涨停(ask1=0)次数: {anomaly_stats['limit_up_count']}")
print(f"跌停(bid1=0)次数: {anomaly_stats['limit_down_count']}")
