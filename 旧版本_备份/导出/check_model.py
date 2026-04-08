import torch
import json

# 检查模型文件
checkpoint = torch.load('model_seed42.pt', map_location='cpu', weights_only=False)
print('模型文件内容:')
print('  键:', list(checkpoint.keys()))
if 'meta' in checkpoint:
    print('  元数据:', checkpoint['meta'])
if 'feature_cols' in checkpoint:
    print('  特征数:', len(checkpoint['feature_cols']))
if 'hidden_dim' in checkpoint:
    print('  hidden_dim:', checkpoint['hidden_dim'])
if 'num_features' in checkpoint:
    print('  num_features:', checkpoint['num_features'])
