import torch
import numpy as np

# 加载模型
checkpoint = torch.load('model_seed42.pt', map_location='cpu', weights_only=False)
print('模型键:', list(checkpoint.keys())[:10])

# 检查是否有meta信息
if 'meta' in checkpoint:
    print('元数据:', checkpoint['meta'])
else:
    print('没有元数据')

# 创建测试输入
dummy_input = torch.randn(1, 100, 154)
print(f'测试输入形状: {dummy_input.shape}')

# 尝试加载模型并前向传播
import sys
sys.path.insert(0, 'd:\\量化\\良文杯\\导出')
from Predictor import HFTPredictor

model = HFTPredictor(num_features=154, num_classes_per_head=[3, 3, 3, 3, 3], hidden_dim=128)
model.load_state_dict(checkpoint, strict=False)
model.eval()

with torch.no_grad():
    outputs = model(dummy_input)
    print(f'输出数量: {len(outputs)}')
    for i, o in enumerate(outputs):
        print(f'  输出{i}形状: {o.shape}')
