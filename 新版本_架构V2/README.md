# T-KAN+OFI 高频方向预测模型

## 项目结构

```
新版本_架构V2/
├── models/                     # 模型定义
│   ├── tkan_encoder.py        # T-KAN编码器
│   └── model.py               # 完整模型
├── training/                   # 训练相关
│   ├── losses.py              # 损失函数
│   ├── train.py               # 训练脚本
│   └── threshold_search.py    # 阈值搜索
├── data_processing/            # 数据处理
│   ├── ofi_features.py        # OFI特征计算
│   └── data_loader.py         # 数据加载器
├── submission/                 # 提交文件
│   ├── Predictor.py           # 预测类
│   ├── config.json            # 配置文件
│   ├── requirements.txt       # 依赖
│   └── best_model.pt          # 模型权重（训练后生成）
└── README.md
```

## 训练流程

### 1. 安装依赖

```bash
pip install torch pandas numpy pyarrow tqdm
```

### 2. 开始训练

```bash
cd training
python train.py --data_dir "d:\量化\良文杯\2026train_set\2026train_set" --output_dir "..\submission"
```

### 3. 阈值搜索（可选）

```bash
python threshold_search.py --model_path "..\submission\best_model.pt" --data_dir "d:\量化\良文杯\2026train_set\2026train_set"
```

### 4. 打包提交

```powershell
cd ..\submission
Compress-Archive -Path Predictor.py,model.py,config.json,best_model.pt,best_thresholds.json,requirements.txt -DestinationPath submission.zip -Force
```

## 关键特性

1. **T-KAN编码器**：使用B样条学习非线性时序特征
2. **OFI特征**：订单流不平衡指标，捕捉买卖压力
3. **收益感知损失**：直接优化交易收益
4. **出手门机制**：自动学习何时出手

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hidden_dim | 256 | 隐藏维度 |
| output_dim | 128 | 输出维度 |
| num_layers | 2 | 编码器层数 |
| grid_size | 8 | B样条网格数 |
| spline_order | 3 | B样条阶数 |
| batch_size | 256 | 批大小 |
| stage1_epochs | 50 | Stage 1训练轮数 |
| stage2_epochs | 20 | Stage 2训练轮数 |
| lr | 1e-3 | 学习率 |
| gamma | 0.03 | 出手惩罚系数 |
