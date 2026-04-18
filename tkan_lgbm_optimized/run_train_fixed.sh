#!/bin/bash

# 安装依赖
pip install -r requirements.txt

# 训练模型（使用全量数据）
python train_tkan_classifier.py \
    --data_dir ./train_data \
    --output_dir ./output_tkan_classifier \
    --hidden_dim 256 \
    --num_layers 4 \
    --dropout 0.2 \
    --batch_size 256 \
    --epochs 50 \
    --lr 5e-4 \
    --weight_decay 0.05 \
    --sample_interval 1 \
    --use_weights \
    --patience 10

echo "训练完成！"
