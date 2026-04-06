#!/bin/bash
# 高频量化预测模型 - 一键训练脚本
# 适配 NVIDIA A10 GPU (24GB显存)

set -e

echo "=========================================="
echo "高频量化预测模型 - 云GPU训练"
echo "=========================================="

# 检查GPU
echo "检查GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

# 创建输出目录
mkdir -p output_final

echo ""
echo "开始训练..."
echo "=========================================="

# 运行训练
python train_cloud.py \
    --data_dir ./2026train_set/2026train_set \
    --output_dir ./output_final \
    --epochs 60 \
    --batch_size 256 \
    --lr 2e-4 \
    --hidden_dim 128 \
    --num_blocks 3 \
    --dropout 0.2 \
    --max_files 1200 \
    --stride 3 \
    --use_amp \
    --augment \
    --use_tta \
    --tta_rounds 5 \
    --label_smoothing 0.1 \
    --seed 42

echo ""
echo "=========================================="
echo "训练完成！准备提交文件..."
echo "=========================================="

# 准备提交文件
python prepare_submission.py --model_dir ./output_final --output_dir ./

echo ""
echo "=========================================="
echo "全部完成！"
echo "提交文件: submission.zip"
echo "=========================================="
