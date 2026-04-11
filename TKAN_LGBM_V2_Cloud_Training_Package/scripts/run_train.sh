#!/bin/bash
# T-KAN + LightGBM 云GPU训练启动脚本

set -e

echo "=============================================="
echo "T-KAN + LightGBM 训练脚本"
echo "=============================================="

# 设置数据路径
DATA_DIR="${DATA_DIR:-./data/2026train_set}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# 默认参数
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-3}"
MAX_FILES="${MAX_FILES:-}"

echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "学习率: $LR"
echo "隐藏维度: $HIDDEN_DIM"
echo "T-KAN层数: $NUM_LAYERS"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo ""
echo "=============================================="
echo "步骤 1/2: T-KAN 编码器预训练"
echo "=============================================="

cd src

python train_tkan.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    ${MAX_FILES:+--max_files $MAX_FILES}

echo ""
echo "=============================================="
echo "步骤 2/2: LightGBM 分类器训练"
echo "=============================================="

python train_lgbm.py \
    --data_dir $DATA_DIR \
    --encoder_path $OUTPUT_DIR/tkan_encoder.pt \
    --output_dir $OUTPUT_DIR \
    ${MAX_FILES:+--max_files $MAX_FILES}

echo ""
echo "=============================================="
echo "训练完成！"
echo "=============================================="
echo "提交文件位置: $OUTPUT_DIR/submission/"
echo ""
echo "文件列表:"
ls -lh $OUTPUT_DIR/submission/
