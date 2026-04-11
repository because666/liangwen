#!/bin/bash
echo "========================================"
echo "T-KAN Pro V100 Training"
echo "========================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

mkdir -p /root/submission
mkdir -p /root/cache

echo ""
echo "Starting training..."
echo ""

cd /root/src
python train.py --data_dir /root/2026train_set/2026train_set --output_dir /root/submission --cache_dir /root/cache --batch_size 256 --epochs 25 --lr 3e-5 --accumulation_steps 4 --num_workers 4 --use_cache "$@"

echo ""
echo "========================================"
echo "Training completed!"
echo "Model saved at: /root/submission/best_model.pt"
echo "========================================"