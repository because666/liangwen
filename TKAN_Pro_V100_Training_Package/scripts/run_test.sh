#!/bin/bash
echo "========================================"
echo "Quick Test Mode"
echo "========================================"

cd /root/src
python train.py --data_dir /root/2026train_set/2026train_set --output_dir /root/submission --cache_dir /root/cache --batch_size 128 --epochs 2 --max_files 50 --accumulation_steps 2 --num_workers 2 --use_cache --force_reload

echo "Quick test completed!"