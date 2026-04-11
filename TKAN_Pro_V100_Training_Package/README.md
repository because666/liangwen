# T-KAN Pro V100 Training Package

## What's Included
- src/ - Source code (train.py, model.py, losses.py, Predictor.py, config.json, requirements.txt)
- scripts/ - Launch scripts (run_train.sh, run_test.sh, setup_env.sh)
- 2026train_set/ - Training data

## Quick Start

### 1. Upload to server
```bash
scp TKAN_Pro_V100_Training_Package.zip root@your_server:/root/
```

### 2. Setup on server
```bash
cd /root
unzip TKAN_Pro_V100_Training_Package.zip
mv TKAN_Pro_V100_Training_Package/* .
chmod +x scripts/*.sh
./scripts/setup_env.sh
```

### 3. Run training
```bash
# Quick test first
./scripts/run_test.sh

# Full training
./scripts/run_train.sh
```

## Optimizations
- Mixed precision training (AMP)
- CUDA optimizations (cudnn.benchmark, TF32)
- Multi-process data loading
- Gradient accumulation
- Data caching

## Expected Performance
- V100 16GB: ~3-5 it/s, 12-25 hours for 25 epochs
- A100 40GB: ~5-8 it/s, 8-17 hours for 25 epochs

## Output
After training, check /root/submission/ for:
- best_model.pt
- config.json
- requirements.txt