$ErrorActionPreference = 'Stop'
$PACKAGE_DIR = 'train_package_complete'
$OUTPUT_ZIP = 'train_package_complete.zip'

Write-Host '========================================'
Write-Host 'Packing complete training package'
Write-Host '========================================'

if (Test-Path $PACKAGE_DIR) { Remove-Item -Recurse -Force $PACKAGE_DIR }
New-Item -ItemType Directory -Path $PACKAGE_DIR | Out-Null

# Create directory structure
New-Item -ItemType Directory -Path "$PACKAGE_DIR\src" | Out-Null
New-Item -ItemType Directory -Path "$PACKAGE_DIR\configs" | Out-Null
New-Item -ItemType Directory -Path "$PACKAGE_DIR\scripts" | Out-Null

Write-Host 'Copying core code files...'
Copy-Item 'src\train_v100_optimized.py' "$PACKAGE_DIR\src\train.py"
Copy-Item 'src\model.py' "$PACKAGE_DIR\src\"
Copy-Item 'src\losses.py' "$PACKAGE_DIR\src\"
Copy-Item 'src\Predictor.py' "$PACKAGE_DIR\src\"
Copy-Item 'src\config.json' "$PACKAGE_DIR\src\"
Copy-Item 'src\requirements.txt' "$PACKAGE_DIR\src\"

Write-Host 'Copying config files...'
Copy-Item 'src\config.json' "$PACKAGE_DIR\configs\config.json"
Copy-Item 'src\requirements.txt' "$PACKAGE_DIR\configs\requirements.txt"

Write-Host 'Creating launch scripts...'
$runTrain = @'
#!/bin/bash
echo "========================================"
echo "T-KAN Pro V100 Optimized Training"
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
'@
[System.IO.File]::WriteAllText("$PACKAGE_DIR\scripts\run_train.sh", $runTrain, [System.Text.Encoding]::UTF8)

$runTest = @'
#!/bin/bash
echo "========================================"
echo "Quick Test Mode"
echo "========================================"

cd /root/src
python train.py --data_dir /root/2026train_set/2026train_set --output_dir /root/submission --cache_dir /root/cache --batch_size 128 --epochs 2 --max_files 50 --accumulation_steps 2 --num_workers 2 --use_cache --force_reload

echo "Quick test completed!"
'@
[System.IO.File]::WriteAllText("$PACKAGE_DIR\scripts\run_test.sh", $runTest, [System.Text.Encoding]::UTF8)

$setupEnv = @'
#!/bin/bash
echo "========================================"
echo "Setup Environment"
echo "========================================"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy pyarrow tqdm

echo "Environment setup completed!"
'@
[System.IO.File]::WriteAllText("$PACKAGE_DIR\scripts\setup_env.sh", $setupEnv, [System.Text.Encoding]::UTF8)

Write-Host 'Creating README...'
$readme = @'
# T-KAN Pro V100 Complete Training Package

## Directory Structure
```
/root/
├── src/                    # Source code
│   ├── train.py           # Main training script (V100 optimized)
│   ├── model.py           # T-KAN Pro model
│   ├── losses.py          # Profit-guided loss functions
│   ├── Predictor.py       # Evaluation predictor
│   ├── config.json        # Configuration file
│   └── requirements.txt   # Dependencies
├── configs/               # Config backup
├── scripts/               # Launch scripts
│   ├── run_train.sh       # Full training script
│   ├── run_test.sh        # Quick test script
│   └── setup_env.sh       # Environment setup
├── 2026train_set/         # Training data (upload separately)
├── submission/            # Output directory (auto-created)
└── cache/                 # Cache directory (auto-created)
```

## Quick Start

### 1. Upload to server
```bash
scp train_package_complete.zip root@your_server:/root/
```

### 2. Setup environment
```bash
cd /root
unzip train_package_complete.zip
mv train_package_complete/* .
chmod +x scripts/*.sh
./scripts/setup_env.sh
```

### 3. Prepare data
Upload training data to /root/2026train_set/2026train_set/

### 4. Run training
```bash
# Quick test (recommended first)
./scripts/run_test.sh

# Full training
./scripts/run_train.sh
```

## Optimizations
- Mixed precision training (AMP) - 2-3x speedup on V100
- CUDA optimizations (cudnn.benchmark, TF32)
- Multi-process data loading (num_workers=4)
- Gradient accumulation (accumulation=4, simulates batch=1024)
- Vectorized feature computation
- Data caching (avoids repeated IO)
- torch.compile support (PyTorch 2.0+)

## Performance
| GPU | Speed | Per Epoch | 25 Epochs |
|-----|-------|-----------|-----------|
| V100 16GB | ~3-5 it/s | ~30-60 min | ~12-25 hours |
| A100 40GB | ~5-8 it/s | ~20-40 min | ~8-17 hours |
| RTX 3090 24GB | ~4-6 it/s | ~25-50 min | ~10-21 hours |

## Training Parameters
```bash
python train.py \
    --data_dir /root/2026train_set/2026train_set \
    --output_dir /root/submission \
    --cache_dir /root/cache \
    --batch_size 256 \
    --epochs 25 \
    --lr 3e-5 \
    --accumulation_steps 4 \
    --num_workers 4 \
    --use_cache \
    --compile
```

## Notes
1. Data path: Default is /root/2026train_set/2026train_set/
2. GPU memory: batch_size=256 needs ~12-16GB, reduce if OOM
3. Cache space: First run caches data, needs ~20-30GB disk
4. config.json: Contains only raw features (46 cols), derived features computed in code

## Output Files
After training, these files are saved in /root/submission/:
- best_model.pt - Best model weights
- config.json - Evaluation config
- requirements.txt - Evaluation dependencies

## Evaluation Submission
```bash
cd /root/submission
zip submission.zip Predictor.py config.json best_model.pt requirements.txt
```
'@
[System.IO.File]::WriteAllText("$PACKAGE_DIR\README.md", $readme, [System.Text.Encoding]::UTF8)

Write-Host 'Creating INSTALL guide...'
$installGuide = @'
# Installation Guide

## Requirements
- Python 3.10+
- CUDA 11.8+
- PyTorch 2.0+
- 16GB+ GPU memory
- 50GB+ disk space

## Auto Install
```bash
./scripts/setup_env.sh
```

## Manual Install
```bash
# 1. Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 2. Install other dependencies
pip install pandas==2.0.3 numpy==1.24.3 pyarrow==12.0.1 tqdm==4.65.0
```

## Verify Installation
```bash
python -c "
import torch
import pandas
import numpy
import pyarrow
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"
```
'@
[System.IO.File]::WriteAllText("$PACKAGE_DIR\INSTALL.md", $installGuide, [System.Text.Encoding]::UTF8)

# Remove old zip
if (Test-Path $OUTPUT_ZIP) { Remove-Item -Force $OUTPUT_ZIP }

Write-Host 'Creating zip file...'
Compress-Archive -Path "$PACKAGE_DIR\*" -DestinationPath $OUTPUT_ZIP -Force

# Get file size
$zipSize = (Get-Item $OUTPUT_ZIP).Length
$zipSizeMB = [math]::Round($zipSize / 1MB, 2)

Write-Host ''
Write-Host '========================================'
Write-Host 'Packaging completed!'
Write-Host "Output: $OUTPUT_ZIP"
Write-Host "Size: $zipSizeMB MB"
Write-Host '========================================'
Write-Host ''
Write-Host 'Directory structure:'
Get-ChildItem $PACKAGE_DIR -Recurse | ForEach-Object {
    $indent = '  ' * ($_.FullName.Split('\').Count - $PACKAGE_DIR.Split('\').Count - 1)
    if ($_.PSIsContainer) {
        Write-Host "$indent$($_.Name)/"
    } else {
        Write-Host "$indent$($_.Name)"
    }
}
