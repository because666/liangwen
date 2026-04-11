$ErrorActionPreference = 'Stop'
$PACKAGE_DIR = 'TKAN_Pro_V100_Training_Package'
$OUTPUT_ZIP = 'TKAN_Pro_V100_Training_Package.zip'

Write-Host '========================================'
Write-Host 'Creating Final Training Package'
Write-Host '========================================'

if (Test-Path $PACKAGE_DIR) { Remove-Item -Recurse -Force $PACKAGE_DIR }
New-Item -ItemType Directory -Path $PACKAGE_DIR | Out-Null

# Create directory structure
New-Item -ItemType Directory -Path "$PACKAGE_DIR\src" | Out-Null
New-Item -ItemType Directory -Path "$PACKAGE_DIR\scripts" | Out-Null
New-Item -ItemType Directory -Path "$PACKAGE_DIR\2026train_set" | Out-Null

Write-Host 'Copying source code...'
Copy-Item 'src\train_v100_optimized.py' "$PACKAGE_DIR\src\train.py"
Copy-Item 'src\model.py' "$PACKAGE_DIR\src\"
Copy-Item 'src\losses.py' "$PACKAGE_DIR\src\"
Copy-Item 'src\Predictor.py' "$PACKAGE_DIR\src\"
Copy-Item 'src\config.json' "$PACKAGE_DIR\src\"
Copy-Item 'src\requirements.txt' "$PACKAGE_DIR\src\"

Write-Host 'Extracting training data...'
if (Test-Path '2026train_set.zip') {
    Expand-Archive -Path '2026train_set.zip' -DestinationPath "$PACKAGE_DIR\2026train_set" -Force
    Write-Host 'Data extracted successfully'
} else {
    Write-Host 'WARNING: 2026train_set.zip not found!' -ForegroundColor Red
}

Write-Host 'Creating launch scripts...'
$runTrain = @'
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

$readme = @'
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
'@
[System.IO.File]::WriteAllText("$PACKAGE_DIR\README.md", $readme, [System.Text.Encoding]::UTF8)

if (Test-Path $OUTPUT_ZIP) { Remove-Item -Force $OUTPUT_ZIP }

Write-Host 'Creating zip file (this may take a while)...'
Compress-Archive -Path "$PACKAGE_DIR\*" -DestinationPath $OUTPUT_ZIP -Force

$zipSize = (Get-Item $OUTPUT_ZIP).Length
$zipSizeMB = [math]::Round($zipSize / 1MB, 2)

Write-Host ''
Write-Host '========================================'
Write-Host 'Package created successfully!'
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
