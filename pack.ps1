$ErrorActionPreference = 'Stop'
$PACKAGE_DIR = 'train_package'
$OUTPUT_ZIP = 'train_package.zip'

Write-Host '========================================'
Write-Host 'Packing training files...'
Write-Host '========================================'

if (Test-Path $PACKAGE_DIR) { Remove-Item -Recurse -Force $PACKAGE_DIR }
New-Item -ItemType Directory -Path $PACKAGE_DIR | Out-Null

Write-Host 'Copying files...'
Copy-Item 'src\train_v100_optimized.py' "$PACKAGE_DIR\train.py"
Copy-Item 'src\model.py' $PACKAGE_DIR
Copy-Item 'src\losses.py' $PACKAGE_DIR
Copy-Item 'src\Predictor.py' $PACKAGE_DIR
Copy-Item 'src\config.json' $PACKAGE_DIR
Copy-Item 'src\requirements.txt' $PACKAGE_DIR

Write-Host 'Creating run scripts...'
$runTrain = @'
#!/bin/bash
echo "========================================"
echo "T-KAN Pro V100 Optimized Training"
echo "========================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

mkdir -p /root/submission
mkdir -p /root/cache

echo ""
echo "Starting training..."
echo ""

python train.py --data_dir /root/2026train_set/2026train_set --output_dir /root/submission --cache_dir /root/cache --batch_size 256 --epochs 25 --lr 3e-5 --accumulation_steps 4 --num_workers 4 --use_cache "$@"

echo ""
echo "========================================"
echo "Training completed!"
echo "Model saved at: /root/submission/best_model.pt"
echo "========================================"
'@
[System.IO.File]::WriteAllText("$PACKAGE_DIR\run_train.sh", $runTrain, [System.Text.Encoding]::UTF8)

$runTest = @'
#!/bin/bash
echo "========================================"
echo "Quick Test Mode"
echo "========================================"

python train.py --data_dir /root/2026train_set/2026train_set --output_dir /root/submission --cache_dir /root/cache --batch_size 128 --epochs 2 --max_files 50 --accumulation_steps 2 --num_workers 2 --use_cache --force_reload

echo "Quick test completed!"
'@
[System.IO.File]::WriteAllText("$PACKAGE_DIR\run_test.sh", $runTest, [System.Text.Encoding]::UTF8)

Write-Host 'Creating README...'
$readme = @'
# T-KAN Pro V100 Optimized Training Package

## Files
- train.py: V100 optimized training script
- model.py: T-KAN Pro model definition
- losses.py: Profit-guided loss functions
- Predictor.py: Evaluation predictor
- config.json: Configuration file
- requirements.txt: Dependencies
- run_train.sh: Training launch script
- run_test.sh: Quick test script

## Usage

1. Install dependencies: pip install -r requirements.txt
2. Quick test: chmod +x run_test.sh && ./run_test.sh
3. Full training: chmod +x run_train.sh && ./run_train.sh
4. Custom parameters: python train.py --help

## Optimizations
- Mixed precision training (AMP) - 2-3x speedup
- CUDA optimizations (cudnn.benchmark, TF32)
- Multi-process data loading (num_workers=4)
- Gradient accumulation (simulate larger batch)
- Vectorized feature computation
- Data caching (avoid repeated IO)

## Expected Performance
- V100 16GB: ~3-5 it/s, 30-60 min per epoch
- A100 40GB: ~5-8 it/s, 20-40 min per epoch

## Notes
1. First run will cache data, needs sufficient disk space
2. If OOM, reduce batch_size or accumulation_steps
3. Model saved at /root/submission/best_model.pt
'@
[System.IO.File]::WriteAllText("$PACKAGE_DIR\README.md", $readme, [System.Text.Encoding]::UTF8)

if (Test-Path $OUTPUT_ZIP) { Remove-Item -Force $OUTPUT_ZIP }

Write-Host 'Creating zip file...'
Compress-Archive -Path "$PACKAGE_DIR\*" -DestinationPath $OUTPUT_ZIP -Force

Write-Host ''
Write-Host '========================================'
Write-Host 'Packaging completed!'
Write-Host "Output: $OUTPUT_ZIP"
Write-Host '========================================'
Write-Host ''
Write-Host 'Included files:'
Get-ChildItem $PACKAGE_DIR | ForEach-Object { Write-Host "  $($_.Name)" }
