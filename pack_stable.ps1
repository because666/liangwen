$ErrorActionPreference = 'Stop'
$OUTPUT_ZIP = 'train_stable_fix.zip'

Write-Host '========================================'
Write-Host 'Packing stable training script'
Write-Host '========================================'

if (Test-Path $OUTPUT_ZIP) { Remove-Item -Force $OUTPUT_ZIP }

# 只包含必要的文件
$files = @(
    'src\train_stable.py',
    'src\model.py',
    'src\losses.py',
    'src\Predictor.py',
    'src\config.json',
    'src\requirements.txt',
    'src\test_train_stable.py'
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "  Adding: $file"
    } else {
        Write-Host "  Missing: $file" -ForegroundColor Yellow
    }
}

Compress-Archive -Path $files -DestinationPath $OUTPUT_ZIP -Force

$zipSize = (Get-Item $OUTPUT_ZIP).Length / 1KB
Write-Host ''
Write-Host "Created: $OUTPUT_ZIP ($([math]::Round($zipSize, 2)) KB)"
