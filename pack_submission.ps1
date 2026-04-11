$ErrorActionPreference = 'Stop'
$EXPORT_DIR = 'd:\量化\良文杯\导出'
$OUTPUT_ZIP = 'd:\量化\良文杯\submission.zip'

Write-Host '========================================'
Write-Host '打包评测提交文件'
Write-Host '========================================'

# 检查必要文件
$requiredFiles = @('Predictor.py', 'config.json', 'best_model.pt', 'requirements.txt')

Write-Host ''
Write-Host '检查文件...'
foreach ($file in $requiredFiles) {
    $path = Join-Path $EXPORT_DIR $file
    if (Test-Path $path) {
        $size = (Get-Item $path).Length
        $sizeMB = [math]::Round($size / 1MB, 2)
        Write-Host "  OK $file ($sizeMB MB)"
    } else {
        Write-Host "  MISSING $file" -ForegroundColor Red
        exit 1
    }
}

# 删除旧的zip
if (Test-Path $OUTPUT_ZIP) {
    Remove-Item -Force $OUTPUT_ZIP
    Write-Host ''
    Write-Host '删除旧的 submission.zip'
}

# 创建zip
Write-Host ''
Write-Host '创建 submission.zip...'
Compress-Archive -Path "$EXPORT_DIR\*" -DestinationPath $OUTPUT_ZIP -Force

# 验证
$zipSize = (Get-Item $OUTPUT_ZIP).Length
$zipSizeMB = [math]::Round($zipSize / 1MB, 2)

Write-Host ''
Write-Host '========================================'
Write-Host '打包完成'
Write-Host "文件: $OUTPUT_ZIP"
Write-Host "大小: $zipSizeMB MB"
Write-Host '========================================'
