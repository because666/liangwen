@echo off
chcp 65001 >nul
echo ========================================
echo 打包训练文件
echo ========================================

set PACKAGE_DIR=train_package
set OUTPUT_ZIP=train_package.zip

if exist %PACKAGE_DIR% rmdir /s /q %PACKAGE_DIR%
mkdir %PACKAGE_DIR%

echo 复制文件...
copy src\train_v100_optimized.py %PACKAGE_DIR%\train.py
copy src\model.py %PACKAGE_DIR%\
copy src\losses.py %PACKAGE_DIR%\
copy src\Predictor.py %PACKAGE_DIR%\
copy src\config.json %PACKAGE_DIR%\
copy src\requirements.txt %PACKAGE_DIR%\

echo 创建启动脚本...
(
echo #!/bin/bash
echo # V100优化训练启动脚本
echo.
echo echo "========================================"
echo echo "T-KAN Pro V100优化训练"
echo echo "========================================"
echo.
echo # 设置环境变量
echo export PYTHONUNBUFFERED=1
echo export OMP_NUM_THREADS=4
echo.
echo # 检查GPU
echo python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"
echo.
echo # 创建输出目录
echo mkdir -p /root/submission
echo mkdir -p /root/cache
echo.
echo # 运行训练
echo echo ""
echo echo "启动训练..."
echo echo ""
echo.
echo python train.py \
echo     --data_dir /root/2026train_set/2026train_set \
echo     --output_dir /root/submission \
echo     --cache_dir /root/cache \
echo     --batch_size 256 \
echo     --epochs 25 \
echo     --lr 3e-5 \
echo     --accumulation_steps 4 \
echo     --num_workers 4 \
echo     --use_cache \
echo     "$@"
echo.
echo echo ""
echo echo "========================================"
echo echo "训练完成！"
echo echo "模型保存在: /root/submission/best_model.pt"
echo echo "========================================"
) > %PACKAGE_DIR%\run_train.sh

echo 创建快速测试脚本...
(
echo #!/bin/bash
echo # 快速测试脚本 - 使用少量数据验证代码
echo.
echo echo "========================================"
echo echo "快速测试模式"
echo echo "========================================"
echo.
echo python train.py \
echo     --data_dir /root/2026train_set/2026train_set \
echo     --output_dir /root/submission \
echo     --cache_dir /root/cache \
echo     --batch_size 128 \
echo     --epochs 2 \
echo     --max_files 50 \
echo     --accumulation_steps 2 \
echo     --num_workers 2 \
echo     --use_cache \
echo     --force_reload
echo.
echo echo "快速测试完成！"
) > %PACKAGE_DIR%\run_test.sh

echo 创建README...
(
echo # T-KAN Pro V100优化训练包
echo.
echo ## 文件说明
echo.
echo | 文件 | 说明 |
echo |------|------|
echo | train.py | V100优化训练脚本 |
echo | model.py | T-KAN Pro模型定义 |
echo | losses.py | 收益导向损失函数 |
echo | Predictor.py | 评测预测器 |
echo | config.json | 配置文件 |
echo | requirements.txt | 依赖文件 |
echo | run_train.sh | 训练启动脚本 |
echo | run_test.sh | 快速测试脚本 |
echo.
echo ## 使用方法
echo.
echo ### 1. 安装依赖
echo ```bash
echo pip install -r requirements.txt
echo ```
echo.
echo ### 2. 快速测试（推荐先执行）
echo ```bash
echo chmod +x run_test.sh
echo ./run_test.sh
echo ```
echo.
echo ### 3. 完整训练
echo ```bash
echo chmod +x run_train.sh
echo ./run_train.sh
echo ```
echo.
echo ### 4. 自定义参数
echo ```bash
echo python train.py --help
echo python train.py --batch_size 512 --epochs 50
echo ```
echo.
echo ## 优化特性
echo.
echo - 混合精度训练（AMP）- 速度提升2-3倍
echo - CUDA优化（cudnn.benchmark, TF32）
echo - 多进程数据加载（num_workers=4）
echo - 梯度累积（模拟更大batch）
echo - 向量化特征计算
echo - 数据缓存（避免重复IO）
echo.
echo ## 预期性能
echo.
echo | GPU | 速度 | 每轮时间 |
echo |-----|------|----------|
echo | V100 16GB | ~3-5 it/s | ~30-60分钟 |
echo | A100 40GB | ~5-8 it/s | ~20-40分钟 |
echo.
echo ## 注意事项
echo.
echo 1. 首次运行会缓存数据，需要足够磁盘空间
echo 2. 如果显存不足，减小batch_size或accumulation_steps
echo 3. 训练完成后，模型保存在 /root/submission/best_model.pt
) > %PACKAGE_DIR%\README.md

echo 删除旧的zip文件...
if exist %OUTPUT_ZIP% del %OUTPUT_ZIP%

echo 创建zip文件...
powershell -Command "Compress-Archive -Path %PACKAGE_DIR%\* -DestinationPath %OUTPUT_ZIP% -Force"

echo.
echo ========================================
echo 打包完成！
echo 输出文件: %OUTPUT_ZIP%
echo ========================================
echo.
echo 包含文件:
dir /b %PACKAGE_DIR%
echo.
echo 请将 %OUTPUT_ZIP% 上传到云服务器
echo 解压后运行: chmod +x run_train.sh ^&^& ./run_train.sh
pause
