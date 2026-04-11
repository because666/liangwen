#!/bin/bash
# 环境安装脚本

echo "=============================================="
echo "安装 T-KAN + LightGBM 训练环境"
echo "=============================================="

# 检查Python版本
python --version

# 安装PyTorch (CUDA版本)
echo "安装 PyTorch..."
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
echo "安装其他依赖..."
pip install pandas>=2.0.0 numpy>=1.24.0 pyarrow>=10.0.0 lightgbm>=4.0.0 tqdm

echo ""
echo "=============================================="
echo "环境安装完成！"
echo "=============================================="
echo ""
echo "验证安装:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"
python -c "import lightgbm as lgb; print(f'LightGBM: {lgb.__version__}')"
