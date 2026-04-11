#!/bin/bash
echo "========================================"
echo "Setup Environment"
echo "========================================"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy pyarrow tqdm

echo "Environment setup completed!"