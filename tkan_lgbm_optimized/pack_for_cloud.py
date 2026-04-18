"""
打包训练代码和数据用于云GPU训练

包含：
1. 训练脚本
2. 预测器
3. 配置文件
4. 依赖
5. 训练数据（parquet文件）

使用方法：
python pack_for_cloud.py
"""

import os
import shutil
import zipfile

def pack_for_cloud():
    print("\n" + "=" * 80)
    print("打包训练代码和数据")
    print("=" * 80)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    # 创建临时目录
    temp_dir = os.path.join(current_dir, 'cloud_train_package')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # 1. 复制训练脚本
    print("\n复制训练脚本...")
    files_to_copy = [
        ('train_tkan_classifier.py', 'train_tkan_classifier.py'),
        ('Predictor_tkan_classifier.py', 'Predictor_tkan_classifier.py'),
        ('config.json', 'config.json'),
        ('requirements.txt', 'requirements.txt'),
    ]
    
    for src_name, dst_name in files_to_copy:
        src_path = os.path.join(current_dir, src_name)
        dst_path = os.path.join(temp_dir, dst_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  复制: {src_name}")
        else:
            print(f"  警告: 文件不存在 {src_name}")
    
    # 2. 创建训练数据目录并复制数据
    print("\n复制训练数据...")
    data_src = os.path.join(project_dir, '2026train_set', '2026train_set')
    data_dst = os.path.join(temp_dir, 'train_data')
    
    if os.path.exists(data_src):
        os.makedirs(data_dst, exist_ok=True)
        
        parquet_files = [f for f in os.listdir(data_src) if f.endswith('.parquet')]
        total_size = 0
        
        for i, fname in enumerate(parquet_files):
            src_file = os.path.join(data_src, fname)
            dst_file = os.path.join(data_dst, fname)
            file_size = os.path.getsize(src_file) / (1024 * 1024)  # MB
            total_size += file_size
            shutil.copy2(src_file, dst_file)
            
            if (i + 1) % 20 == 0:
                print(f"  已复制 {i + 1}/{len(parquet_files)} 个文件, "
                      f"累计 {total_size:.1f} MB")
        
        print(f"  总计复制 {len(parquet_files)} 个数据文件, {total_size:.1f} MB")
    else:
        print(f"  警告: 数据目录不存在 {data_src}")
    
    # 3. 创建运行脚本
    print("\n创建运行脚本...")
    
    # Linux 运行脚本
    run_script_linux = """#!/bin/bash
# 安装依赖
pip install -r requirements.txt

# 训练模型（使用全量数据）
python train_tkan_classifier.py \\
    --data_dir ./train_data \\
    --output_dir ./output_tkan_classifier \\
    --hidden_dim 256 \\
    --num_layers 4 \\
    --dropout 0.2 \\
    --batch_size 256 \\
    --epochs 50 \\
    --lr 5e-4 \\
    --weight_decay 0.05 \\
    --sample_interval 1 \\
    --use_weights \\
    --patience 10

echo "训练完成！"
"""
    
    with open(os.path.join(temp_dir, 'run_train.sh'), 'w', encoding='utf-8') as f:
        f.write(run_script_linux)
    print("  创建 run_train.sh")
    
    # Windows 运行脚本
    run_script_windows = """@echo off
REM 安装依赖
pip install -r requirements.txt

REM 训练模型（使用全量数据）
python train_tkan_classifier.py ^
    --data_dir ./train_data ^
    --output_dir ./output_tkan_classifier ^
    --hidden_dim 256 ^
    --num_layers 4 ^
    --dropout 0.2 ^
    --batch_size 256 ^
    --epochs 50 ^
    --lr 5e-4 ^
    --weight_decay 0.05 ^
    --sample_interval 1 ^
    --use_weights ^
    --patience 10

echo 训练完成！
"""
    
    with open(os.path.join(temp_dir, 'run_train.bat'), 'w', encoding='utf-8') as f:
        f.write(run_script_windows)
    print("  创建 run_train.bat")
    
    # 4. 打包
    print("\n打包压缩...")
    zip_path = os.path.join(current_dir, 'cloud_train_package.zip')
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    total_files = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zf.write(file_path, arcname)
                total_files += 1
                if total_files % 50 == 0:
                    print(f"  已打包 {total_files} 个文件...")
    
    # 5. 清理临时目录
    shutil.rmtree(temp_dir)
    
    # 6. 显示结果
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)  # MB
    print(f"\n打包完成！")
    print(f"  文件: {zip_path}")
    print(f"  大小: {zip_size:.1f} MB")
    print(f"  文件数: {total_files}")
    
    print("\n" + "=" * 80)
    print("上传说明")
    print("=" * 80)
    print("1. 上传 cloud_train_package.zip 到云GPU")
    print("2. 解压: unzip cloud_train_package.zip")
    print("3. 运行: bash run_train.sh")
    print("4. 训练完成后，下载 output_tkan_classifier/best_model.pt")
    print("=" * 80)


if __name__ == '__main__':
    pack_for_cloud()
