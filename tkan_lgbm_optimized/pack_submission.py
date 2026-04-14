"""
提交打包脚本

将训练好的模型打包成评测平台要求的格式：
submission.zip
├── config.json
├── Predictor.py
├── model.py
├── tkan_encoder.pt (或 best_model.pt)
└── requirements.txt

使用方法：
python pack_submission.py --mode [dynamic|regression|two_stage|cost_sensitive]
"""

import os
import sys
import argparse
import zipfile
import shutil
import tempfile


def pack_dynamic_threshold_mode(output_dir: str):
    """打包动态阈值 + 后处理过滤模式"""
    print("\n打包模式: 动态阈值 + 后处理过滤")
    
    submission_dir = os.path.join(output_dir, 'submission_dynamic')
    os.makedirs(submission_dir, exist_ok=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    files_to_copy = [
        ('config.json', 'config.json'),
        ('model.py', 'model.py'),
        ('requirements.txt', 'requirements.txt'),
        ('Predictor_optimized.py', 'Predictor.py'),
    ]
    
    for src_name, dst_name in files_to_copy:
        src_path = os.path.join(current_dir, src_name)
        dst_path = os.path.join(submission_dir, dst_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  复制: {src_name} -> {dst_name}")
        else:
            print(f"  警告: 文件不存在 {src_name}")
    
    encoder_src = os.path.join(current_dir, 'output_optimized', 'submission', 'tkan_encoder.pt')
    if os.path.exists(encoder_src):
        shutil.copy2(encoder_src, os.path.join(submission_dir, 'best_model.pt'))
        print(f"  复制: tkan_encoder.pt -> best_model.pt")
    else:
        print(f"  警告: 模型文件不存在 {encoder_src}")
    
    zip_path = os.path.join(output_dir, 'submission_dynamic.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(submission_dir):
            fpath = os.path.join(submission_dir, fname)
            zf.write(fpath, fname)
    
    print(f"\n打包完成: {zip_path}")
    return zip_path


def pack_regression_mode(output_dir: str):
    """打包回归预测模式"""
    print("\n打包模式: 回归预测")
    
    submission_dir = os.path.join(output_dir, 'submission_regression')
    os.makedirs(submission_dir, exist_ok=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    files_to_copy = [
        ('config.json', 'config.json'),
        ('model.py', 'model.py'),
        ('requirements.txt', 'requirements.txt'),
        ('Predictor_regression.py', 'Predictor.py'),
    ]
    
    for src_name, dst_name in files_to_copy:
        src_path = os.path.join(current_dir, src_name)
        dst_path = os.path.join(submission_dir, dst_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  复制: {src_name} -> {dst_name}")
        else:
            print(f"  警告: 文件不存在 {src_name}")
    
    encoder_src = os.path.join(current_dir, 'output_optimized', 'regression', 'tkan_encoder.pt')
    if os.path.exists(encoder_src):
        shutil.copy2(encoder_src, os.path.join(submission_dir, 'best_model.pt'))
        print(f"  复制: tkan_encoder.pt -> best_model.pt")
    else:
        print(f"  警告: 模型文件不存在 {encoder_src}")
    
    zip_path = os.path.join(output_dir, 'submission_regression.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(submission_dir):
            fpath = os.path.join(submission_dir, fname)
            zf.write(fpath, fname)
    
    print(f"\n打包完成: {zip_path}")
    return zip_path


def pack_two_stage_mode(output_dir: str):
    """打包两阶段模型模式"""
    print("\n打包模式: 两阶段模型")
    
    submission_dir = os.path.join(output_dir, 'submission_two_stage')
    os.makedirs(submission_dir, exist_ok=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    files_to_copy = [
        ('config.json', 'config.json'),
        ('model.py', 'model.py'),
        ('requirements.txt', 'requirements.txt'),
        ('Predictor_two_stage.py', 'Predictor.py'),
    ]
    
    for src_name, dst_name in files_to_copy:
        src_path = os.path.join(current_dir, src_name)
        dst_path = os.path.join(submission_dir, dst_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  复制: {src_name} -> {dst_name}")
        else:
            print(f"  警告: 文件不存在 {src_name}")
    
    encoder_src = os.path.join(current_dir, 'output_optimized', 'two_stage', 'tkan_encoder.pt')
    if os.path.exists(encoder_src):
        shutil.copy2(encoder_src, os.path.join(submission_dir, 'best_model.pt'))
        print(f"  复制: tkan_encoder.pt -> best_model.pt")
    else:
        print(f"  警告: 模型文件不存在 {encoder_src}")
    
    zip_path = os.path.join(output_dir, 'submission_two_stage.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(submission_dir):
            fpath = os.path.join(submission_dir, fname)
            zf.write(fpath, fname)
    
    print(f"\n打包完成: {zip_path}")
    return zip_path


def pack_cost_sensitive_mode(output_dir: str):
    """打包代价敏感学习模式"""
    print("\n打包模式: 代价敏感学习")
    
    submission_dir = os.path.join(output_dir, 'submission_cost_sensitive')
    os.makedirs(submission_dir, exist_ok=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    files_to_copy = [
        ('config.json', 'config.json'),
        ('model.py', 'model.py'),
        ('requirements.txt', 'requirements.txt'),
        ('Predictor_cost_sensitive.py', 'Predictor.py'),
    ]
    
    for src_name, dst_name in files_to_copy:
        src_path = os.path.join(current_dir, src_name)
        dst_path = os.path.join(submission_dir, dst_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  复制: {src_name} -> {dst_name}")
        else:
            print(f"  警告: 文件不存在 {src_name}")
    
    encoder_src = os.path.join(current_dir, 'output_optimized', 'cost_sensitive', 'tkan_encoder.pt')
    if os.path.exists(encoder_src):
        shutil.copy2(encoder_src, os.path.join(submission_dir, 'best_model.pt'))
        print(f"  复制: tkan_encoder.pt -> best_model.pt")
    else:
        print(f"  警告: 模型文件不存在 {encoder_src}")
    
    zip_path = os.path.join(output_dir, 'submission_cost_sensitive.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(submission_dir):
            fpath = os.path.join(submission_dir, fname)
            zf.write(fpath, fname)
    
    print(f"\n打包完成: {zip_path}")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description='提交打包脚本')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['dynamic', 'regression', 'two_stage', 'cost_sensitive', 'all'],
                       help='打包模式')
    parser.add_argument('--output_dir', type=str, default='submissions',
                       help='输出目录')
    
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("提交打包脚本")
    print("=" * 80)
    
    if args.mode == 'dynamic' or args.mode == 'all':
        pack_dynamic_threshold_mode(output_dir)
    
    if args.mode == 'regression' or args.mode == 'all':
        pack_regression_mode(output_dir)
    
    if args.mode == 'two_stage' or args.mode == 'all':
        pack_two_stage_mode(output_dir)
    
    if args.mode == 'cost_sensitive' or args.mode == 'all':
        pack_cost_sensitive_mode(output_dir)
    
    print("\n" + "=" * 80)
    print("打包完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
