"""
提交准备脚本 - 最终版

将训练好的模型打包成符合比赛要求的格式

比赛提交要求：
- submission.zip
├── config.json          # 配置文件
├── Predictor.py         # 预测类
├── best_model.pt        # 模型权重
└── requirements.txt     # 依赖文件
"""

import os
import sys
import json
import shutil
import zipfile
import argparse


def prepare_submission(
    model_dir: str,
    output_dir: str = './',
    output_name: str = "submission"
):
    """
    准备提交文件
    
    Args:
        model_dir: 模型目录（训练输出目录）
        output_dir: 输出目录
        output_name: 输出文件名
    """
    print("="*60)
    print("准备提交文件")
    print("="*60)
    
    submission_dir = os.path.join(output_dir, output_name)
    os.makedirs(submission_dir, exist_ok=True)
    
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}
    
    model_path = os.path.join(model_dir, "best_model.pt")
    if os.path.exists(model_path):
        dest_path = os.path.join(submission_dir, "best_model.pt")
        shutil.copy2(model_path, dest_path)
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"复制模型: {model_path}")
        print(f"  大小: {size_mb:.2f} MB")
    else:
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    predictor_candidates = [
        os.path.join(current_dir, "Predictor_profit.py"),
        os.path.join(current_dir, "Predictor_final.py"),
        os.path.join(current_dir, "Predictor_advanced.py"),
        os.path.join(current_dir, "Predictor_v2.py"),
    ]
    
    predictor_src = None
    for candidate in predictor_candidates:
        if os.path.exists(candidate):
            predictor_src = candidate
            break
    
    if predictor_src:
        dest_path = os.path.join(submission_dir, "Predictor.py")
        shutil.copy2(predictor_src, dest_path)
        print(f"复制预测器: {predictor_src}")
    else:
        print(f"警告: 预测器文件不存在")
    
    thresholds = config.get("thresholds", {str(i): 0.5 for i in range(5)})
    feature_cols = config.get("feature", [])
    use_tta = config.get("use_tta", True)
    tta_rounds = config.get("tta_rounds", 5)
    
    submission_config = {
        "python_version": "3.10",
        "batch": config.get("batch", 256),
        "feature": feature_cols,
        "label": ["label_5", "label_10", "label_20", "label_40", "label_60"],
        "thresholds": thresholds,
        "use_tta": use_tta,
        "tta_rounds": tta_rounds if use_tta else 0
    }
    
    config_dest = os.path.join(submission_dir, "config.json")
    with open(config_dest, "w", encoding="utf-8") as f:
        json.dump(submission_config, f, indent=2, ensure_ascii=False)
    print(f"创建配置文件: {config_dest}")
    
    requirements_content = """torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0
tqdm>=4.65.0
scipy>=1.10.0
"""
    
    requirements_dest = os.path.join(submission_dir, "requirements.txt")
    with open(requirements_dest, "w", encoding="utf-8") as f:
        f.write(requirements_content)
    print(f"创建依赖文件: {requirements_dest}")
    
    zip_path = os.path.join(output_dir, f"{output_name}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(submission_dir):
            file_path = os.path.join(submission_dir, file)
            zipf.write(file_path, file)
    
    print(f"\n创建提交包: {zip_path}")
    
    print("\n提交文件内容:")
    print("-"*40)
    total_size = 0
    for file in sorted(os.listdir(submission_dir)):
        file_path = os.path.join(submission_dir, file)
        size = os.path.getsize(file_path) / 1024 / 1024
        total_size += size
        print(f"  {file}: {size:.2f} MB")
    print("-"*40)
    print(f"  总大小: {total_size:.2f} MB")
    
    print("\n" + "="*60)
    print("提交准备完成!")
    print(f"提交目录: {submission_dir}")
    print(f"提交包: {zip_path}")
    print("="*60)
    
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="准备提交文件")
    parser.add_argument('--model_dir', type=str, default='./output_final', help='模型目录')
    parser.add_argument('--output_dir', type=str, default='./', help='输出目录')
    parser.add_argument('--output_name', type=str, default='submission', help='输出文件名')
    
    args = parser.parse_args()
    
    prepare_submission(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        output_name=args.output_name
    )


if __name__ == "__main__":
    main()
