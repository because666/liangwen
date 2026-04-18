"""
打包端到端 T-KAN 分类器

使用方法：
python pack_tkan_classifier.py
"""

import os
import shutil
import zipfile

def pack_tkan_classifier():
    print("\n打包端到端 T-KAN 分类器")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'submissions')
    submission_dir = os.path.join(output_dir, 'submission_tkan_classifier')
    
    os.makedirs(submission_dir, exist_ok=True)
    
    files_to_copy = [
        ('config.json', 'config.json'),
        ('requirements.txt', 'requirements.txt'),
        ('Predictor_tkan_classifier.py', 'Predictor.py'),
    ]
    
    for src_name, dst_name in files_to_copy:
        src_path = os.path.join(current_dir, src_name)
        dst_path = os.path.join(submission_dir, dst_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  复制: {src_name} -> {dst_name}")
        else:
            print(f"  警告: 文件不存在 {src_name}")
    
    # 复制模型文件
    model_src = os.path.join(current_dir, 'output_tkan_classifier', 'best_model.pt')
    if os.path.exists(model_src):
        shutil.copy2(model_src, os.path.join(submission_dir, 'best_model.pt'))
        print(f"  复制: best_model.pt")
    else:
        print(f"  警告: 模型文件不存在 {model_src}")
        print(f"  请先运行 train_tkan_classifier.py 训练模型")
        return
    
    # 打包
    zip_path = os.path.join(output_dir, 'submission_tkan_classifier.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(submission_dir):
            fpath = os.path.join(submission_dir, fname)
            zf.write(fpath, fname)
    
    print(f"\n打包完成: {zip_path}")


if __name__ == '__main__':
    pack_tkan_classifier()
