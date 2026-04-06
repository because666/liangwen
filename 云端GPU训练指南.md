# 云端GPU训练指南 - 阿里云PAI

## 概述

本文档介绍如何使用阿里云PAI平台的免费GPU资源训练DeepLOB模型。

## 阿里云PAI平台简介

PAI（Platform of Artificial Intelligence）是阿里云机器学习平台，提供：
- **免费GPU资源**：新用户可申请免费试用
- **多种GPU类型**：V100、A10、A100等
- **分布式训练**：支持多卡训练
- **Notebook环境**：交互式开发

## 准备工作

### 1. 注册阿里云账号

1. 访问 [阿里云官网](https://www.aliyun.com/)
2. 注册账号并完成实名认证
3. 申请PAI免费试用额度

### 2. 创建PAI工作空间

1. 登录 [PAI控制台](https://pai.console.aliyun.com/)
2. 点击"创建工作空间"
3. 选择地域（推荐：华东1/上海）
4. 配置资源组（选择GPU资源）

### 3. 上传训练数据到OSS

```bash
# 安装ossutil
wget http://gosspublic.alicdn.com/ossutil/1.7.14/ossutil64
chmod 755 ossutil64

# 配置ossutil
./ossutil64 config

# 创建bucket（如果还没有）
./ossutil64 mb oss://your-bucket-name

# 上传训练数据
./ossutil64 cp -r ./2026train_set/ oss://your-bucket-name/data/
```

## 云端训练步骤

### 方法一：使用PAI-DSW Notebook（推荐）

#### 1. 创建DSW实例

1. 进入PAI控制台 → 模型开发与训练 → 交互式建模（DSW）
2. 点击"创建实例"
3. 配置：
   - **实例名称**：deeplob-training
   - **资源组**：选择GPU资源组
   - **实例规格**：
     - 入门级：ecs.gn6v-c8g1.2xlarge（V100, 8GB显存）
     - 高级：ecs.gn7i-c8g1.2xlarge（A10, 24GB显存）
   - **镜像**：pytorch:2.1.0-gpu-py310-cu121-ubuntu22.04
   - **存储**：100GB ESSD

4. 点击"创建"，等待实例启动

#### 2. 上传代码

在DSW Notebook中：

```bash
# 克隆代码仓库（如果使用git）
git clone https://github.com/your-repo/deeplob.git
cd deeplob

# 或者手动上传文件
# 点击左侧"上传"按钮，上传以下文件：
# - model.py
# - Predictor.py
# - train_cloud.py
# - config.json
# - requirements_cloud.txt
```

#### 3. 安装依赖

```bash
# 创建虚拟环境（可选）
conda create -n deeplob python=3.10
conda activate deeplob

# 安装依赖
pip install -r requirements_cloud.txt
```

#### 4. 下载训练数据

```bash
# 安装ossutil
wget http://gosspublic.alicdn.com/ossutil/1.7.14/ossutil64
chmod 755 ossutil64
./ossutil64 config

# 下载数据
mkdir -p data
./ossutil64 cp -r oss://your-bucket-name/data/ ./data/
```

#### 5. 开始训练

```bash
# 单GPU训练
python train_cloud.py \
    --data_dir ./data \
    --output_dir ./output \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3 \
    --model_type enhanced

# 使用更大的batch_size（如果GPU显存足够）
python train_cloud.py \
    --data_dir ./data \
    --output_dir ./output \
    --epochs 100 \
    --batch_size 512 \
    --lr 1e-3 \
    --model_type enhanced
```

#### 6. 监控训练

```bash
# 使用tensorboard
pip install tensorboard
tensorboard --logdir ./output/logs --port 6006

# 在DSW中打开6006端口，即可在浏览器中查看
```

#### 7. 保存和下载模型

```bash
# 训练完成后，模型保存在 ./output/best_model.pt

# 上传到OSS
./ossutil64 cp ./output/best_model.pt oss://your-bucket-name/models/

# 或者下载到本地
# 在DSW界面点击文件下载
```

### 方法二：使用PAI-DLC分布式训练

适合大规模训练，支持多机多卡。

#### 1. 准备训练脚本

创建 `dlc_train.py`：

```python
import os

# DLC会自动设置这些环境变量
# RANK, WORLD_SIZE, LOCAL_RANK

if __name__ == "__main__":
    os.system("""
    python train_cloud.py \
        --data_url oss://your-bucket-name/data/ \
        --output_dir ./output \
        --upload_url oss://your-bucket-name/models/ \
        --epochs 100 \
        --batch_size 256
    """)
```

#### 2. 创建DLC任务

1. 进入PAI控制台 → 模型开发与训练 → 分布式训练（DLC）
2. 点击"创建任务"
3. 配置：
   - **任务名称**：deeplob-distributed
   - **任务类型**：PyTorch
   - **资源组**：选择GPU资源组
   - **实例规格**：
     - Worker：2个 ecs.gn7i-c8g1.2xlarge（A10, 24GB显存）
   - **镜像**：pytorch:2.1.0-gpu-py310-cu121-ubuntu22.04
   - **启动命令**：
     ```bash
     python dlc_train.py
     ```
   - **数据配置**：
     - 数据源：OSS
     - 路径：oss://your-bucket-name/data/
     - 挂载到：/data

4. 点击"提交"，开始训练

#### 3. 查看任务日志

在DLC任务列表中，点击任务名称查看：
- 训练日志
- 资源监控
- 模型输出

## 性能优化建议

### 1. 选择合适的GPU

| GPU型号 | 显存 | 适用场景 | 推荐batch_size |
|---------|------|----------|----------------|
| V100 | 16GB | 入门级 | 128-256 |
| A10 | 24GB | 中级 | 256-512 |
| A100 | 40GB | 高级 | 512-1024 |

### 2. 数据加载优化

```python
# 在train_cloud.py中调整
DataLoader(
    dataset,
    batch_size=256,
    num_workers=4,  # 根据CPU核心数调整
    pin_memory=True,  # GPU训练时启用
    prefetch_factor=2,  # 预加载批次
)
```

### 3. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for features, labels in dataloader:
    with autocast():
        outputs = model(features)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 4. 梯度累积

```python
# 模拟更大的batch_size
accumulation_steps = 4

for i, (features, labels) in enumerate(dataloader):
    outputs = model(features)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 常见问题

### Q1: 显存不足（OOM）

**解决方案：**
1. 减小batch_size： `--batch_size 128`
2. 使用轻量版模型： `--model_type lite`
3. 启用梯度检查点：
   ```python
   from torch.utils.checkpoint import checkpoint
   ```

### Q2: 训练速度慢

**解决方案：**
1. 使用更大的batch_size
2. 启用混合精度训练
3. 增加num_workers
4. 使用更快的存储（ESSD）

### Q3: 数据下载慢

**解决方案：**
1. 使用PAI内置的数据集功能
2. 将数据打包成tar.gz格式
3. 使用ossutil的并发下载：
   ```bash
   ./ossutil64 cp -r --jobs 16 oss://bucket/data/ ./data/
   ```

### Q4: 任务被中断

**解决方案：**
1. 启用断点续训：
   ```bash
   python train_cloud.py --resume ./output/checkpoint_epoch_10.pt
   ```
2. 定期保存checkpoint
3. 使用DLC的自动保存功能

## 成本估算

### 免费额度

- 新用户：500元免费额度
- 学生认证：额外1000元

### GPU实例价格（按量付费）

| 实例规格 | GPU | 价格/小时 |
|---------|-----|----------|
| ecs.gn6v-c8g1.2xlarge | V100 | ~12元 |
| ecs.gn7i-c8g1.2xlarge | A10 | ~8元 |
| ecs.gn7e-c16g1.4xlarge | A100 | ~35元 |

### 训练成本估算

- 50 epoch训练（V100）：约2-3小时，成本24-36元
- 100 epoch训练（A10）：约4-5小时，成本32-40元

## 最佳实践

1. **先本地测试**：在本地用少量数据测试代码
2. **逐步扩大**：先用小数据集验证，再全量训练
3. **监控资源**：使用PAI的监控功能查看GPU利用率
4. **及时保存**：训练过程中定期保存checkpoint
5. **多实验对比**：尝试不同的超参数组合

## 相关链接

- [阿里云PAI文档](https://help.aliyun.com/product/30347.html)
- [PAI-DLC文档](https://help.aliyun.com/document_detail/203124.html)
- [OSS使用指南](https://help.aliyun.com/document_detail/31883.html)

## 技术支持

遇到问题可以：
1. 查看PAI控制台的帮助文档
2. 在阿里云社区提问
3. 提交工单联系技术支持
