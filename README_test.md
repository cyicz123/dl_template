# PlateNet 测试脚本使用指南

## 概述

`test.py` 脚本用于加载训练好的模型权重，对测试集进行推理，识别干净盘子的ID，并生成提交到大作业结果测评系统的请求命令。

## 功能特性

- ✅ 加载训练好的模型权重
- ✅ 对测试集进行批量推理
- ✅ 自动提取图像ID（支持多种命名格式）
- ✅ 筛选干净盘子的预测结果
- ✅ 生成不同操作系统的curl请求命令
- ✅ 保存详细的推理结果和统计信息

## 使用方法

### 数据集文件夹结构说明：
 当 use_split_folders: true 时，期望的文件夹结构为：
```
 data/plate/
 ├── train/
 │   ├── clean_dish/
 │   │   ├── image1.jpg
 │   │   └── image2.jpg
 │   └── dirty_dish/
 │       ├── image1.jpg
 │       └── image2.jpg
 ├── val/
 │   ├── clean_dish/
 │   └── dirty_dish/
 └── test/
     ├── clean_dish/
     └── dirty_dish/
```
### 基本用法

```bash
python tools/test.py \
    --config configs/datasets/plate.yaml \
    --model-config configs/models/vgg16.yaml \
    --checkpoint experiments/your_experiment/checkpoints/model_best.pth \
    --device cuda \
    --batch-size 32 \
    --output-dir test_results
```

### 参数说明

- `--config`: 数据集配置文件路径（必需）
- `--model-config`: 模型配置文件路径（必需）
- `--checkpoint`: 模型权重文件路径（必需）
- `--device`: 设备类型，可选 `cuda` 或 `cpu`，默认为 `cuda`
- `--batch-size`: 批处理大小，默认为 32
- `--output-dir`: 结果输出目录，默认为 `test_results`

## 使用示例

### 1. 使用VGG16模型进行测试

```bash
python tools/test.py \
    --config configs/datasets/plate.yaml \
    --model-config configs/models/vgg16.yaml \
    --checkpoint experiments/vgg-lr1e5-plate3-warmup_20241201_143022/checkpoints/model_best.pth
```

### 2. 使用GoogleNet模型进行测试

```bash
python tools/test.py \
    --config configs/datasets/plate.yaml \
    --model-config configs/models/googlenet.yaml \
    --checkpoint experiments/googlenet-lr1e5-plate3-warmup_20241201_150000/checkpoints/model_best.pth
```

### 3. 使用DenseNet模型进行测试

```bash
python tools/test.py \
    --config configs/datasets/plate.yaml \
    --model-config configs/models/densenet.yaml \
    --checkpoint experiments/densenet-lr1e5-plate3-warmup_20241201_160000/checkpoints/model_best.pth
```

### 4. 使用Vision Transformer模型进行测试

```bash
python tools/test.py \
    --config configs/datasets/plate.yaml \
    --model-config configs/models/image_vit.yaml \
    --checkpoint experiments/vit-lr1e5-plate3-warmup_20241201_170000/checkpoints/model_best.pth
```

## 输出结果

脚本运行后会在指定的输出目录中生成以下文件：

### 1. `inference_results.txt`
包含所有测试图像的详细推理结果：
```
图像路径	预测类别	置信度	图像ID
data/plate(3)/test/clean_dish/0001.jpg	干净盘子	0.9876	1
data/plate(3)/test/dirty_dish/0002.jpg	脏盘子	0.8765	2
...
```

### 2. `clean_dish_ids.txt`
包含所有被识别为干净盘子的ID列表：
```
1,5,8,12,15,18,22,25,28,31,34,37,40,43,46,49
```

### 3. `request_commands.txt`
包含不同操作系统的请求命令：
```
=== 大作业结果测评系统 - 请求命令 ===

干净盘子ID列表: 1,5,8,12,15,18,22,25,28,31,34,37,40,43,46,49
总共识别出 16 个干净盘子

Windows 系统请求命令:
curl "http://202.207.12.156:20000/calculate_accuracy?cleaned_ids=1,5,8,12,15,18,22,25,28,31,34,37,40,43,46,49"

Linux 系统请求命令:
curl 'http://202.207.12.156:20000/calculate_accuracy?cleaned_ids=1,5,8,12,15,18,22,25,28,31,34,37,40,43,46,49'

Mac 系统请求命令:
curl 'http://202.207.12.156:20000/calculate_accuracy?cleaned_ids=1,5,8,12,15,18,22,25,28,31,34,37,40,43,46,49'
```

## 控制台输出

脚本运行时会在控制台显示：

```
============================================================
测试结果
============================================================
总测试图像数: 100
识别为干净盘子的数量: 16
识别为脏盘子的数量: 84

干净盘子ID列表: 1,5,8,12,15,18,22,25,28,31,34,37,40,43,46,49

============================================================
请求命令
============================================================
Windows 系统:
curl "http://202.207.12.156:20000/calculate_accuracy?cleaned_ids=1,5,8,12,15,18,22,25,28,31,34,37,40,43,46,49"

Linux 系统:
curl 'http://202.207.12.156:20000/calculate_accuracy?cleaned_ids=1,5,8,12,15,18,22,25,28,31,34,37,40,43,46,49'

Mac 系统:
curl 'http://202.207.12.156:20000/calculate_accuracy?cleaned_ids=1,5,8,12,15,18,22,25,28,31,34,37,40,43,46,49'
============================================================
```

## 图像ID提取规则

脚本支持多种图像文件命名格式，自动提取数字ID：

- `0028.jpg` → `28`
- `test_0028.jpg` → `28`  
- `image28.jpg` → `28`
- `plate_001.png` → `1`

如果文件名中没有数字，则使用完整的文件名（不含扩展名）作为ID。

## 数据集要求

确保测试数据集按照以下结构组织：

```
data/
└── plate(3)/
    └── test/
        ├── clean_dish/
        │   ├── 0001.jpg
        │   ├── 0002.jpg
        │   └── ...
        └── dirty_dish/
            ├── 0001.jpg
            ├── 0002.jpg
            └── ...
```

或者使用配置文件中指定的其他数据集结构。

## 注意事项

1. **权重文件路径**: 确保指定的权重文件存在且可访问
2. **配置文件匹配**: 模型配置文件必须与权重文件对应的模型架构匹配
3. **数据集配置**: 数据集配置文件中的路径和结构必须正确
4. **设备选择**: 如果没有GPU，脚本会自动切换到CPU模式
5. **内存使用**: 根据可用内存调整批处理大小

## 故障排除

### 常见错误及解决方案

1. **权重文件不存在**
   ```
   错误: 权重文件不存在: /path/to/checkpoint.pth
   解决: 检查权重文件路径是否正确
   ```

2. **模型架构不匹配**
   ```
   错误: size mismatch for fc.weight
   解决: 确保模型配置文件与权重文件匹配
   ```

3. **数据集路径错误**
   ```
   错误: 盘子数据集目录不存在
   解决: 检查数据集配置文件中的路径设置
   ```

4. **CUDA内存不足**
   ```
   错误: CUDA out of memory
   解决: 减小batch_size或使用CPU模式
   ```

## 提交结果

复制生成的curl命令到对应的操作系统终端中执行，即可提交结果到大作业测评系统。

例如在Linux系统中：
```bash
curl 'http://202.207.12.156:20000/calculate_accuracy?cleaned_ids=1,5,8,12,15,18,22,25,28,31,34,37,40,43,46,49'
``` 