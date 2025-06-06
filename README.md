# PlateNet - 车牌识别深度学习框架

<div align="center">
  <strong>一个专注于干净和脏餐具二分类的网络</strong>
  <br>
  <br>
</div>

<p align="center">
  <a href="#特点">特点</a> •
  <a href="#目录结构">目录结构</a> •
  <a href="#安装">安装</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#支持的模型">支持的模型</a> •
  <a href="#数据集">数据集</a> •
  <a href="#文档">文档</a> •
  <a href="#许可证">许可证</a>
</p>

<br>

PlateNet是一个专门用于干净和脏餐具二分类的网络。该框架基于PyTorch构建，提供了完整的工具链用于构建、训练和评估深度学习模型，特别针对干净和脏餐具二分类场景进行了优化。

## 特点

✨ **模块化设计** - 框架各组件解耦，易于扩展和定制  
🚗 **车牌识别专用** - 针对车牌识别任务优化的数据处理和模型配置  
🔄 **注册系统** - 模型、数据集和损失函数的注册和检索系统  
📝 **配置系统** - 灵活的YAML配置支持，使实验管理更简单  
🤖 **多种模型支持** - 支持CNN、VGG、DenseNet、GoogleNet、Vision Transformer等多种模型  
📊 **可视化支持** - 内置TensorBoard支持，提供丰富的可视化功能  
📈 **监控工具** - 梯度流、特征图、系统资源等深度监控工具  
💾 **检查点管理** - 自动保存检查点和恢复训练  
⏱️ **训练优化** - 支持早停、学习率调度、预热等多种训练策略  
🔍 **评估工具** - 完整的模型评估系统，支持准确率、精确率、召回率、F1分数等多种指标  

## 目录结构

```
PlateNet/
├── configs/                    # 配置文件目录
│   ├── datasets/              # 数据集配置
│   │   ├── plate.yaml         # 车牌数据集配置
│   │   ├── cat_dog.yaml       # 猫狗分类数据集配置
│   │   └── cifar10.yaml       # CIFAR10数据集配置
│   ├── models/                # 模型配置
│   │   ├── image_vit.yaml     # Vision Transformer配置
│   │   ├── vgg16_cat_dog.yaml # VGG16配置
│   │   ├── googlenet.yaml     # GoogleNet配置
│   │   ├── densenet.yaml      # DenseNet配置
│   │   └── cnn.yaml           # 简单CNN配置
│   ├── training/              # 训练配置
│   │   ├── default.yaml       # 默认训练配置
│   │   ├── vit_training_example.yaml  # ViT训练示例
│   │   └── ...                # 其他训练配置
│   ├── examples/              # 配置示例
│   └── visualization/         # 可视化配置
├── dl_framework/              # 核心框架代码
│   ├── datasets/              # 数据集模块
│   ├── models/                # 模型模块
│   ├── trainers/              # 训练器模块
│   ├── evaluation/            # 评估模块
│   ├── losses/                # 损失函数模块
│   ├── schedulers/            # 学习率调度器模块
│   ├── hooks/                 # 钩子系统
│   ├── visualization/         # 可视化模块
│   └── utils/                 # 工具函数
├── tools/                     # 工具脚本
│   └── train.py              # 训练脚本
├── experiments/               # 实验结果目录
├── docs/                      # 文档目录
└── README.md                  # 项目说明
```

## 安装

### 要求

- Python 3.10+
- PyTorch 2.0+
- torchvision
- torchaudio
- pyyaml
- matplotlib
- tensorboard
- psutil
- pynvml
- scikit-learn
- timm

### 步骤

1. 克隆仓库：

```bash
git clone https://github.com/你的用户名/PlateNet.git
cd PlateNet
```

2. 安装uv和同步依赖：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## 快速开始

### 基础训练


```bash
# plate(3) 数据集 accuracy 0.98
python tools/train.py --config configs/training/vgg16.yaml --vis configs/visualization/tensorboard.yaml --name vgg-lr1e5-plate3-warmup

# plate(3) 数据集 accuracy 0.9267
python tools/train.py --config configs/training/densenet.yaml --vis configs/visualization/tensorboard.yaml --name densenet-lr1e5-plate3-warmu

# plate(3) 数据集 accuracy 0.985
python tools/train.py --config configs/training/googlenet.yaml --vis configs/visualization/tensorboard.yaml --name googlenet-lr1e5-plate3-warmup

# plate(3) 数据集 accuracy 0.9494
python tools/train.py --config configs/training/vit_training_example.yaml --vis configs/visualization/tensorboard.yaml --name vit-lr1e5-plate3-warmup
```

## 支持的模型

PlateNet支持多种深度学习模型：

- **VGG16** - 经典的VGG架构，适合图像分类任务
- **GoogleNet** - Inception架构，平衡准确率和效率
- **DenseNet** - 密集连接网络，参数效率高
- **Vision Transformer (ViT)** - 基于注意力机制的Transformer模型，适合大规模图像分类

每个模型都支持预训练权重加载和自定义配置。

## 数据集

要使用不同plate数据集，只需要修改`configs/datasets/plate.yaml`中的`data_name`字段为`data`目录下的文件夹名称。
``` bash
├── data
│   ├── plate
│   │   ├── clean_dish
│   │   └── dirty_dish
│   ├── plate(3)
│   │   ├── clean_dish
│   │   └── dirty_dish
```


### 数据集配置示例

```yaml
dataset:
  type: "plate"
  data_name: "plate(3)"
  data_dir: "data"
  batch_size: 32
  num_workers: 4
  shuffle: true
  train_ratio: 0.8
  transforms:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

## 评估指标

框架支持多种评估指标：

- **准确率 (Accuracy)** - Top-1和Top-K准确率
- **精确率 (Precision)** - 宏平均和微平均
- **召回率 (Recall)** - 宏平均和微平均  
- **F1分数 (F1Score)** - 宏平均和微平均
- **混淆矩阵 (Confusion Matrix)** - 分类结果可视化
- **回归指标** - MSE、MAE、RMSE、R²等

## 文档

详细文档请参考以下链接：

- [使用指南](docs/usage_guide.md) - 框架基本使用方法
- [自定义模型教程](docs/custom_model.md) - 如何创建和注册自定义模型
- [自定义数据集教程](docs/custom_dataset.md) - 如何创建和注册自定义数据集
- [自定义损失函数教程](docs/losses.md) - 如何创建和注册自定义损失函数
- [TensorBoard可视化](docs/tensorboard_visualization.md) - 如何使用TensorBoard进行可视化
- [钩子系统使用指南](docs/hooks_usage.md) - 如何使用钩子系统扩展训练功能
- [学习率调度器使用指南](docs/scheduler_usage.md) - 如何使用和自定义学习率调度策略
- [评估工具使用指南](docs/evaluation_usage_guide.md) - 如何使用和自定义评估指标
- [团队协作Git使用指南](docs/github-team-workflow.md) - 如何使用Git和Github进行标准化开发

## 贡献

欢迎贡献代码和提出建议！请参考[贡献指南](docs/github-team-workflow.md)。

## 许可证

MIT
