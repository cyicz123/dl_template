# DL-Template - 深度学习训练框架

<div align="center">
  <strong>一个轻量级、模块化的PyTorch深度学习训练框架</strong>
  <br>
  <br>
</div>

<p align="center">
  <a href="#特点">特点</a> •
  <a href="#目录结构">目录结构</a> •
  <a href="#安装">安装</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#文档">文档</a> •
  <a href="#许可证">许可证</a>
</p>

<br>

DL-Template是一个用于深度学习实验和研究的轻量级PyTorch训练框架。该框架提供了一套完整的工具，用于构建、训练和评估深度学习模型，同时支持模型和数据集的注册机制以及可视化功能。

## 特点

✨ **模块化设计** - 框架各组件解耦，易于扩展和定制  
🔄 **注册系统** - 模型、数据集和损失函数的注册和检索系统  
📝 **配置系统** - 灵活的YAML配置支持，使实验管理更简单  
📊 **可视化支持** - 内置TensorBoard支持，提供丰富的可视化功能  
📈 **监控工具** - 梯度流、特征图等深度监控工具  
💾 **检查点管理** - 自动保存检查点和恢复训练  
⏱️ **训练优化** - 支持早停和多种学习率调度策略  

## 目录结构

```
dl_framework/
├── models/              # 模型定义
│   ├── registry.py      # 模型注册系统
│   ├── base_model.py    # 模型基类
│   └── cnn.py           # CNN模型实现
├── datasets/            # 数据集定义
│   ├── registry.py      # 数据集注册系统
│   ├── base_dataset.py  # 数据集基类
│   └── cifar10.py       # CIFAR-10数据集实现
├── losses/              # 损失函数定义
│   ├── registry.py      # 损失函数注册系统
│   ├── base_loss.py     # 损失函数基类
│   ├── cross_entropy.py # 交叉熵损失函数
│   ├── mse_loss.py      # 均方误差损失函数
│   └── l1_loss.py       # L1损失函数
├── trainers/            # 训练器
│   └── base_trainer.py  # 训练器基类
├── visualization/       # 可视化工具
│   ├── base_visualizer.py  # 可视化器基类
│   ├── tensorboard.py      # TensorBoard可视化器
│   └── hooks/              # 可视化钩子
│       ├── base_hook.py    # 钩子基类
│       └── grad_flow.py    # 梯度流可视化钩子
└── utils/               # 工具函数
    ├── logger.py        # 日志工具
    ├── checkpoint.py    # 检查点工具
    └── config.py        # 配置工具

configs/              # 配置文件
├── models/           # 模型配置
│   └── cnn.yaml      # CNN模型配置
├── datasets/         # 数据集配置
│   └── cifar10.yaml  # CIFAR-10数据集配置
├── training/         # 训练配置
│   └── default.yaml  # 默认训练配置
└── visualization/    # 可视化配置
    └── tensorboard.yaml  # TensorBoard配置

tools/                # 命令行工具
└── train.py          # 训练入口脚本

docs/                 # 文档
experiments/          # 实验结果文件夹
└── experiment_20250330_235610 # 单次结果文件夹
    ├── checkpoints # 检查点文件夹
    ├── config.yaml # 配置文件
    ├── logs # 日志文件夹
    └── visualization # 可视化文件夹
```

## 安装

### 要求

- Python 3.8+
- PyTorch 1.8+
- torchvision
- pyyaml
- matplotlib
- tensorboard
- psutil
- pynvml

### 步骤

1. Fork并克隆仓库：

```bash
# 1. 在GitHub上fork此仓库到你的账号下
# 2. 将fork后的仓库重命名为你想要的名称（可选）
# 3. 克隆你的仓库
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名

# 4. 设置原始仓库为upstream
git remote add upstream https://github.com/cyicz123/dl_template.git

# 5. 验证远程仓库设置
git remote -v

# 6. 当原始仓库有更新时，使用以下命令更新你的仓库
git fetch upstream
git rebase upstream/main
```

2. 安装uv和同步依赖：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## 快速开始

使用默认配置训练模型：

```bash
python tools/train.py --config configs/training/default.yaml
```

使用TensorBoard可视化：

```bash
python tools/train.py --config configs/training/default.yaml --vis configs/visualization/tensorboard.yaml
```

指定GPU：

```bash
python tools/train.py --config configs/training/default.yaml --device cuda:0
```

## 文档

详细文档请参考以下链接：

- [使用指南](docs/usage_guide.md) - 框架基本使用方法
- [自定义模型教程](docs/custom_model.md) - 如何创建和注册自定义模型
- [自定义数据集教程](docs/custom_dataset.md) - 如何创建和注册自定义数据集
- [自定义损失函数教程](docs/losses.md) - 如何创建和注册自定义损失函数
- [TensorBoard可视化](docs/tensorboard_visualization.md) - 如何使用TensorBoard进行可视化
- [钩子系统使用指南](docs/hooks_usage.md) - 如何使用钩子系统扩展训练功能
- [学习率调度器使用指南](docs/scheduler_usage.md) - 如何使用和自定义学习率调度策略
- [团队协作Git使用指南](docs/github-team-workflow.md) - 如何使用Git和Github进行标准化开发

## 贡献

欢迎贡献代码和提出建议！请参考[贡献指南](docs/github-team-workflow.md)。

## 许可证

MIT
