# 学习率调度器使用指南

本文档介绍如何在DL框架中使用学习率调度器，以及如何自定义自己的调度策略。

## 基本使用

在配置文件中，可以通过设置`training.scheduler`来使用学习率调度器：

```yaml
training:
  epochs: 100
  optimizer:
    type: adam
    lr: 0.001
  scheduler:
    type: StepLR         # 调度器类型
    step_size: 10        # 每10个epoch降低学习率
    gamma: 0.1           # 学习率降低为原来的0.1倍
    step_frequency: epoch # 按epoch更新学习率
```

## 内置调度器

框架已经包装了PyTorch的常用调度器，可以直接在配置中使用：

### 1. StepLR

每隔一定步长衰减学习率：

```yaml
scheduler:
  type: StepLR
  step_size: 10     # 每10个epoch衰减一次
  gamma: 0.1        # 衰减为原来的0.1倍
  step_frequency: epoch
```

### 2. CosineAnnealingLR

余弦退火学习率调度：

```yaml
scheduler:
  type: CosineAnnealingLR
  T_max: 100        # 周期长度
  eta_min: 0.00001  # 最小学习率
  step_frequency: epoch
```

### 3. ReduceLROnPlateau

基于验证性能动态调整学习率：

```yaml
scheduler:
  type: ReduceLROnPlateau
  mode: min          # 监控指标的模式（min或max）
  factor: 0.1        # 衰减因子
  patience: 5        # 容忍的epoch数
  threshold: 0.0001  # 改善阈值
  step_metric: loss  # 监控的指标名称
  step_frequency: epoch # 必须是epoch
```

### 4. MultiStepLR

在指定的里程碑处降低学习率：

```yaml
scheduler:
  type: MultiStepLR
  milestones: [30, 60, 90] # 在这些epoch减少学习率
  gamma: 0.1               # 衰减因子
  step_frequency: epoch
```

### 5. ExponentialLR

指数衰减学习率：

```yaml
scheduler:
  type: ExponentialLR
  gamma: 0.95      # 每个epoch将学习率乘以0.95
  step_frequency: epoch
```

### 6. CosineAnnealingWarmRestarts

带重启的余弦退火学习率调度：

```yaml
scheduler:
  type: CosineAnnealingWarmRestarts
  T_0: 10           # 第一次重启的epoch数
  T_mult: 2         # 每次重启后T_0乘以的倍数
  eta_min: 0.00001  # 最小学习率
  step_frequency: epoch
```

## 学习率预热 (Warmup)

预热是一种常见的学习率策略，在训练初期使用较小的学习率，然后逐渐增加到初始设定值。框架提供了预热调度器：

### 1. LinearWarmup

线性增加学习率从初始值的一小部分到完整的初始值：

```yaml
scheduler:
  type: LinearWarmup
  warmup_epochs: 5       # 预热5个epoch
  warmup_start_factor: 0.1 # 从初始学习率的10%开始
  step_frequency: epoch
```

### 2. ConstantWarmup

在预热阶段保持一个恒定的低学习率，然后突增到初始学习率：

```yaml
scheduler:
  type: ConstantWarmup
  warmup_epochs: 5       # 预热5个epoch
  warmup_start_factor: 0.1 # 从初始学习率的10%开始
  step_frequency: epoch
```

### 3. ExponentialWarmup

按指数曲线增加学习率，可以调整增长的曲线斜率：

```yaml
scheduler:
  type: ExponentialWarmup
  warmup_epochs: 5       # 预热5个epoch
  warmup_start_factor: 0.1 # 从初始学习率的10%开始
  warmup_exponent: 2.0   # 指数增长的幂
  step_frequency: epoch
```

## 组合调度器

框架支持将多个调度器按顺序组合，以便在不同训练阶段使用不同的学习率策略：

### CompositeLR

`CompositeLR` 是一个强大的复合调度器，允许您按顺序组合多个调度器，每个调度器在指定的时间段内工作。它是对 PyTorch SequentialLR 的封装，通过配置文件就能轻松实现复杂的学习率调度策略。

基本用法示例：

```yaml
scheduler:
  type: CompositeLR
  step_frequency: epoch
  phases:
    # 第一阶段：线性预热
    - scheduler_name: LinearWarmup  # 调度器名称
      config:                       # 该调度器的配置
        warmup_epochs: 5
        warmup_start_factor: 0.01
      duration: 5                   # 此阶段持续的epoch数
    
    # 第二阶段：恒定学习率
    - scheduler_name: StepLR
      config:
        step_size: 100  # 大步长，相当于恒定学习率
        gamma: 1.0
      duration: 15
    
    # 第三阶段：余弦退火
    - scheduler_name: CosineAnnealingLR
      config:
        T_max: 80
        eta_min: 0.00001
      duration: 80
```

其中：
- `phases` 是一个列表，包含了多个训练阶段的配置
- 每个阶段需要指定三个关键属性：
  - `scheduler_name`: 该阶段使用的调度器名称，必须是已在 `SchedulerRegistry` 中注册的调度器
  - `config`: 该调度器的配置参数
  - `duration`: 该阶段持续的 epoch 数（或 step 数，取决于 `step_frequency`）
- 系统会根据每个阶段的持续时间自动计算切换点

#### 常见复合策略示例

1. **预热 + 多步长衰减**:

```yaml
scheduler:
  type: CompositeLR
  step_frequency: epoch
  phases:
    - scheduler_name: LinearWarmup
      config:
        warmup_epochs: 5
        warmup_start_factor: 0.1
      duration: 5
    
    - scheduler_name: MultiStepLR
      config:
        milestones: [25, 55, 85]  # 相对于此阶段开始的epoch
        gamma: 0.1
      duration: 95
```

2. **预热 + 余弦退火 + 微调**:

```yaml
scheduler:
  type: CompositeLR
  step_frequency: epoch
  phases:
    - scheduler_name: LinearWarmup
      config:
        warmup_epochs: 5
        warmup_start_factor: 0.01
      duration: 5
    
    - scheduler_name: CosineAnnealingLR
      config:
        T_max: 85
        eta_min: 0.0001
      duration: 85
    
    - scheduler_name: ExponentialLR
      config:
        gamma: 0.98  # 轻微衰减
      duration: 10
```

3. **三阶段训练策略**:

```yaml
scheduler:
  type: CompositeLR
  step_frequency: epoch
  phases:
    # 第一阶段：快速学习
    - scheduler_name: StepLR
      config:
        step_size: 10
        gamma: 0.5
      duration: 30
    
    # 第二阶段：缓慢衰减
    - scheduler_name: ExponentialLR
      config:
        gamma: 0.98
      duration: 40
    
    # 第三阶段：精细调整
    - scheduler_name: CosineAnnealingLR
      config:
        T_max: 30
        eta_min: 0.000001
      duration: 30
```

## 按步更新与按周期更新

框架支持两种更新频率：

- `step_frequency: epoch`: 每个epoch结束后更新学习率（默认）
- `step_frequency: step`: 每个训练步结束后更新学习率

注意：某些调度器如`ReduceLROnPlateau`只能按epoch更新，因为它们需要验证指标。

对于组合调度器，`step_frequency`应该设为相同的值并一致应用于所有子调度器。

## 自定义学习率调度器

你可以通过以下步骤创建自定义的学习率调度器：

1. 创建一个继承自`BaseLRSchedulerWrapper`的新类
2. 实现`_build_internal_scheduler`方法
3. 使用`SchedulerRegistry.register`装饰器注册调度器

例如，创建一个自定义的调度器：

```python
# 在 dl_framework/schedulers/custom_schedulers.py 中
import torch.optim.lr_scheduler as lr_scheduler
from .base_scheduler import BaseLRSchedulerWrapper
from .registry import SchedulerRegistry

@SchedulerRegistry.register('CustomStepLR')
class CustomStepLRWrapper(BaseLRSchedulerWrapper):
    def _build_internal_scheduler(self):
        steps = self.config.get('steps', [30, 60, 90])
        gamma = self.config.get('gamma', 0.1)
        
        # 创建一个Lambda调度器
        def lr_lambda(epoch):
            factor = 1.0
            for step in steps:
                if epoch >= step:
                    factor *= gamma
            return factor
        
        return lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
```

然后在配置中使用：

```yaml
scheduler:
  type: CustomStepLR
  steps: [30, 60, 90]
  gamma: 0.1
  step_frequency: epoch
```

## 使用PyTorch的原生调度器

如果需要使用PyTorch的原生调度器而不是包装器，框架仍然支持旧的配置方式以确保向后兼容性：

```yaml
training:
  scheduler:
    type: step  # 注意：这里使用的是小写类型名
    step_size: 10
    gamma: 0.1
```

但是建议使用新的包装器系统，因为它提供了更一致的接口和更多功能。 