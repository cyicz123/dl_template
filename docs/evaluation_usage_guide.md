# Evaluation 系统使用指南

## 概述

本文档介绍现有指标的使用方法、自定义指标的开发和最佳实践建议。

## 1. 使用现有指标

在配置文件中，可以通过`evaluation`字段配置评估指标。

```yaml
# config.yaml
evaluation:
  train_metrics:
    - type: Accuracy
      top_k: 1
      prefix: train
    - type: F1Score
      average: macro
      prefix: train

  val_metrics:
    - type: Accuracy
      top_k: [1, 5]
      prefix: val
    - type: Precision
      average: macro
      prefix: val
    - type: Recall
      average: macro
      prefix: val
    - type: F1Score
      average: macro
      prefix: val
    - type: ConfusionMatrix
      num_classes: 10
      prefix: val
```

### 1.1 内置指标详解

#### 分类指标

```python
# 准确率 - 支持 top-k
accuracy = Accuracy(top_k=1)              # top-1 准确率
accuracy = Accuracy(top_k=(1, 3, 5))      # 多个 top-k 准确率

# 精确率、召回率、F1分数
precision = Precision(average='macro')     # macro/micro/weighted/binary
recall = Recall(average='macro')
f1 = F1Score(average='macro')

# 混淆矩阵
cm = ConfusionMatrix(num_classes=10)       # 10分类任务
```

#### 回归指标

```python
# 各种误差指标
mse = MeanSquaredError()      # 均方误差
mae = MeanAbsoluteError()     # 平均绝对误差  
rmse = RootMeanSquaredError() # 均方根误差
r2 = R2Score()               # 决定系数

# 回归任务示例
regression_evaluator = Evaluator([
    {'type': 'MSE'},
    {'type': 'MAE'},
    {'type': 'RMSE'},
    {'type': 'R2Score'}
])
```
## 2. 新增自定义指标

### 2.1 基础模板

```python
from dl_framework.evaluation import BaseMetric, MetricRegistry
import torch
import numpy as np
from typing import Dict, List, Union

@MetricRegistry.register('CustomAccuracy')
class CustomAccuracy(BaseMetric):
    """自定义准确率指标示例"""
    
    # 默认前缀
    default_prefix = 'custom_acc'
    
    def __init__(self, 
                 threshold: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
    
    def process(self, 
                predictions: Union[torch.Tensor, np.ndarray], 
                targets: Union[torch.Tensor, np.ndarray],
                **kwargs) -> None:
        """处理一个批次的数据"""
        # 1. 转换为 numpy 数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # 2. 数据预处理（根据需要）
        if predictions.ndim > 1:
            # 如果是 logits，转换为概率或标签
            pred_labels = np.argmax(predictions, axis=1)
        else:
            # 如果是概率，应用阈值
            pred_labels = (predictions > self.threshold).astype(int)
        
        # 3. 存储批次结果
        self.results.append({
            'predictions': pred_labels,
            'targets': targets
        })
    
    def compute_metrics(self, results: List) -> Dict[str, float]:
        """计算最终指标"""
        # 1. 合并所有批次结果
        all_predictions = np.concatenate([res['predictions'] for res in results])
        all_targets = np.concatenate([res['targets'] for res in results])
        
        # 2. 计算自定义指标
        accuracy = np.mean(all_predictions == all_targets)
        
        # 3. 返回指标字典
        return {'custom_accuracy': float(accuracy)}
```

### 2.2 复杂指标示例

```python
@MetricRegistry.register('WeightedAccuracy')
class WeightedAccuracy(BaseMetric):
    """加权准确率指标"""
    
    default_prefix = 'weighted_acc'
    
    def __init__(self, 
                 class_weights: List[float] = None,
                 num_classes: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.num_classes = num_classes
        
        # 参数验证
        if class_weights is not None and num_classes is not None:
            if len(class_weights) != num_classes:
                raise ValueError("class_weights 长度必须等于 num_classes")
    
    def process(self, predictions, targets, **kwargs):
        """处理批次数据"""
        # 转换为 numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # 获取预测标签
        if predictions.ndim > 1:
            pred_labels = np.argmax(predictions, axis=1)
        else:
            pred_labels = predictions
        
        # 存储结果，包含额外信息
        self.results.append({
            'predictions': pred_labels,
            'targets': targets,
            'batch_size': len(targets)
        })
    
    def compute_metrics(self, results):
        """计算加权准确率"""
        all_predictions = np.concatenate([res['predictions'] for res in results])
        all_targets = np.concatenate([res['targets'] for res in results])
        
        if self.class_weights is None:
            # 普通准确率
            accuracy = np.mean(all_predictions == all_targets)
        else:
            # 加权准确率
            correct_mask = (all_predictions == all_targets)
            weights = np.array([self.class_weights[target] for target in all_targets])
            weighted_accuracy = np.sum(correct_mask * weights) / np.sum(weights)
            accuracy = weighted_accuracy
        
        return {'weighted_accuracy': float(accuracy)}
```

### 2.3 在线计算指标（内存优化）

```python
@MetricRegistry.register('OnlineAccuracy')
class OnlineAccuracy(BaseMetric):
    """在线计算准确率，适用于大数据集"""
    
    default_prefix = 'online_acc'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化累积变量
        self.correct_count = 0
        self.total_count = 0
    
    def process(self, predictions, targets, **kwargs):
        """在线更新统计量"""
        # 转换为 numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # 获取预测标签
        if predictions.ndim > 1:
            pred_labels = np.argmax(predictions, axis=1)
        else:
            pred_labels = predictions
        
        # 在线更新统计量
        batch_correct = np.sum(pred_labels == targets)
        batch_total = len(targets)
        
        self.correct_count += batch_correct
        self.total_count += batch_total
        
        # 不需要存储所有结果，节省内存
    
    def compute_metrics(self, results):
        """计算最终准确率"""
        if self.total_count == 0:
            return {'online_accuracy': 0.0}
        
        accuracy = self.correct_count / self.total_count
        return {'online_accuracy': float(accuracy)}
    
    def reset(self):
        """重置内部状态"""
        super().reset()
        self.correct_count = 0
        self.total_count = 0
```

### 2.4 使用自定义指标

```python
# 方式1: 直接实例化
custom_metric = CustomAccuracy(threshold=0.7, prefix='custom')

# 方式2: 通过配置字典
evaluator = Evaluator([
    {'type': 'CustomAccuracy', 'threshold': 0.7, 'prefix': 'custom'},
    {'type': 'WeightedAccuracy', 'class_weights': [1.0, 2.0, 1.5], 'num_classes': 3},
    {'type': 'OnlineAccuracy', 'prefix': 'online'}
])

# 方式3: 配置文件
# config.yaml
evaluation:
  metrics:
    - type: CustomAccuracy
      threshold: 0.7
      prefix: custom
    - type: WeightedAccuracy
      class_weights: [1.0, 2.0, 1.5]
      num_classes: 3
```

## 3. 最佳实践

### 3.1 指标选择原则

#### 分类任务指标选择

```python
# 平衡数据集
balanced_metrics = [
    {'type': 'Accuracy', 'top_k': 1},
    {'type': 'F1Score', 'average': 'macro'}
]

# 不平衡数据集
imbalanced_metrics = [
    {'type': 'Precision', 'average': 'weighted'},
    {'type': 'Recall', 'average': 'weighted'},
    {'type': 'F1Score', 'average': 'weighted'},
    {'type': 'F1Score', 'average': 'macro'}  # 对比用
]

# 多标签分类
multilabel_metrics = [
    {'type': 'F1Score', 'average': 'macro'},
    {'type': 'F1Score', 'average': 'micro'},
    {'type': 'Precision', 'average': 'macro'},
    {'type': 'Recall', 'average': 'macro'}
]

# Top-K 任务（如推荐系统）
topk_metrics = [
    {'type': 'Accuracy', 'top_k': (1, 5, 10, 20)}
]
```

#### 回归任务指标选择

```python
# 标准回归评估
regression_metrics = [
    {'type': 'MSE'},      # 关注大误差
    {'type': 'MAE'},      # 关注平均误差
    {'type': 'R2Score'}   # 模型解释性
]

# 关注绝对误差
mae_focused = [
    {'type': 'MAE'},
    {'type': 'MSE'}
]

# 时间序列预测
timeseries_metrics = [
    {'type': 'MAE'},
    {'type': 'RMSE'},
    {'type': 'R2Score'}
]
```

### 3.2 性能优化

#### GPU内存管理

```python
def process(self, predictions, targets, **kwargs):
    """正确的GPU内存管理"""
    # ✅ 立即转移到CPU避免GPU内存累积
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # 继续处理...
```

#### 大数据集优化

```python
# ✅ 对于大数据集，使用在线计算
class OnlineMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.running_sum = 0
        self.count = 0
    
    def process(self, predictions, targets, **kwargs):
        # 直接更新累积值，不存储所有数据
        batch_metric = self._compute_batch_metric(predictions, targets)
        self.running_sum += batch_metric * len(targets)
        self.count += len(targets)
    
    def compute_metrics(self, results):
        return {'metric': self.running_sum / self.count if self.count > 0 else 0.0}

# ❌ 避免这样做（内存占用过大）
class BadMetric(BaseMetric):
    def process(self, predictions, targets, **kwargs):
        # 存储所有原始数据会导致内存爆炸
        self.results.append({
            'predictions': predictions,  # 可能非常大
            'targets': targets
        })
```

### 3.3 错误处理

#### 健壮的指标实现

```python
def compute_metrics(self, results):
    """带完善错误处理的指标计算"""
    try:
        # 检查输入
        if not results:
            self.logger.warning(f"{self.__class__.__name__}: 无可用结果")
            return {self.metric_name: 0.0}
        
        # 合并数据
        all_predictions = np.concatenate([res['predictions'] for res in results])
        all_targets = np.concatenate([res['targets'] for res in results])
        
        # 维度检查
        if len(all_predictions) != len(all_targets):
            raise ValueError(f"预测和目标数量不匹配: {len(all_predictions)} vs {len(all_targets)}")
        
        # 计算指标
        metric_value = self._safe_compute(all_predictions, all_targets)
        
        return {self.metric_name: float(metric_value)}
        
    except Exception as e:
        self.logger.error(f"{self.__class__.__name__} 计算错误: {e}")
        return {self.metric_name: 0.0}

def _safe_compute(self, predictions, targets):
    """安全的指标计算"""
    try:
        # 核心计算逻辑
        result = self._compute_core_metric(predictions, targets)
        
        # 结果验证
        if np.isnan(result) or np.isinf(result):
            self.logger.warning("计算结果为 NaN 或 Inf，返回 0.0")
            return 0.0
            
        return result
        
    except ZeroDivisionError:
        self.logger.warning("除零错误，返回 0.0")
        return 0.0
    except ValueError as e:
        self.logger.warning(f"数值错误: {e}")
        return 0.0
```

### 3.4 调试技巧

#### 验证指标正确性

```python
def test_metric():
    """测试自定义指标的正确性"""
    # 创建已知结果的测试数据
    test_predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    test_targets = torch.tensor([1, 0, 1])
    
    # 预期结果: 全部正确，准确率应为1.0
    metric = CustomAccuracy()
    metric.process(test_predictions, test_targets)
    results = metric.evaluate(3)
    
    expected = 1.0
    actual = results['custom_acc/custom_accuracy']
    
    assert abs(actual - expected) < 1e-6, f"预期 {expected}, 实际 {actual}"
    print("✅ 指标测试通过")

# 运行测试
test_metric()
```

#### 调试模式

```python
import logging

# 启用详细日志
logging.getLogger('dl_framework.evaluation').setLevel(logging.DEBUG)

# 检查已注册指标
from dl_framework.evaluation import MetricRegistry
print("可用指标:", list(MetricRegistry.list().keys()))

# 单步调试
metric = Accuracy()
print(f"初始状态: {len(metric.results)}")

metric.process(test_predictions, test_targets)
print(f"处理后: {len(metric.results)}")

results = metric.evaluate(len(test_targets))
print(f"最终结果: {results}")
```

### 3.5 集成到训练流程

#### 完整训练示例

```python
class Trainer:
    def __init__(self, config):
        # 创建评估器
        self.train_evaluator = Evaluator(config['train_metrics'])
        self.val_evaluator = Evaluator(config['val_metrics'])
    
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        
        for batch in self.train_dataloader:
            # 前向传播
            outputs = self.model(batch['inputs'])
            loss = self.criterion(outputs, batch['targets'])
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新训练指标
            self.train_evaluator.process(outputs.detach(), batch['targets'])
        
        # 获取训练指标
        train_metrics = self.train_evaluator.evaluate(len(self.train_dataset))
        return train_metrics
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs = self.model(batch['inputs'])
                self.val_evaluator.process(outputs, batch['targets'])
        
        # 获取验证指标
        val_metrics = self.val_evaluator.evaluate(len(self.val_dataset))
        return val_metrics
    
    def train(self, num_epochs):
        """完整训练流程"""
        for epoch in range(num_epochs):
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 日志记录
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  训练: {train_metrics}")
            print(f"  验证: {val_metrics}")
            
            # 可以在这里添加早停、学习率调整等逻辑
```

### 3.6 常见陷阱和解决方案

#### 陷阱1: 指标名称冲突

```python
# ❌ 问题：多个指标返回相同名称
evaluator = Evaluator([
    {'type': 'Accuracy'},  # 默认前缀: 'acc'
    {'type': 'Accuracy'}   # 同样前缀: 'acc' -> 冲突！
])

# ✅ 解决：使用不同前缀
evaluator = Evaluator([
    {'type': 'Accuracy', 'prefix': 'train'},
    {'type': 'Accuracy', 'prefix': 'val'}
])
```

#### 陷阱2: 维度不匹配

```python
def process(self, predictions, targets, **kwargs):
    # ✅ 添加维度检查和转换
    if predictions.ndim > targets.ndim:
        if predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)  # 去除多余维度
        else:
            predictions = torch.argmax(predictions, dim=1)  # 取最大值索引
    
    # 维度验证
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError(f"批次大小不匹配: {predictions.shape[0]} vs {targets.shape[0]}")
```

#### 陷阱3: 分布式训练不一致

```python
def evaluate(self, size):
    """支持分布式训练的评估"""
    # 分布式环境中同步结果
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # 收集所有进程的结果
        world_size = torch.distributed.get_world_size()
        gathered_results = [None] * world_size
        torch.distributed.all_gather_object(gathered_results, self.results)
        
        # 合并结果
        all_results = []
        for results in gathered_results:
            all_results.extend(results)
        
        metrics = self.compute_metrics(all_results)
    else:
        metrics = super().evaluate(size)
    
    return metrics
```

## 总结

通过本指南，您应该能够：

1. **熟练使用现有指标** - 掌握各种内置指标的配置和使用方法
2. **开发自定义指标** - 了解指标开发的标准流程和注意事项  
3. **应用最佳实践** - 避免常见陷阱，编写高效、健壮的评估代码

evaluation 系统的设计目标是提供统一、灵活、高效的模型评估解决方案。合理使用这些工具，可以大大提升模型开发和评估的效率。 