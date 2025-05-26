import torch
import numpy as np
from typing import Dict, List, Any, Sequence, Optional, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from .base_metric import BaseMetric
from .registry import MetricRegistry


@MetricRegistry.register('Accuracy')
class Accuracy(BaseMetric):
    """准确率评估指标
    
    Args:
        top_k: 计算top-k准确率，可以是单个数值或元组
        prefix: 指标前缀
        **kwargs: 其他参数
    """
    
    default_prefix = 'acc'

    def __init__(self, 
                 top_k: Union[int, tuple] = 1,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(top_k, int):
            self.top_k = (top_k,)
        else:
            self.top_k = top_k

    def process(self, 
                predictions: Union[torch.Tensor, np.ndarray], 
                targets: Union[torch.Tensor, np.ndarray],
                **kwargs) -> None:
        """处理一个批次的数据"""
        # 转换为numpy数组以便统一处理
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        # 存储整个batch的结果
        self.results.append({
            'predictions': predictions,
            'targets': targets
        })

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """计算准确率指标"""
        # 拼接所有batch的结果
        all_predictions = np.concatenate([res['predictions'] for res in results], axis=0)
        all_targets = np.concatenate([res['targets'] for res in results], axis=0)
        
        metrics = {}
        
        for k in self.top_k:
            if k == 1:
                # Top-1准确率
                if all_predictions.ndim > 1:
                    # 如果是logits，取最大值索引
                    pred_labels = np.argmax(all_predictions, axis=1)
                else:
                    pred_labels = all_predictions
                acc = np.mean(pred_labels == all_targets)
            else:
                # Top-k准确率
                if all_predictions.ndim == 1:
                    raise ValueError(f"无法计算top-{k}准确率，预测结果必须是二维数组")
                
                # 获取top-k预测索引
                top_k_indices = np.argsort(all_predictions, axis=1)[:, -k:]
                # 检查真实标签是否在top-k中
                acc = np.mean([target in pred_indices for target, pred_indices in zip(all_targets, top_k_indices)])
            
            if len(self.top_k) == 1:
                metrics['accuracy'] = float(acc)
            else:
                metrics[f'top_{k}_accuracy'] = float(acc)
        
        return metrics


@MetricRegistry.register('Precision')
class Precision(BaseMetric):
    """精确率评估指标"""
    
    default_prefix = 'precision'

    def __init__(self, 
                 average: str = 'macro',
                 **kwargs):
        super().__init__(**kwargs)
        self.average = average

    def process(self, 
                predictions: Union[torch.Tensor, np.ndarray], 
                targets: Union[torch.Tensor, np.ndarray],
                **kwargs) -> None:
        """处理一个批次的数据"""
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        # 如果是logits，转换为标签
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
                
        self.results.append({
            'predictions': predictions,
            'targets': targets
        })

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """计算精确率指标"""
        all_predictions = np.concatenate([res['predictions'] for res in results], axis=0)
        all_targets = np.concatenate([res['targets'] for res in results], axis=0)
        
        precision, _, _, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average=self.average, zero_division=0)
        
        return {'precision': float(precision)}


@MetricRegistry.register('Recall')
class Recall(BaseMetric):
    """召回率评估指标"""
    
    default_prefix = 'recall'

    def __init__(self, 
                 average: str = 'macro',
                 **kwargs):
        super().__init__(**kwargs)
        self.average = average

    def process(self, 
                predictions: Union[torch.Tensor, np.ndarray], 
                targets: Union[torch.Tensor, np.ndarray],
                **kwargs) -> None:
        """处理一个批次的数据"""
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        # 如果是logits，转换为标签
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
                
        self.results.append({
            'predictions': predictions,
            'targets': targets
        })

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """计算召回率指标"""
        all_predictions = np.concatenate([res['predictions'] for res in results], axis=0)
        all_targets = np.concatenate([res['targets'] for res in results], axis=0)
        
        _, recall, _, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average=self.average, zero_division=0)
        
        return {'recall': float(recall)}


@MetricRegistry.register('F1Score')
class F1Score(BaseMetric):
    """F1分数评估指标"""
    
    default_prefix = 'f1'

    def __init__(self, 
                 average: str = 'macro',
                 **kwargs):
        super().__init__(**kwargs)
        self.average = average

    def process(self, 
                predictions: Union[torch.Tensor, np.ndarray], 
                targets: Union[torch.Tensor, np.ndarray],
                **kwargs) -> None:
        """处理一个批次的数据"""
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        # 如果是logits，转换为标签
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
                
        self.results.append({
            'predictions': predictions,
            'targets': targets
        })

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """计算F1分数指标"""
        all_predictions = np.concatenate([res['predictions'] for res in results], axis=0)
        all_targets = np.concatenate([res['targets'] for res in results], axis=0)
        
        _, _, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average=self.average, zero_division=0)
        
        return {'f1_score': float(f1)}


@MetricRegistry.register('ConfusionMatrix')
class ConfusionMatrix(BaseMetric):
    """混淆矩阵评估指标"""
    
    default_prefix = 'cm'

    def __init__(self, 
                 num_classes: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def process(self, 
                predictions: Union[torch.Tensor, np.ndarray], 
                targets: Union[torch.Tensor, np.ndarray],
                **kwargs) -> None:
        """处理一个批次的数据"""
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        # 如果是logits，转换为标签
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
                
        self.results.append({
            'predictions': predictions,
            'targets': targets
        })

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """计算混淆矩阵相关指标"""
        all_predictions = np.concatenate([res['predictions'] for res in results], axis=0)
        all_targets = np.concatenate([res['targets'] for res in results], axis=0)
        
        cm = confusion_matrix(all_targets, all_predictions)
        
        # 计算每类的精确率、召回率和F1分数
        metrics = {}
        
        if self.num_classes is None:
            self.num_classes = len(np.unique(all_targets))
        
        for i in range(min(self.num_classes, cm.shape[0])):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[f'class_{i}_precision'] = float(precision)
            metrics[f'class_{i}_recall'] = float(recall)
            metrics[f'class_{i}_f1'] = float(f1)
        
        return metrics 