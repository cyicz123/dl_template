from typing import Any, List, Optional, Sequence, Union, Dict

import torch
import numpy as np

from .base_metric import BaseMetric
from .registry import MetricRegistry
from ..utils.logger import get_logger


class Evaluator:
    """评估器，用于组合多个评估指标并统一管理评估过程
    
    Args:
        metrics: 评估指标配置，可以是单个指标配置或指标配置列表
    """

    def __init__(self, metrics: Union[dict, BaseMetric, Sequence]):
        self.logger = get_logger(self.__class__.__name__)
        self._dataset_meta: Optional[dict] = None
        
        if not isinstance(metrics, Sequence):
            metrics = [metrics]
            
        self.metrics: List[BaseMetric] = []
        for metric in metrics:
            if isinstance(metric, dict):
                self.metrics.append(MetricRegistry.create(metric))
            elif isinstance(metric, BaseMetric):
                self.metrics.append(metric)
            else:
                raise ValueError(f"不支持的指标类型: {type(metric)}")

    @property
    def dataset_meta(self) -> Optional[dict]:
        """数据集元信息"""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """设置数据集元信息"""
        self._dataset_meta = dataset_meta
        for metric in self.metrics:
            if hasattr(metric, 'dataset_meta'):
                metric.dataset_meta = dataset_meta

    def process(self,
                predictions: Union[torch.Tensor, np.ndarray],
                targets: Union[torch.Tensor, np.ndarray],
                **kwargs):
        """处理一个批次的数据
        
        Args:
            predictions: 模型预测结果，shape为 [batch_size, ...]
            targets: 目标标签，shape为 [batch_size, ...]
            **kwargs: 其他可选参数
        """
        for metric in self.metrics:
            metric.process(predictions, targets, **kwargs)

    def evaluate(self, size: int) -> Dict[str, float]:
        """评估所有指标
        
        Args:
            size: 数据集大小
            
        Returns:
            所有指标的评估结果
        """
        metrics = {}
        for metric in self.metrics:
            _results = metric.evaluate(size)

            # 检查指标名冲突
            for name in _results.keys():
                if name in metrics:
                    self.logger.warning(
                        f'指标名称冲突: {name}. 请确保所有指标具有不同的前缀.')

            metrics.update(_results)
        return metrics

    def reset(self):
        """重置所有指标"""
        for metric in self.metrics:
            metric.reset()