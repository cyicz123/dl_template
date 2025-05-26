import logging
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Sequence, Union, Dict

import torch
from torch import Tensor
import numpy as np

from ..utils.logger import get_logger


class BaseMetric(metaclass=ABCMeta):
    """评估指标基类
    
    所有评估指标都应该继承这个基类，并实现process和compute_metrics方法。
    
    Args:
        collect_device (str): 在分布式评测中用于同步结果的设备名，如 'cpu' 或 'gpu'
        prefix (str, optional): 评测指标名前缀，用以区别多个同名的评测指标
        **kwargs: 其他配置参数
    """

    default_prefix: Optional[str] = None

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        self.collect_device = collect_device
        self.prefix = prefix or self.default_prefix
        self.results: List = []
        self.logger = get_logger(self.__class__.__name__)
        
        # 额外配置参数
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def process(self, 
                predictions: Union[torch.Tensor, np.ndarray], 
                targets: Union[torch.Tensor, np.ndarray],
                **kwargs) -> None:
        """处理每个批次的预测结果和目标标签
        
        Args:
            predictions: 模型预测结果，shape为 [batch_size, ...] 
            targets: 目标标签，shape为 [batch_size, ...]
            **kwargs: 其他可选参数，如data_batch等
        """

    @abstractmethod
    def compute_metrics(self, results: List) -> Dict[str, float]:
        """从处理结果中计算评测指标
        
        Args:
            results: 所有批次测试数据经过process()方法处理后得到的结果列表
            
        Returns:
            评测指标字典，键为指标名，值为对应的评测值
        """

    def evaluate(self, size: int) -> Dict[str, float]:
        """评估整个数据集的模型性能
        
        Args:
            size: 整个验证数据集的长度
            
        Returns:
            评测指标字典
        """
        if len(self.results) == 0:
            self.logger.warning(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.')

        # 计算指标
        _metrics = self.compute_metrics(self.results)
        
        # 添加前缀
        if self.prefix:
            _metrics = {
                '/'.join((self.prefix, k)): v
                for k, v in _metrics.items()
            }

        # 重置结果列表
        self.results.clear()
        return _metrics

    def reset(self) -> None:
        """重置评估指标的内部状态"""
        self.results.clear()