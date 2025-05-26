from typing import Dict, Type, Any, List, Union

from .base_metric import BaseMetric


class MetricRegistry:
    """评估指标注册器，用于注册和获取评估指标类"""
    _metrics: Dict[str, Type[BaseMetric]] = {}
    
    @classmethod
    def register(cls, name: str = None):
        """注册评估指标类
        
        Args:
            name: 指标名称，如果为None则使用类名
            
        Returns:
            装饰器函数
        """
        def decorator(metric_cls):
            metric_name = name or metric_cls.__name__
            cls._metrics[metric_name] = metric_cls
            return metric_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseMetric]:
        """获取评估指标类
        
        Args:
            name: 指标名称
            
        Returns:
            指标类
            
        Raises:
            ValueError: 如果指标未注册
        """
        if name not in cls._metrics:
            raise ValueError(f"未注册的评估指标: {name}")
        return cls._metrics[name]
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseMetric:
        """根据配置创建评估指标实例
        
        Args:
            config: 指标配置，必须包含'type'字段
            
        Returns:
            指标实例
        """
        if not isinstance(config, dict):
            raise ValueError(f"指标配置必须是字典类型，得到了: {type(config)}")
            
        if 'type' not in config:
            raise ValueError("指标配置必须包含'type'字段")
            
        config = config.copy()  # 避免修改原配置
        metric_type = config.pop('type')
        metric_class = cls.get(metric_type)
        return metric_class(**config)
    
    @classmethod
    def list(cls) -> Dict[str, Type[BaseMetric]]:
        """获取所有注册的评估指标类
        
        Returns:
            指标名称到类的映射
        """
        return cls._metrics.copy()