from .registry import SchedulerRegistry
from .base_scheduler import BaseLRSchedulerWrapper
from .schedulers import *

def build_scheduler(config, optimizer):
    """构建学习率调度器
    
    Args:
        config: 调度器配置，需要包含'type'字段
        optimizer: 优化器实例
        
    Returns:
        调度器实例，如果配置为空或类型未注册则返回None
    """
    if not config:
        return None
        
    scheduler_type = config.get('type')
    if not scheduler_type:
        return None
        
    try:
        scheduler_wrapper = SchedulerRegistry.create(config, optimizer)
        return scheduler_wrapper
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"构建调度器失败: {e}")
        return None

__all__ = [
    'SchedulerRegistry',
    'BaseLRSchedulerWrapper',
    'build_scheduler',
    'StepLRWrapper',
    'CosineAnnealingLRWrapper',
    'ReduceLROnPlateauWrapper',
    'CosineAnnealingWarmRestartsWrapper',
    'MultiStepLRWrapper',
    'ExponentialLRWrapper',
    'LinearWarmupWrapper',
    'ConstantWarmupWrapper',
    'ExponentialWarmupWrapper',
    'CompositeSchedulerWrapper'
] 