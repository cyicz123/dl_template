import torch.optim.lr_scheduler as lr_scheduler
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

class BaseLRSchedulerWrapper(ABC):
    """学习率调度器基类
    
    提供统一的接口来管理不同类型的PyTorch学习率调度器，
    支持按epoch或按step更新学习率，并处理需要指标的调度器。
    """
    
    def __init__(self, optimizer, config):
        """初始化
        
        Args:
            optimizer: 优化器
            config: 调度器配置
        """
        self.optimizer = optimizer
        self.config = config
        
        # 从配置读取更新频率，默认为'epoch'
        self.step_frequency = config.get('step_frequency', 'epoch')
        if self.step_frequency not in ['epoch', 'step']:
            raise ValueError(f"不支持的更新频率: {self.step_frequency}，必须是'epoch'或'step'")
            
        # 对于需要指标的调度器（如ReduceLROnPlateau），指定要监控的指标
        self.step_metric_key = config.get('step_metric', 'loss')
        
        # 创建内部调度器
        self.scheduler = self._build_internal_scheduler()
    
    @abstractmethod
    def _build_internal_scheduler(self):
        """创建内部调度器
        
        Returns:
            PyTorch学习率调度器
        """
        pass
    
    def step(self, metrics: Optional[Dict[str, float]] = None, epoch: Optional[int] = None, step: Optional[int] = None):
        """执行调度器的step
        
        Args:
            metrics: 指标字典，用于需要指标的调度器
            epoch: 当前epoch，用于按epoch更新的调度器
            step: 当前step，用于按step更新的调度器
        """
        # 根据更新频率决定是否执行step
        if self.step_frequency == 'epoch' and epoch is not None:
            return self._step_logic(metrics)
        elif self.step_frequency == 'step' and step is not None:
            return self._step_logic(metrics)
    
    def _step_logic(self, metrics: Optional[Dict[str, float]] = None):
        """内部step逻辑
        
        Args:
            metrics: 指标字典，用于需要指标的调度器
        """
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            if metrics is not None and self.step_metric_key in metrics:
                metric_value = metrics[self.step_metric_key]
                self.scheduler.step(metric_value)
            else:
                # 如果没有指定指标，但调度器需要指标，发出警告但不执行step
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"ReduceLROnPlateau需要指标'{self.step_metric_key}'，但未在metrics中找到。"
                    f"本次step将不会更新学习率。"
                )
        else:
            # 对于不需要指标的调度器，直接执行step
            self.scheduler.step()
    
    def state_dict(self):
        """获取调度器状态
        
        Returns:
            调度器状态字典
        """
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载调度器状态
        
        Args:
            state_dict: 调度器状态字典
        """
        self.scheduler.load_state_dict(state_dict)
    
    def get_last_lr(self) -> List[float]:
        """获取最后的学习率
        
        Returns:
            学习率列表
        """
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        else:
            # 对于不支持get_last_lr的调度器，尝试从optimizer获取
            return [group['lr'] for group in self.optimizer.param_groups] 