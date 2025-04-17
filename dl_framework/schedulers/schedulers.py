import torch.optim.lr_scheduler as lr_scheduler
from .base_scheduler import BaseLRSchedulerWrapper
from .registry import SchedulerRegistry
from torch.optim.lr_scheduler import SequentialLR # 导入 SequentialLR

@SchedulerRegistry.register('StepLR')
class StepLRWrapper(BaseLRSchedulerWrapper):
    """StepLR包装器
    
    配置参数:
        step_size: 学习率调整的步长
        gamma: 学习率衰减因子
        last_epoch: 上次更新的epoch
        step_frequency: 更新频率，'epoch'或'step'
    """
    
    def _build_internal_scheduler(self):
        """创建StepLR调度器
        
        Returns:
            StepLR调度器
        """
        step_size = self.config.get('step_size', 10)
        gamma = self.config.get('gamma', 0.1)
        last_epoch = self.config.get('last_epoch', -1)
        
        return lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch
        )


@SchedulerRegistry.register('CosineAnnealingLR')
class CosineAnnealingLRWrapper(BaseLRSchedulerWrapper):
    """CosineAnnealingLR包装器
    
    配置参数:
        T_max: 最大迭代次数
        eta_min: 最小学习率
        last_epoch: 上次更新的epoch
        step_frequency: 更新频率，'epoch'或'step'
    """
    
    def _build_internal_scheduler(self):
        """创建CosineAnnealingLR调度器
        
        Returns:
            CosineAnnealingLR调度器
        """
        T_max = self.config.get('T_max', 100)
        eta_min = self.config.get('eta_min', 0)
        last_epoch = self.config.get('last_epoch', -1)
        
        return lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch
        )


@SchedulerRegistry.register('ReduceLROnPlateau')
class ReduceLROnPlateauWrapper(BaseLRSchedulerWrapper):
    """ReduceLROnPlateau包装器
    
    配置参数:
        mode: 'min'或'max'
        factor: 学习率衰减因子
        patience: 容忍的epoch数
        threshold: 阈值
        threshold_mode: 阈值模式，'rel'或'abs'
        cooldown: 冷却期
        min_lr: 最小学习率
        eps: 最小变化量
        step_frequency: 更新频率，应为'epoch'
        step_metric: 监控指标，默认为'loss'
    """
    
    def __init__(self, optimizer, config):
        # 确保ReduceLROnPlateau的更新频率为'epoch'
        if config.get('step_frequency', 'epoch') != 'epoch':
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "ReduceLROnPlateau只支持按epoch更新，已将step_frequency强制设置为'epoch'"
            )
            config['step_frequency'] = 'epoch'
        
        super().__init__(optimizer, config)
    
    def _build_internal_scheduler(self):
        """创建ReduceLROnPlateau调度器
        
        Returns:
            ReduceLROnPlateau调度器
        """
        mode = self.config.get('mode', 'min')
        factor = self.config.get('factor', 0.1)
        patience = self.config.get('patience', 10)
        threshold = self.config.get('threshold', 1e-4)
        threshold_mode = self.config.get('threshold_mode', 'rel')
        cooldown = self.config.get('cooldown', 0)
        min_lr = self.config.get('min_lr', 0)
        eps = self.config.get('eps', 1e-8)
        
        return lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps
        )


@SchedulerRegistry.register('CosineAnnealingWarmRestarts')
class CosineAnnealingWarmRestartsWrapper(BaseLRSchedulerWrapper):
    """CosineAnnealingWarmRestarts包装器
    
    配置参数:
        T_0: 第一次重启的迭代次数
        T_mult: 每次重启后T_0乘以的因子
        eta_min: 最小学习率
        last_epoch: 上次更新的epoch
        step_frequency: 更新频率，'epoch'或'step'
    """
    
    def _build_internal_scheduler(self):
        """创建CosineAnnealingWarmRestarts调度器
        
        Returns:
            CosineAnnealingWarmRestarts调度器
        """
        T_0 = self.config.get('T_0', 10)
        T_mult = self.config.get('T_mult', 1)
        eta_min = self.config.get('eta_min', 0)
        last_epoch = self.config.get('last_epoch', -1)
        
        return lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch
        )


@SchedulerRegistry.register('MultiStepLR')
class MultiStepLRWrapper(BaseLRSchedulerWrapper):
    """MultiStepLR包装器
    
    配置参数:
        milestones: 里程碑列表
        gamma: 学习率衰减因子
        last_epoch: 上次更新的epoch
        step_frequency: 更新频率，'epoch'或'step'
    """
    
    def _build_internal_scheduler(self):
        """创建MultiStepLR调度器
        
        Returns:
            MultiStepLR调度器
        """
        milestones = self.config.get('milestones', [30, 60, 90])
        gamma = self.config.get('gamma', 0.1)
        last_epoch = self.config.get('last_epoch', -1)
        
        return lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=milestones,
            gamma=gamma,
            last_epoch=last_epoch
        )


@SchedulerRegistry.register('ExponentialLR')
class ExponentialLRWrapper(BaseLRSchedulerWrapper):
    """ExponentialLR包装器
    
    配置参数:
        gamma: 学习率衰减因子
        last_epoch: 上次更新的epoch
        step_frequency: 更新频率，'epoch'或'step'
    """
    
    def _build_internal_scheduler(self):
        """创建ExponentialLR调度器
        
        Returns:
            ExponentialLR调度器
        """
        gamma = self.config.get('gamma', 0.1)
        last_epoch = self.config.get('last_epoch', -1)
        
        return lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=gamma,
            last_epoch=last_epoch
        )

class BaseWarmupWrapper(BaseLRSchedulerWrapper):
    """预热调度器的基类
    
    提供共用的预热参数处理逻辑
    """
    
    def __init__(self, optimizer, config):
        # 获取预热参数（在调用super().__init__之前设置属性）
        self.step_frequency = config.get('step_frequency', 'epoch')
        
        # 获取预热参数
        if self.step_frequency == 'epoch':
            self.warmup_duration = config.get('warmup_epochs', 5)
        else:  # 'step'
            self.warmup_duration = config.get('warmup_steps', 1000)
            
        self.warmup_start_factor = config.get('warmup_start_factor', 0.1)
        
        # 调用父类的初始化方法
        super().__init__(optimizer, config)
        

@SchedulerRegistry.register('LinearWarmup')
class LinearWarmupWrapper(BaseWarmupWrapper):
    """线性预热调度器
    
    线性增加学习率，从初始值的一个小比例到初始值。
    
    配置参数:
        warmup_epochs: 预热的 epoch 数 (当 step_frequency='epoch')
        warmup_steps: 预热的 step 数 (当 step_frequency='step')
        warmup_start_factor: 初始学习率的比例，默认为 0.1
        step_frequency: 更新频率，'epoch' 或 'step'
    """
    
    def _build_internal_scheduler(self):
        """创建线性预热调度器
        
        Returns:
            使用LambdaLR实现的线性预热调度器
        """
        # 保存一个局部的引用以便在闭包中使用
        warmup_duration = self.warmup_duration
        warmup_start_factor = self.warmup_start_factor
        
        def lr_lambda(step):
            if step < warmup_duration:
                # 线性预热：从 warmup_start_factor 到 1.0
                return warmup_start_factor + (1.0 - warmup_start_factor) * (
                    step / warmup_duration
                )
            else:
                # 预热完成，保持为初始学习率
                return 1.0
                
        return lr_scheduler.LambdaLR(self.optimizer, lr_lambda)


@SchedulerRegistry.register('ConstantWarmup')
class ConstantWarmupWrapper(BaseWarmupWrapper):
    """常数预热调度器
    
    在预热阶段保持一个较小的常数学习率，之后恢复到初始学习率。
    
    配置参数:
        warmup_epochs: 预热的 epoch 数 (当 step_frequency='epoch')
        warmup_steps: 预热的 step 数 (当 step_frequency='step')
        warmup_start_factor: 初始学习率的比例，默认为 0.1
        step_frequency: 更新频率，'epoch' 或 'step'
    """
    
    def _build_internal_scheduler(self):
        """创建常数预热调度器
        
        Returns:
            使用LambdaLR实现的常数预热调度器
        """
        # 保存一个局部的引用以便在闭包中使用
        warmup_duration = self.warmup_duration
        warmup_start_factor = self.warmup_start_factor
        
        def lr_lambda(step):
            if step < warmup_duration:
                # 常数预热：保持 warmup_start_factor
                return warmup_start_factor
            else:
                # 预热完成，恢复到初始学习率
                return 1.0
                
        return lr_scheduler.LambdaLR(self.optimizer, lr_lambda)


@SchedulerRegistry.register('ExponentialWarmup')
class ExponentialWarmupWrapper(BaseWarmupWrapper):
    """指数预热调度器
    
    指数增加学习率，从初始值的一个小比例到初始值。
    
    配置参数:
        warmup_epochs: 预热的 epoch 数 (当 step_frequency='epoch')
        warmup_steps: 预热的 step 数 (当 step_frequency='step')
        warmup_start_factor: 初始学习率的比例，默认为 0.1
        warmup_exponent: 指数增长的幂，默认为 2.0
        step_frequency: 更新频率，'epoch' 或 'step'
    """
    
    def __init__(self, optimizer, config):
        # 在调用基类初始化之前设置特定属性
        self.warmup_exponent = config.get('warmup_exponent', 2.0)
        super().__init__(optimizer, config)
    
    def _build_internal_scheduler(self):
        """创建指数预热调度器
        
        Returns:
            使用LambdaLR实现的指数预热调度器
        """
        # 保存一个局部的引用以便在闭包中使用
        warmup_duration = self.warmup_duration
        warmup_start_factor = self.warmup_start_factor
        warmup_exponent = self.warmup_exponent
        
        def lr_lambda(step):
            if step < warmup_duration:
                # 指数预热：从 warmup_start_factor 到 1.0，使用指数曲线
                # 计算当前进度的百分比
                progress = step / warmup_duration
                # 应用指数函数：progress^exponent
                scaled_progress = progress ** warmup_exponent
                # 线性插值得到最终的学习率因子
                return warmup_start_factor + (1.0 - warmup_start_factor) * scaled_progress
            else:
                # 预热完成，保持为初始学习率
                return 1.0
                
        return lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

@SchedulerRegistry.register('CompositeLR')
class CompositeSchedulerWrapper(BaseLRSchedulerWrapper):
    """复合调度器包装器
    
    按顺序组合多个调度器阶段，每个阶段有指定的持续时间。
    这是对 PyTorch SequentialLR 的封装。
    
    配置参数:
        phases: 一个列表，每个元素是一个阶段的配置字典，包含：
            - scheduler_name: (str) 调度器注册名称
            - config: (dict) 该调度器的配置
            - duration: (int) 该阶段持续的epoch或者step数
    """
    
    def __init__(self, optimizer, config):
        super().__init__(optimizer, config)

    def _build_internal_scheduler(self):
        """创建SequentialLR调度器
        
        Returns:
            SequentialLR调度器
        """
        phases_config = self.config.get('phases', [])
        if not phases_config:
            raise ValueError("CompositeLR 需要定义 'phases' 列表")

        internal_schedulers = []
        milestones = []
        current_milestone = 0

        for i, phase in enumerate(phases_config):
            scheduler_name = phase.get('scheduler_name')
            phase_config = phase.get('config', {})
            duration = phase.get('duration')

            if not scheduler_name or duration is None:
                raise ValueError(f"Phase {i} 必须包含 'scheduler_name' 和 'duration'")
                
            if duration <= 0:
                 raise ValueError(f"Phase {i} 的 duration 必须是正整数")

            # 获取调度器包装器类
            try:
                scheduler_wrapper_cls = SchedulerRegistry.get(scheduler_name)
            except ValueError:
                raise ValueError(f"未知的调度器名称: {scheduler_name}")
            
            # 确保步频匹配当前调度器
            if 'step_frequency' not in phase_config:
                phase_config['step_frequency'] = self.step_frequency
            
            # 创建包装器实例并获取内部调度器
            # 注意：这里我们传递 self.optimizer 和 phase_config 给包装器
            # 包装器内部会创建对应的 Pytorch scheduler
            wrapper_instance = scheduler_wrapper_cls(self.optimizer, phase_config)
            internal_schedulers.append(wrapper_instance.scheduler)

            # 计算里程碑，最后一个阶段不需要里程碑
            if i < len(phases_config) - 1:
                current_milestone += duration
                milestones.append(current_milestone)

        if not internal_schedulers:
             raise ValueError("未能从 'phases' 配置中构建任何调度器")
             
        # SequentialLR 需要至少一个 scheduler
        # milestones 的长度应该比 schedulers 少一个
        if len(milestones) != len(internal_schedulers) - 1 and len(internal_schedulers) > 1:
             # 这个错误理论上不应该发生，除非phases配置只有一个元素
             import logging
             logger = logging.getLogger(__name__)
             logger.warning(f"Milestones ({len(milestones)}) 和 Schedulers ({len(internal_schedulers)}) 数量不匹配，可能导致意外行为。")
             # 修正 milestones 长度，尽管 SequentialLR 可能内部会处理
             milestones = milestones[:len(internal_schedulers)-1]


        return SequentialLR(
            self.optimizer,
            schedulers=internal_schedulers,
            milestones=milestones,
            last_epoch=self.config.get('last_epoch', -1) # 允许从配置文件加载last_epoch
        ) 