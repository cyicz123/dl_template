class SchedulerRegistry:
    """学习率调度器注册器，用于注册和获取调度器类"""
    _schedulers = {}
    
    @classmethod
    def register(cls, name):
        """注册调度器类
        
        Args:
            name: 调度器名称
            
        Returns:
            装饰器函数
        """
        def wrapper(scheduler_class):
            cls._schedulers[name] = scheduler_class
            return scheduler_class
        return wrapper
    
    @classmethod
    def get(cls, name):
        """获取调度器类
        
        Args:
            name: 调度器名称
            
        Returns:
            调度器类
        
        Raises:
            ValueError: 如果调度器未注册
        """
        if name not in cls._schedulers:
            raise ValueError(f"未注册的调度器: {name}")
        return cls._schedulers[name]
    
    @classmethod
    def list(cls):
        """列出所有已注册的调度器
        
        Returns:
            已注册调度器名称字典，键为名称，值为类
        """
        return cls._schedulers
    
    @classmethod
    def create(cls, config, optimizer):
        """根据配置创建调度器实例
        
        Args:
            config: 调度器配置
            optimizer: 优化器实例
            
        Returns:
            调度器实例
        """
        scheduler_type = config.get('type')
        if not scheduler_type:
            raise ValueError("调度器类型未指定")
            
        scheduler_class = cls.get(scheduler_type)
        return scheduler_class(optimizer, config) 