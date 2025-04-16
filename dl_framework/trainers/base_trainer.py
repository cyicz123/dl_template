import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Callable, Optional, Union

from ..models.registry import ModelRegistry
from ..datasets.registry import DatasetRegistry
from ..losses.registry import LossRegistry
from ..utils.logger import get_logger
import logging
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..visualization.base_visualizer import BaseVisualizer
from ..hooks.base_hook import BaseHook
from ..hooks.registry import HookRegistry
from ..visualization.registry import VisualizerRegistry

class BaseTrainer:
    """基础训练器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 训练配置
        """
        self.config = config
        
        # 实验目录设置 - 所有输出都统一放在这个目录下
        self.experiment_dir = config.get('experiment_dir', 'experiments/default')
        
        # 设置标准子目录
        self.checkpoints_dir = config.get('checkpoints_dir', os.path.join(self.experiment_dir, 'checkpoints'))
        self.logs_dir = config.get('logs_dir', os.path.join(self.experiment_dir, 'logs'))
        self.visualization_dir = config.get('visualization_dir', os.path.join(self.experiment_dir, 'visualization'))
        
        # 确保所有目录存在
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        self.logger = self._build_logger()
        self.device = self._get_device()
        
        # 设置随机种子以确保可重复性
        self._set_seed(config.get('seed', 42))
        
        # 构建模型和数据加载器
        self.model = self._build_model()
        self.train_loader, self.val_loader = self._build_data_loaders()
        
        # 构建损失函数（如果在配置中指定）
        self.loss = self._build_loss()
        
        # 构建优化器和学习率调度器
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # 构建可视化器
        self.visualizer = self._build_visualizer()
        
        # 服务容器 - 存储所有可用服务
        self._services: Dict[str, Any] = {}
        
        # 注册默认服务
        self._register_default_services()
        
        # 构建钩子
        self.hooks = [] 
        self._build_hooks()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')  # 用于模型检查点保存

        # 早停法相关参数
        self.early_stopping_counter = 0
        self.best_early_stopping_metric = float('inf')
        
        # 加载检查点（如果指定）
        resume_path = config.get('resume')
        if resume_path and os.path.exists(resume_path):
            checkpoint = load_checkpoint(
                self.model,
                resume_path,
                self.device,
                optimizer=self.optimizer,
                scheduler=self.scheduler
            )
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_metric = checkpoint.get('metrics', {}).get('loss', float('inf'))
            self.best_early_stopping_metric = self.best_metric  # 初始化早停指标
            self.logger.info(f"从检查点恢复训练: {resume_path}")
            self.logger.info(f"从epoch {self.current_epoch} 继续训练")
    
    def _register_default_services(self) -> None:
        """注册默认服务"""
        # 注册训练器自身
        self.register_service("trainer", self)
        
        # 注册配置
        self.register_service("config", self.config)
        
        # 注册模型
        self.register_service("model", self.model)
        
        # 注册数据加载器
        self.register_service("train_loader", self.train_loader)
        self.register_service("val_loader", self.val_loader)
        
        # 注册优化器和调度器
        self.register_service("optimizer", self.optimizer)
        self.register_service("scheduler", self.scheduler)
        
        # 注册设备
        self.register_service("device", self.device)
        
        # 注册日志器
        self.register_service("logger", self.logger)
        
        # 注册目录路径
        self.register_service("experiment_dir", self.experiment_dir)
        self.register_service("checkpoints_dir", self.checkpoints_dir)
        self.register_service("logs_dir", self.logs_dir)
        self.register_service("visualization_dir", self.visualization_dir)
        
        # 注册可视化器(如果存在)
        if self.visualizer is not None:
            self.register_service("visualizer", self.visualizer)
        
        # 注册损失函数（如果存在）
        if hasattr(self, 'loss') and self.loss is not None:
            self.register_service("loss", self.loss)
    
    def register_service(self, service_name: str, service: Any) -> None:
        """注册服务
        
        Args:
            service_name: 服务名称，用于Hook获取服务
            service: 服务实例
        """
        self._services[service_name] = service
        self.logger.debug(f"服务已注册: {service_name}")
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """获取服务
        
        Args:
            service_name: 服务名称
            
        Returns:
            服务实例，如果不存在则返回None
        """
        return self._services.get(service_name)
    
    def register_hook(self, hook: BaseHook) -> None:
        """注册钩子
        
        Args:
            hook: 钩子实例
        """
        # 注入所有当前可用的服务
        for service_name, service in self._services.items():
            hook.register_service(service_name, service)
        
        # 将钩子添加到列表中
        self.hooks.append(hook)
        self.logger.info(f"钩子已注册: {hook.name} ({hook.__class__.__name__})")
    
    def _build_logger(self) -> logging.Logger:
        """构建日志记录器
        
        Returns:
            日志记录器
        """
        return get_logger(self.__class__.__module__ + '.' + self.__class__.__name__)
    
    def _get_device(self) -> torch.device:
        """获取设备
        
        Returns:
            torch设备
        """
        device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _set_seed(self, seed: int) -> None:
        """设置随机种子
        
        Args:
            seed: 随机种子
        """
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _build_model(self) -> nn.Module:
        """构建模型
        
        Returns:
            模型
        """
        model_config = self.config.get('model', {}).get('model', {})
        model_type = model_config.get('type')
        
        if not model_type:
            raise ValueError("模型类型未指定")
            
        model_class = ModelRegistry.get(model_type)
        model = model_class(model_config)
        
        # 将模型移动到设备
        model = model.to(self.device)
        
        return model
    
    def _build_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """构建数据加载器
        
        Returns:
            训练数据加载器和验证数据加载器
        """
        dataset_config = self.config.get('dataset', {}).get('dataset', {})
        dataset_type = dataset_config.get('type')
        
        if not dataset_type:
            raise ValueError("数据集类型未指定")
            
        dataset_class = DatasetRegistry.get(dataset_type)
        
        # 创建训练和验证数据集
        train_dataset = dataset_class(dataset_config, is_training=True)
        val_dataset = dataset_class(dataset_config, is_training=False)
        
        # 配置数据加载器
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_dataset.get_collate_fn(),
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=val_dataset.get_collate_fn(),
        )
        
        return train_loader, val_loader
    
    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器
        
        Returns:
            优化器
        """
        optimizer_config = self.config.get('training', {}).get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam').lower()
        lr = float(optimizer_config.get('lr', 0.001))
        weight_decay = float(optimizer_config.get('weight_decay', 0))
        
        # 获取模型可训练参数
        params = self.model.get_trainable_parameters()
        
        # 创建优化器
        if optimizer_type == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = float(optimizer_config.get('momentum', 0.9))
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """构建学习率调度器
        
        Returns:
            学习率调度器
        """
        scheduler_config = self.config.get('training', {}).get('scheduler', {})
        if not scheduler_config:
            return None
            
        scheduler_type = scheduler_config.get('type', '').lower()
        
        if not scheduler_type:
            return None
            
        if scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 10)
            gamma = scheduler_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 100)
            eta_min = scheduler_config.get('eta_min', 0)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == 'plateau':
            factor = scheduler_config.get('factor', 0.1)
            patience = scheduler_config.get('patience', 10)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    def _build_visualizer(self) -> Optional[BaseVisualizer]:
        """构建可视化器
        
        Returns:
            可视化器
        """
        # 检查配置中是否有visualization部分
        vis_config = self.config.get('visualization')
        if not vis_config:
            return None
            
        # 检查配置中是否指定了可视化器类型
        visualizer_type = None
        registry_keys = VisualizerRegistry.list().keys()
        registry_keys_lower = [key.lower() for key in registry_keys]
        
        for key in vis_config.keys():
            # 大小写不敏感的匹配
            if key.lower() in registry_keys_lower:
                # 找到对应的原始键名
                index = registry_keys_lower.index(key.lower())
                visualizer_type = list(registry_keys)[index]
                break
            
        # 如果找不到指定的可视化器类型，尝试查找默认类型
        if not visualizer_type:
            if 'tensorboard' in vis_config:
                # 查找大小写不敏感的tensorboard可视化器
                for key in registry_keys:
                    if key.lower() == 'tensorboard':
                        visualizer_type = key
                        break
                # 如果找不到，尝试使用默认名称
                if not visualizer_type:
                    visualizer_type = 'tensorboard'
            else:
                self.logger.warning("未找到支持的可视化器类型")
                return None
                
        try:
            visualizer_class = VisualizerRegistry.get(visualizer_type)
            visualizer = visualizer_class(self.config)
            self.logger.info(f"使用可视化器: {visualizer_type}")
            return visualizer
        except KeyError:
            self.logger.warning(f"未知的可视化器类型: {visualizer_type}")
        except Exception as e:
            self.logger.error(f"创建可视化器 {visualizer_type} 时出错: {e}")
            
        return None
    
    def _build_hooks(self) -> List[BaseHook]:
        """构建钩子
        
        Returns:
            钩子列表
        """
        # 检查配置中是否有hooks部分
        hooks_config = self.config.get('hooks', [])
        
        if not hooks_config:
            self.logger.info("未配置钩子")
            return
        
        # 使用Hook工厂创建钩子实例
        try:
            from ..hooks.base_hook import HookFactory
            hooks = HookFactory.create_hooks_from_config(hooks_config)
            
            # 注册服务到每个钩子
            for hook in hooks:
                self.register_hook(hook)
                
        except ImportError:
            raise ImportError("无法导入HookFactory，请确保hooks配置正确")
    
    def register_hooks_from_config(self) -> None:
        """从配置创建并注册所有钩子"""
        # 如果已经有钩子，先清空
        self.hooks = []
        
        # 重新构建钩子
        self.hooks = self._build_hooks()
    
    def _build_loss(self) -> Optional[nn.Module]:
        """构建损失函数
        
        优先级:
        1. 训练配置中的独立损失函数配置
        2. 模型配置中的损失函数配置（已在模型初始化时处理）
        
        如果都没有指定，返回None，使用模型内置的损失函数逻辑
        
        Returns:
            损失函数或None
        """
        # 检查训练配置中是否有损失函数配置
        loss_config = self.config.get('loss')
        if not loss_config:
            # 无损失函数配置，使用模型内置损失函数
            self.logger.info("使用模型内置损失函数")
            return None
            
        self.logger.info(f"从配置构建损失函数: {loss_config.get('type')}")
        return LossRegistry.create(loss_config)
    
    def train(self) -> nn.Module:
        """训练模型
        
        Returns:
            训练完成的模型
        """
        self.logger.info("开始训练...")
        
        # 获取训练配置
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 100)
        
        # 执行钩子的训练前方法
        for hook in self.hooks:
            hook.before_training(self.model)
        
        # 迭代每个epoch
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # 执行钩子的epoch前方法
            for hook in self.hooks:
                hook.before_epoch(epoch, self.model)
            
            # 训练一个epoch
            train_metrics = self._train_epoch()
            
            # 验证
            val_metrics = self._validate_epoch()
            
            # 更新学习率调度器
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', float('inf')))
                else:
                    self.scheduler.step()
            
            # 记录指标
            self.logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_metrics.get('loss', 0):.4f} "
                f"Val Loss: {val_metrics.get('loss', 0):.4f} "
                f"Val Metrics: {val_metrics}"
            )
            
            # 执行钩子的epoch后方法
            for hook in self.hooks:
                hook.after_epoch(epoch, self.model, val_metrics)
            
            # 保存检查点
            self._save_checkpoint(val_metrics)
            
            # 检查早停
            if self._should_stop_early(val_metrics):
                self.logger.info(f"早停触发，在epoch {epoch+1}停止训练")
                break
        
        # 执行钩子的训练后方法
        for hook in self.hooks:
            hook.after_training(self.model, val_metrics)
            hook.cleanup()
        
        # 关闭可视化器
        if self.visualizer is not None:
            self.visualizer.flush()
            self.visualizer.close()
        
        # 加载最佳模型
        best_checkpoint_path = os.path.join(self.checkpoints_dir, 'model_best.pth')
        if os.path.exists(best_checkpoint_path):
            load_checkpoint(self.model, best_checkpoint_path, self.device)
        
        self.logger.info("训练完成")
        return self.model
    
    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch
        
        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 执行钩子的step前方法
            for hook in self.hooks:
                hook.before_step(self.global_step, batch, self.model)
            
            # 准备数据
            if isinstance(batch, dict):
                # 如果batch是字典，每个键对应一个张量
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'targets'}
                #targets = batch['targets'].to(self.device)
                handlers = {
                'tensor': lambda x: x.to(self.device),
                'dict': lambda x: {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in x.items()}
                }
                targets = handlers[type(batch['targets']).__name__](batch['targets'])
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                # 如果batch是元组或列表，假设是(inputs, targets)
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            else:
                raise ValueError(f"不支持的batch格式: {type(batch)}")
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 计算损失
            if self.loss is not None:
                # 使用独立配置的损失函数
                loss = self.loss(outputs, targets)
            else:
                # 使用模型内置的损失计算逻辑
                loss = self.model.get_loss(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 更新全局步数
            self.global_step += 1
            
            # 累积损失
            total_loss += loss.item()
            
            # 使用可视化器记录每个batch的损失
            if self.visualizer is not None:
                self.visualizer.add_scalar('train/batch_loss', loss.item(), self.global_step)
            
            # 执行钩子的step后方法
            for hook in self.hooks:
                hook.after_step(self.global_step, batch, outputs, loss, self.model)
            
            # 打印进度
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Train Epoch: {self.current_epoch+1} "
                    f"[{batch_idx}/{num_batches} "
                    f"({100. * batch_idx / num_batches:.0f}%)] "
                    f"Loss: {loss.item():.6f}"
                )
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 使用可视化器记录每个epoch的平均损失
        if self.visualizer is not None:
            self.visualizer.add_scalar('train/epoch_loss', avg_loss, self.current_epoch)
        
        return {'loss': avg_loss}
    
    def _validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # 准备数据
                if isinstance(batch, dict):
                    # 如果batch是字典，每个键对应一个张量
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'targets'}
                    targets = batch['targets'].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # 如果batch是元组或列表，假设是(inputs, targets)
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    raise ValueError(f"不支持的batch格式: {type(batch)}")
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                if self.loss is not None:
                    # 使用独立配置的损失函数
                    loss = self.loss(outputs, targets)
                else:
                    # 使用模型内置的损失计算逻辑
                    loss = self.model.get_loss(outputs, targets)
                
                # 累积损失
                total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 计算其他指标（子类可以重写此方法添加更多指标）
        metrics = {'loss': avg_loss}
        
        # 使用可视化器记录验证指标
        if self.visualizer is not None:
            self.visualizer.add_scalar('val/loss', avg_loss, self.current_epoch)
            # 如果有其他指标，也记录它们
            for key, value in metrics.items():
                if key != 'loss':  # loss已经记录过了
                    self.visualizer.add_scalar(f'val/{key}', value, self.current_epoch)
        
        return metrics
    
    def _save_checkpoint(self, metrics: Dict[str, float]) -> None:
        """保存检查点
        
        Args:
            metrics: 性能指标
        """
        # 获取checkpoint配置
        checkpoint_config = self.config.get('training', {}).get('checkpoint', {})
        save_frequency = checkpoint_config.get('save_frequency', None)  # 如果未配置，则为None
        
        # 只有当显式声明了keep_num参数时才会使用它进行清理
        # 使用None作为默认值而不是5，以检测是否显式设置了该参数
        keep_num = checkpoint_config.get('keep_num', None)
        
        # 当前模型检查点路径
        latest_checkpoint_path = os.path.join(self.checkpoints_dir, 'model_latest.pth')
        
        # 根据频率保存周期性检查点 - 只有当显式配置了save_frequency时才执行
        should_save_periodic = (save_frequency is not None and (self.current_epoch + 1) % save_frequency == 0)
        
        # 只有在需要保存周期性检查点或是最佳模型时才实际保存文件
        if should_save_periodic:
            # 保存周期性检查点
            epoch_checkpoint_path = os.path.join(
                self.checkpoints_dir, f'model_epoch_{self.current_epoch + 1}.pth'
            )
            
            # 保存检查点
            save_checkpoint(
                self.model,
                epoch_checkpoint_path,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.current_epoch,
                metrics=metrics
            )
            self.logger.info(f"在epoch {self.current_epoch + 1}保存周期性检查点")
            
            # 如果当前系统支持硬链接，则创建一个硬链接作为最新检查点
            try:
                # 如果最新检查点文件已存在，先删除它
                if os.path.exists(latest_checkpoint_path):
                    os.remove(latest_checkpoint_path)
                
                # 创建硬链接，这样latest就指向最近保存的周期性检查点
                os.link(epoch_checkpoint_path, latest_checkpoint_path)
                self.logger.debug(f"创建了硬链接从 {epoch_checkpoint_path} 到 {latest_checkpoint_path}")
            except OSError as e:
                # 如果硬链接失败（例如，跨文件系统或其他限制），则直接复制文件
                self.logger.warning(f"创建硬链接失败，将直接复制文件: {e}")
                save_checkpoint(
                    self.model,
                    latest_checkpoint_path,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=self.current_epoch,
                    metrics=metrics
                )
            
            # 只有当显式设置了keep_num时才清理旧的检查点文件
            if keep_num is not None:
                self.logger.info(f"根据配置的keep_num={keep_num}清理旧检查点")
                self._cleanup_old_checkpoints(keep_num)

            if not os.path.exists(latest_checkpoint_path):
                # 如果没有保存周期性检查点，但最新检查点不存在，则创建一个
                save_checkpoint(
                    self.model,
                    latest_checkpoint_path,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=self.current_epoch,
                    metrics=metrics
                )
        
            # 判断是否需要保存最佳模型
            if self._is_best_checkpoint(metrics):
                best_checkpoint_path = os.path.join(self.checkpoints_dir, 'model_best.pth')
                save_checkpoint(
                    self.model,
                    best_checkpoint_path,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=self.current_epoch,
                    metrics=metrics
                )
                self.best_metric = metrics['loss']
                self.logger.info(f"保存最佳模型，验证损失: {metrics['loss']:.4f}")
    
    def _cleanup_old_checkpoints(self, keep_num: int) -> None:
        """清理旧的检查点文件，只保留最新的几个
        
        Args:
            keep_num: 要保留的检查点数量
        """
        # 获取所有周期性检查点
        checkpoints = []
        for filename in os.listdir(self.checkpoints_dir):
            if filename.startswith('model_epoch_') and filename.endswith('.pth'):
                try:
                    epoch = int(filename.split('_')[-1].split('.')[0])
                    checkpoints.append((epoch, os.path.join(self.checkpoints_dir, filename)))
                except:
                    continue
        
        # 按照epoch排序
        checkpoints.sort(reverse=True)
        
        # 删除多余的旧检查点
        if len(checkpoints) > keep_num:
            for _, checkpoint_path in checkpoints[keep_num:]:
                try:
                    os.remove(checkpoint_path)
                    self.logger.debug(f"删除旧检查点: {checkpoint_path}")
                except:
                    self.logger.warning(f"无法删除检查点: {checkpoint_path}")
    
    def _is_best_checkpoint(self, metrics: Dict[str, float]) -> bool:
        """判断是否需要保存最佳模型
        
        Args:
            metrics: 验证指标
            
        Returns:
            是否需要保存最佳模型
        """
        return metrics['loss'] < self.best_metric
    
    def _should_stop_early(self, metrics: Dict[str, float]) -> bool:
        """检查是否应该早停
        
        Args:
            metrics: 验证指标
            
        Returns:
            是否应该早停
        """
        early_stopping_config = self.config.get('training', {}).get('early_stopping', {})
        if not early_stopping_config:
            return False
            
        # 从配置中获取早停参数
        patience = early_stopping_config.get('patience', 10)
        min_delta = early_stopping_config.get('min_delta', 0.001)
        monitor = early_stopping_config.get('monitor', 'loss')  # 监控的指标，默认为loss
        mode = early_stopping_config.get('mode', 'min')  # 模式，min表示指标越小越好，max表示指标越大越好
        
        # 如果指定的指标不存在，直接返回False
        if monitor not in metrics:
            self.logger.warning(f"早停监控的指标 {monitor} 不存在于当前指标中")
            return False
        
        current_metric = metrics[monitor]
        
        # 判断是否需要更新最佳指标
        if mode == 'min':
            is_better = current_metric < self.best_early_stopping_metric - min_delta
        else:  # mode == 'max'
            is_better = current_metric > self.best_early_stopping_metric + min_delta
            
        if is_better:
            # 如果当前指标更好，重置计数器并更新最佳指标
            self.best_early_stopping_metric = current_metric
            self.early_stopping_counter = 0
            return False
        else:
            # 如果当前指标没有改善，增加计数器
            self.early_stopping_counter += 1
            self.logger.info(f"早停计数器: {self.early_stopping_counter}/{patience}")
            
            # 如果连续patience次没有改善，触发早停
            if self.early_stopping_counter >= patience:
                self.logger.info(f"早停触发: {monitor} 在 {patience} 个epoch中没有改善")
                return True
        
        return False
    
    def test(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """测试模型
        
        Args:
            test_loader: 测试数据加载器，如果为None则使用验证数据加载器
            
        Returns:
            测试指标字典
        """
        if test_loader is None:
            test_loader = self.val_loader
        
        self.model.eval()
        total_loss = 0
        num_batches = len(test_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # 准备数据
                if isinstance(batch, dict):
                    # 如果batch是字典，每个键对应一个张量
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'targets'}
                    #targets = batch['targets'].to(self.device)
                    handlers = {
                    'tensor': lambda x: x.to(self.device),
                    'dict': lambda x: {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in x.items()}
                    }
                    targets = handlers[type(batch['targets']).__name__](batch['targets'])
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # 如果batch是元组或列表，假设是(inputs, targets)
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    raise ValueError(f"不支持的batch格式: {type(batch)}")
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                if self.loss is not None:
                    loss = self.loss(outputs, targets)
                else:
                    loss = self.model.get_loss(outputs, targets)
                
                # 累积损失
                total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 计算其他指标（子类可以重写此方法添加更多指标）
        metrics = {'loss': avg_loss}
        
        self.logger.info(f"测试结果: {metrics}")
        
        return metrics 