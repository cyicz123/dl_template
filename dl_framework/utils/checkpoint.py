import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Union

def save_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """保存模型检查点
    
    Args:
        model: 模型
        path: 保存路径
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        metrics: 指标
    """
    checkpoint = {
        'model': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存检查点
    torch.save(checkpoint, path)

def load_checkpoint(
    model: nn.Module,
    path: str,
    device: Union[str, torch.device],
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """加载模型检查点
    
    Args:
        model: 模型
        path: 检查点路径
        device: 设备
        optimizer: 优化器
        scheduler: 学习率调度器
        
    Returns:
        检查点数据
    """
    # 加载检查点
    checkpoint = torch.load(path, map_location=device)
    
    # 加载模型参数
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    # 加载优化器参数
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 加载调度器参数
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint 