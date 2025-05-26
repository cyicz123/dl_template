import torch
import numpy as np
from typing import Dict, List, Any, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .base_metric import BaseMetric
from .registry import MetricRegistry


@MetricRegistry.register('MSE')
class MeanSquaredError(BaseMetric):
    """均方误差评估指标"""
    
    default_prefix = 'mse'

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
            
        # 展平以处理多维输出
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
            
        self.results.append({
            'predictions': predictions,
            'targets': targets
        })

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """计算均方误差"""
        all_predictions = np.concatenate([res['predictions'] for res in results])
        all_targets = np.concatenate([res['targets'] for res in results])
        
        mse = mean_squared_error(all_targets, all_predictions)
        return {'mse': float(mse)}


@MetricRegistry.register('MAE')
class MeanAbsoluteError(BaseMetric):
    """平均绝对误差评估指标"""
    
    default_prefix = 'mae'

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
            
        # 展平以处理多维输出
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
            
        self.results.append({
            'predictions': predictions,
            'targets': targets
        })

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """计算平均绝对误差"""
        all_predictions = np.concatenate([res['predictions'] for res in results])
        all_targets = np.concatenate([res['targets'] for res in results])
        
        mae = mean_absolute_error(all_targets, all_predictions)
        return {'mae': float(mae)}


@MetricRegistry.register('RMSE')
class RootMeanSquaredError(BaseMetric):
    """均方根误差评估指标"""
    
    default_prefix = 'rmse'

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
            
        # 展平以处理多维输出
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
            
        self.results.append({
            'predictions': predictions,
            'targets': targets
        })

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """计算均方根误差"""
        all_predictions = np.concatenate([res['predictions'] for res in results])
        all_targets = np.concatenate([res['targets'] for res in results])
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        return {'rmse': float(rmse)}


@MetricRegistry.register('R2Score')
class R2Score(BaseMetric):
    """R²分数评估指标"""
    
    default_prefix = 'r2'

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
            
        # 展平以处理多维输出
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
            
        self.results.append({
            'predictions': predictions,
            'targets': targets
        })

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """计算R²分数"""
        all_predictions = np.concatenate([res['predictions'] for res in results])
        all_targets = np.concatenate([res['targets'] for res in results])
        
        r2 = r2_score(all_targets, all_predictions)
        return {'r2_score': float(r2)} 