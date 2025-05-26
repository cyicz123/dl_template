from .base_metric import BaseMetric
from .evaluator import Evaluator
from .registry import MetricRegistry

# 导入所有指标以确保它们被注册
from .classification import (
  Accuracy, Precision, Recall, F1Score, ConfusionMatrix
)
from .regression import (
  MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, R2Score
)

__all__ = [
    'BaseMetric',
    'Evaluator', 
    'MetricRegistry',
    'Accuracy', 'Precision', 'Recall', 'F1Score', 'ConfusionMatrix',
    'MeanSquaredError', 'MeanAbsoluteError', 'RootMeanSquaredError', 'R2Score'
]