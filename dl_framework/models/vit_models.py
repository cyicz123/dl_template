import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import timm
import os

try:
    from .base_model import BaseModel
    from .registry import ModelRegistry
    from ..utils.logger import get_logger
    from ..utils.checkpoint import load_checkpoint
except ImportError:
    # 在直接运行此文件时的导入方式
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from dl_framework.models.base_model import BaseModel
    from dl_framework.models.registry import ModelRegistry


@ModelRegistry.register('image_vit')
class ImageViT(BaseModel):
    """图像数据的Vision Transformer模型
    
    使用Vision Transformer处理图像数据，返回包含CLS向量在内的patch feature embedding序列
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 模型配置
                - model_name: timm模型名称，默认为'vit_base_patch16_224'
                - pretrained: 是否使用预训练权重，默认为True
                - in_chans: 输入通道数，默认为3（RGB图像）
                - img_size: 输入图像大小，默认为(224, 224)
                - num_classes: 类别数，若为0则不使用分类头
                - return_features: 是否返回特征，默认为True
                - dropout_rate: Dropout率，默认为0.0
        """
        super(ImageViT, self).__init__(config)
        
        # 获取配置参数
        self.model_name = config.get('model_name', 'vit_base_patch16_224')
        self.pretrained = config.get('pretrained', True)
        self.in_chans = config.get('in_chans', 3)  # 图像通常是三通道RGB
        self.img_size = config.get('img_size', (224, 224))
        self.num_classes = config.get('num_classes', 2)  # 0表示不使用分类头
        self.return_features = config.get('return_features', False)
        self.dropout_rate = config.get('dropout_rate', 0.0)
        
        # 创建Vision Transformer模型
        self.model = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            in_chans=self.in_chans,
            img_size=self.img_size,
            num_classes=self.num_classes,
            drop_rate=self.dropout_rate,
        )
        
        # 保存模型的维度信息，用于后续处理
        if hasattr(self.model, 'embed_dim'):
            self.embed_dim = self.model.embed_dim
        else:
            # 对于不同的ViT变体，可能需要适配不同的属性名
            self.embed_dim = self.model.num_features

        if self.num_classes > 0:
          self.model.head = nn.Linear(self.embed_dim, self.num_classes)     

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入图像数据，形状为 [batch_size, channels, height, width]
            
        Returns:
            如果return_features为True，返回包含CLS向量在内的patch feature embedding序列，
            形状为 [batch_size, num_patches+1, embed_dim]；
            否则，返回分类结果，形状为 [batch_size, num_classes]
        """
        # 提取特征
        features = self.model.forward_features(x)
        
        if self.return_features:
            return features
        
        # 如果不返回特征，则使用分类头（如果有）
        if self.num_classes > 0:
          return self.model.head(features[:, 0])
        else:
          return features[:, 0]

    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            outputs: 模型输出
            targets: 目标值
            
        Returns:
            损失值
        """
        # 对于二分类任务，使用交叉熵损失
        return F.cross_entropy(outputs, targets)