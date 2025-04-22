import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union, List, Optional

from torchvision.models import vgg16, VGG16_Weights
from .base_model import BaseModel
from .registry import ModelRegistry
from ..utils import get_logger

@ModelRegistry.register('vgg16')
class VGG16(BaseModel):
    """VGG16模型用于猫狗二分类任务"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 从配置获取参数
        self.num_classes = config.get('num_classes', 2)
        self.use_pretrained = config.get('use_pretrained', True)
        self.feature_extract = config.get('feature_extract', True)  # 默认只训练分类器
        # 然后在__init__方法中替换现有的模型创建代码
        if self.use_pretrained:
            weights = VGG16_Weights.IMAGENET1K_V1  # 或使用 VGG16_Weights.DEFAULT
        else:
            weights = None
        self.model = vgg16(weights=weights)
        # 如果只训练分类器部分，冻结特征提取器
        if self.feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # 构建新的全连接层
        self.model.classifier = nn.Sequential(
            nn.Linear(25088, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(100, self.num_classes)
        )
        
        # 将特征提取器和分类器分开保存
        self.features = self.model.features
        self.classifier = self.model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入图像张量，形状为[batch_size, channels, height, width]
            
        Returns:
            分类预测结果
        """
        # 特征提取
        x = self.features(x)
        
        # 展平特征图
        x = torch.flatten(x, 1)
        
        # 分类器
        x = self.classifier(x)
        
        return x
    
    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            outputs: 模型输出，形状为[batch_size, num_classes]
            targets: 目标标签，形状为[batch_size]
            
        Returns:
            损失值
        """
        # 对于二分类任务，使用交叉熵损失
        return F.cross_entropy(outputs, targets)
    
    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """执行前向传播并保存中间特征
        
        Args:
            x: 输入图像张量
            
        Returns:
            模型输出和中间特征的字典
        """
        features = {}
        
        # 分阶段记录特征
        # VGG16有5个阶段，每个阶段包含几个卷积层和一个池化层
        feature_blocks = []
        current_block = []
        
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                feature_blocks.append(nn.Sequential(*current_block, layer))
                current_block = []
            else:
                current_block.append(layer)
        
        # 执行分阶段特征提取
        curr_x = x
        for i, block in enumerate(feature_blocks):
            curr_x = block(curr_x)
            features[f'block{i+1}'] = curr_x
        
        # 展平特征图
        flat_features = torch.flatten(curr_x, 1)
        features['flat'] = flat_features
        
        # 分类器各层特征
        for i, layer in enumerate(self.classifier):
            flat_features = layer(flat_features)
            features[f'fc{i+1}'] = flat_features
        
        return flat_features, features
    
    def visualize_features(self, visualizer, input_tensor: torch.Tensor, global_step: int) -> None:
        """使用指定的可视化器可视化模型的特征图
        
        Args:
            visualizer: 可视化器对象
            input_tensor: 输入张量
            global_step: 全局步数
        """
        # 确保输入是批次形式
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        # 首先可视化输入图像
        visualizer.add_image("input/original", input_tensor[0], global_step)
        
        # 执行前向传播并获取特征
        with torch.no_grad():
            _, features = self.forward_with_features(input_tensor)
            
        # 针对每个卷积块特征图进行可视化
        for block_name, feature_map in features.items():
            # 只可视化卷积块的输出
            if block_name.startswith('block'):
                # 选择第一个样本的特征图
                feature = feature_map[0]
                num_channels = feature.shape[0]
                
                # 为了更好的可视化，只取一部分通道
                channels_to_show = min(16, num_channels)
                
                # 创建网格图像来显示多个通道
                grid_image = visualizer.make_grid(
                    feature[:channels_to_show].unsqueeze(1),
                    normalize=True,
                    nrow=4
                )
                
                visualizer.add_image(f"features/{block_name}", grid_image, global_step)