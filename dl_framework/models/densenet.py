import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Union, Optional
import math

from .base_model import BaseModel
from .registry import ModelRegistry

class _DenseLayer(nn.Module):
    """DenseNet层实现"""
    
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float):
        """初始化
        
        Args:
            num_input_features: 输入特征通道数
            growth_rate: 每层输出的特征通道数
            bn_size: bottleneck层的扩展因子
            drop_rate: dropout比率
        """
        super(_DenseLayer, self).__init__()
        
        # Bottleneck层，用于减少计算量
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, 
                               kernel_size=1, stride=1, bias=False)
        
        # 主卷积层
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        
        # dropout
        self.drop_rate = drop_rate
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            inputs: 前面所有层的输出特征列表
            
        Returns:
            当前层的输出特征
        """
        # 将前面所有层的特征在通道维度上连接起来
        concated_features = torch.cat(inputs, 1)
        
        # Bottleneck层
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        
        # 主卷积层
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        
        # 应用dropout
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
            
        return new_features


class _DenseBlock(nn.ModuleDict):
    """DenseNet块实现，包含多个DenseLayer"""
    
    def __init__(self, num_layers: int, num_input_features: int, 
                bn_size: int, growth_rate: int, drop_rate: float):
        """初始化
        
        Args:
            num_layers: 块中的层数
            num_input_features: 输入特征通道数
            bn_size: bottleneck层的扩展因子
            growth_rate: 每层输出的特征通道数
            drop_rate: dropout比率
        """
        super(_DenseBlock, self).__init__()
        
        # 添加多个DenseLayer
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module('denselayer%d' % (i + 1), layer)
    
    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            init_features: 初始输入特征
            
        Returns:
            区块输出特征
        """
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
            
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    """过渡层，用于降维和下采样"""
    
    def __init__(self, num_input_features: int, num_output_features: int):
        """初始化
        
        Args:
            num_input_features: 输入特征通道数
            num_output_features: 输出特征通道数
        """
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                             kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


@ModelRegistry.register('densenet')
class DenseNet(BaseModel):
    """DenseNet模型实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 模型配置
        """
        super().__init__(config)
        
        # 从配置中获取参数
        self.in_channels = config.get('in_channels', 3)
        self.num_classes = config.get('num_classes', 1000)
        self.growth_rate = config.get('growth_rate', 32)  # 每层增长的特征数
        self.block_config = config.get('block_config', (6, 12, 24, 16))  # 每个块的层数
        self.bn_size = config.get('bn_size', 4)  # Bottleneck层扩展因子
        self.drop_rate = config.get('drop_rate', 0)  # dropout比率
        self.compression = config.get('compression', 0.5)  # 压缩因子
        self.num_init_features = config.get('num_init_features', 64)  # 初始特征数
        
        # 初始卷积层
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_init_features, 
                     kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # DenseBlock和Transition层
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.block_config):
            # 添加DenseBlock
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=self.bn_size,
                growth_rate=self.growth_rate,
                drop_rate=self.drop_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * self.growth_rate
            
            # 除了最后一个block外，添加Transition层
            if i != len(self.block_config) - 1:
                num_output_features = int(math.floor(num_features * self.compression))
                trans = _Transition(num_input_features=num_features,
                                  num_output_features=num_output_features)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_output_features
        
        # 最后的BN和分类器
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # 全局池化和分类器
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, self.num_classes)
        
        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
        # 特征存储
        self.stored_features = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [N, C, H, W]
            
        Returns:
            输出张量，形状为 [N, num_classes]
        """
        # 特征提取
        features = self.features(x)
        
        # 全局池化
        out = self.global_pool(features)
        out = torch.flatten(out, 1)
        
        # 分类器
        out = self.classifier(out)
        
        return out
    
    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """执行前向传播并保存中间特征
        
        Args:
            x: 输入张量，形状为 [N, C, H, W]
            
        Returns:
            元组，包含输出张量和特征字典
        """
        features_dict = {}
        features_dict['input'] = x
        
        # 保存各层输出
        curr_x = x
        
        # 初始特征提取
        for idx, module in enumerate(self.features):
            curr_x = module(curr_x)
            layer_name = f"{type(module).__name__}_{idx}"
            features_dict[layer_name] = curr_x
        
        # 全局池化
        pooled = self.global_pool(curr_x)
        features_dict['global_pool'] = pooled
        
        # 展平
        flattened = torch.flatten(pooled, 1)
        features_dict['flattened'] = flattened
        
        # 分类
        output = self.classifier(flattened)
        features_dict['output'] = output
        
        # 保存所有特征
        self.stored_features = features_dict
        
        return output, features_dict
    
    def visualize_features(self, visualizer, input_tensor: torch.Tensor, global_step: int) -> None:
        """使用指定的可视化器可视化模型的特征图
        
        Args:
            visualizer: 可视化器实例，需要实现add_feature_maps方法
            input_tensor: 输入张量，形状为 [N, C, H, W]
            global_step: 全局步数
        """
        # 确保输入是批次形式
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        # 可视化输入图像
        visualizer.add_image("input/original", input_tensor[0], global_step)
        
        # 执行前向传播并获取特征
        with torch.no_grad():
            _, features = self.forward_with_features(input_tensor)
            
        # 可视化每个卷积层和过渡层的特征图
        for layer_name, feature_map in features.items():
            if 'Conv2d' in layer_name or 'BatchNorm2d' in layer_name or 'transition' in layer_name:
                if len(feature_map.shape) == 4:  # 确保是特征图 [N, C, H, W]
                    # 移动到CPU进行可视化
                    feature_map = feature_map.cpu()
                    
                    # 最多可视化64个通道
                    num_channels = min(feature_map.size(1), 64)
                    if num_channels > 0:
                        # 选择前num_channels个通道
                        selected_channels = feature_map[0, :num_channels]
                        
                        # 添加特征图直方图
                        visualizer.add_histogram(f"features/{layer_name}/histogram", 
                                               feature_map, global_step)
                        
                        # 如果可视化器支持添加特征图网格
                        if hasattr(visualizer, 'add_images_grid'):
                            channels = []
                            for i in range(num_channels):
                                # 获取单个通道并归一化到[0,1]进行可视化
                                channel = selected_channels[i].unsqueeze(0)  # [1, H, W]
                                
                                # 如果特征图数值范围很小，缩放以增强对比度
                                if channel.max() - channel.min() > 0:
                                    channel = (channel - channel.min()) / (channel.max() - channel.min())
                                
                                # 扩展到3通道
                                channel = channel.expand(3, -1, -1)  # [3, H, W]
                                channels.append(channel)
                            
                            if channels:
                                channel_tensor = torch.stack(channels)  # [num_channels, 3, H, W]
                                visualizer.add_images_grid(f"features_grid/{layer_name}", 
                                                        channel_tensor, global_step, nrow=8)
    
    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            outputs: 模型输出，形状为 [N, num_classes]
            targets: 目标张量，形状为 [N]
            
        Returns:
            损失值
        """
        # 尝试使用配置中指定的损失函数
        if hasattr(self, 'loss') and self.loss is not None:
            return super().get_loss(outputs, targets)
        
        # 默认使用交叉熵损失
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(outputs, targets)
    