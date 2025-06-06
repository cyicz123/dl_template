import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union, Optional
import torchvision.models as tv_models

from .base_model import BaseModel
from .registry import ModelRegistry
from ..utils.logger import get_logger


@ModelRegistry.register('googlenet')
class GoogleNet(BaseModel):
    """基于 torchvision 的 GoogleNet 模型包装器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化 GoogleNet 模型
        
        Args:
            config: 模型配置，包含以下参数：
                - num_classes: 分类数量，默认为 1000
                - aux_logits: 是否使用辅助分类器，默认为 True
                - dropout: dropout 概率，默认为 0.2
                - pretrained: 是否使用预训练权重，默认为 False
                - weights: 指定预训练权重类型，可选 'IMAGENET1K_V1' 或 'DEFAULT'
        """
        super(GoogleNet, self).__init__(config)
        self.logger = get_logger(__name__)
        
        # 从配置中获取参数
        self.num_classes = config.get('num_classes', 1000)
        self.aux_logits = config.get('aux_logits', True)
        self.dropout_prob = config.get('dropout', 0.2)
        self.pretrained = config.get('pretrained', False)
        self.weights = config.get('weights', None)
        self.freeze = config.get('freeze_backbone', False)
        
        # 创建 torchvision 的 GoogleNet 模型
        self._create_model()
    
    def _create_model(self):
        """创建 torchvision GoogleNet 模型"""
        # 确定权重类型
        weights = None
        if self.pretrained or self.weights:
            if self.weights:
                if self.weights.upper() in ['DEFAULT', 'IMAGENET1K_V1']:
                    weights = tv_models.GoogLeNet_Weights.IMAGENET1K_V1
                else:
                    # 尝试直接使用字符串
                    weights = self.weights
            else:
                weights = tv_models.GoogLeNet_Weights.DEFAULT
        
        # 创建模型
        self.model = tv_models.googlenet(
            weights=weights,
            aux_logits=self.aux_logits,
            dropout=self.dropout_prob,
            num_classes=1000  # 先创建标准模型
        )
        # 如果需要修改分类层
        if self.num_classes != 1000:
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            
            # 如果有辅助分类器，也需要修改
            if self.aux_logits:
                if hasattr(self.model, 'aux1') and self.model.aux1 is not None:
                    self.model.aux1.fc2 = nn.Linear(self.model.aux1.fc2.in_features, self.num_classes)
                if hasattr(self.model, 'aux2') and self.model.aux2 is not None:
                    self.model.aux2.fc2 = nn.Linear(self.model.aux2.fc2.in_features, self.num_classes)
        
        if self.freeze:
            self.freeze_backbone()
        
        # 打印加载信息
        if weights:
            self.logger.info(f"成功加载 GoogleNet 预训练权重: {weights}")
            if self.num_classes != 1000:
                self.logger.info(f"分类层已调整为 {self.num_classes} 类")
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [N, C, H, W]
            
        Returns:
            如果是训练模式且使用辅助分类器，返回 (main_output, aux1_output, aux2_output)
            否则返回 main_output
        """
        return self.model(x)
    
    def get_loss(self, outputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
                targets: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            outputs: 模型输出
            targets: 目标张量
            
        Returns:
            损失值
        """
        # 首先尝试使用配置中指定的损失函数
        if hasattr(self, 'loss') and self.loss is not None:
            if isinstance(outputs, tuple):
                # 如果有辅助输出，只使用主输出计算损失
                main_output = outputs[0]
                return super().get_loss(main_output, targets)
            else:
                return super().get_loss(outputs, targets)
        
        # 使用默认的交叉熵损失
        criterion = nn.CrossEntropyLoss()
        
        if isinstance(outputs, tuple):
            # 训练时有辅助分类器输出
            main_output, aux1_output, aux2_output = outputs
            
            # 主损失
            main_loss = criterion(main_output, targets)
            
            # 辅助损失
            aux1_loss = criterion(aux1_output, targets)
            aux2_loss = criterion(aux2_output, targets)
            
            # 总损失 = 主损失 + 0.3 * 辅助损失1 + 0.3 * 辅助损失2
            total_loss = main_loss + 0.3 * aux1_loss + 0.3 * aux2_loss
            
            return total_loss
        else:
            # 推理时只有主输出
            return criterion(outputs, targets)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测（用于推理）
        
        Args:
            x: 输入张量
            
        Returns:
            预测结果（只返回主输出）
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            if isinstance(outputs, tuple):
                return outputs[0]  # 只返回主输出
            else:
                return outputs
    
    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数数量
        
        Returns:
            包含总参数和可训练参数数量的字典
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 统计不同部分的参数
        backbone_params = 0
        aux_params = 0
        classifier_params = 0
        
        for name, module in self.model.named_modules():
            if 'inception' in name or 'conv' in name or 'maxpool' in name:
                backbone_params += sum(p.numel() for p in module.parameters())
            elif 'aux' in name:
                aux_params += sum(p.numel() for p in module.parameters())
            elif 'fc' in name:
                classifier_params += sum(p.numel() for p in module.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'backbone': backbone_params,
            'auxiliary_classifiers': aux_params,
            'classifier': classifier_params
        }
    
    @classmethod
    def from_pretrained(cls, num_classes: int = 1000, weights: str = 'DEFAULT', **kwargs):
        """创建预训练的 GoogleNet 模型
        
        Args:
            num_classes: 分类数量
            weights: 预训练权重类型，可选 'DEFAULT', 'IMAGENET1K_V1'
            **kwargs: 其他配置参数
            
        Returns:
            预训练的 GoogleNet 模型
        """
        config = {
            'num_classes': num_classes,
            'pretrained': True,
            'weights': weights,
            **kwargs
        }
        return cls(config)
    
    def freeze_backbone(self):
        """冻结主干网络参数"""
        for name, param in self.model.named_parameters():
            if not name.startswith('fc') and not name.startswith('aux'):
                param.requires_grad = False
        self.logger.info("主干网络已冻结")
    
    def unfreeze_backbone(self):
        """解冻主干网络参数"""
        for param in self.model.parameters():
            param.requires_grad = True
        self.logger.info("主干网络已解冻")
    
    def get_feature_extractor(self):
        """获取特征提取器（不包含分类层）"""
        # 创建一个新的模型，移除分类层
        feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        return feature_extractor 