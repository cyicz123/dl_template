import os
import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Dict, Any, Tuple, Optional, List

from .base_dataset import BaseDataset
from .registry import DatasetRegistry

@DatasetRegistry.register('cat_dog')
class CatDogDataset(BaseDataset):
    """猫狗分类数据集"""
    
    def __init__(self, config: Dict[str, Any], is_training: bool = True):
        """初始化
        
        Args:
            config: 数据集配置
            is_training: 是否为训练集
        """
        super().__init__(config, is_training)
        self.transform = self._get_transforms()
        self._prepare_dataset()
        self._load_data()
        
    def _get_transforms(self) -> transforms.Compose:
        """获取数据变换
        
        Returns:
            数据变换组合
        """
        transform_config = self.config.get('transforms', {})
        
        transform_list = []
        
        # 调整大小
        if 'resize' in transform_config:
            size = transform_config['resize']
            transform_list.append(transforms.Resize(size))
        
        # 如果是训练集，添加数据增强
        if self.is_training:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        
        # 转换为张量
        transform_list.append(transforms.ToTensor())
        
        # 标准化
        if 'normalize' in transform_config:
            mean = transform_config['normalize'].get('mean', [0.485, 0.456, 0.406])
            std = transform_config['normalize'].get('std', [0.229, 0.224, 0.225])
            transform_list.append(transforms.Normalize(mean, std))
        
        return transforms.Compose(transform_list)
        
    def _prepare_dataset(self) -> None:
        """准备数据集，执行数据集拆分和组织
        
        如果原始数据集需要重新组织的情况下使用此方法
        """
        # 获取配置
        original_dataset_dir = self.config.get('original_dataset_dir', '')
        if not original_dataset_dir or not os.path.exists(original_dataset_dir):
            # 如果已经准备好了数据或者没有提供原始数据目录，则跳过
            return
            
        original_datatrain_dir = self.config.get('original_datatrain_dir', '') 
        original_datatest_dir = self.config.get('original_datatest_dir', '')
        if not os.path.exists(original_datatrain_dir) or not os.path.exists(original_datatest_dir):
            # 如果原始数据集目录不存在，则跳过
            return
        

         # 是否需要重新准备数据集
        force_prepare = self.config.get('force_prepare', False)
        base_dir = os.path.join(self.data_dir, 'cat_dog')
        
        # 判断是否已经准备好数据
        train_dir = os.path.join(base_dir, 'train')
        test_dir = os.path.join(base_dir, 'test')
        
        if not force_prepare and os.path.exists(train_dir) and os.path.exists(test_dir):
            # 数据已经准备好，不需要重新处理
            return
            
        # 创建目录结构
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        train_cats_dir = os.path.join(train_dir, 'cats')
        os.makedirs(train_cats_dir, exist_ok=True)
        
        train_dogs_dir = os.path.join(train_dir, 'dogs')
        os.makedirs(train_dogs_dir, exist_ok=True)
        
        test_cats_dir = os.path.join(test_dir, 'cats')
        os.makedirs(test_cats_dir, exist_ok=True)
        
        test_dogs_dir = os.path.join(test_dir, 'dogs')
        os.makedirs(test_dogs_dir, exist_ok=True)
        
        # 复制猫的图片
        train_cat_count = self.config.get('train_cat_count', 200)
        test_cat_count = self.config.get('test_cat_count', 100)
        test_cat_start = self.config.get('test_cat_start', 300)
        
        # 复制训练集猫的图片
        fnames = ['cat.{}.jpg'.format(i) for i in range(train_cat_count)]
        for fname in fnames:
            src = os.path.join(original_datatrain_dir, fname)
            dst = os.path.join(train_cats_dir, fname)
            if os.path.exists(src):
                shutil.copyfile(src, dst)
        
        # 复制测试集猫的图片
        fnames = ['cat.{}.jpg'.format(i) for i in range(test_cat_start, test_cat_start + test_cat_count)]
        for fname in fnames:
            src = os.path.join(original_datatrain_dir, fname)
            dst = os.path.join(test_cats_dir, fname)
            if os.path.exists(src):
                shutil.copyfile(src, dst)
        
        # 复制狗的图片
        train_dog_count = self.config.get('train_dog_count', 200)
        test_dog_count = self.config.get('test_dog_count', 100)
        test_dog_start = self.config.get('test_dog_start', 300)
        
        # 复制训练集狗的图片
        fnames = ['dog.{}.jpg'.format(i) for i in range(train_dog_count)]
        for fname in fnames:
            src = os.path.join(original_datatrain_dir, fname)
            dst = os.path.join(train_dogs_dir, fname)
            if os.path.exists(src):
                shutil.copyfile(src, dst)
        
        # 复制测试集狗的图片
        fnames = ['dog.{}.jpg'.format(i) for i in range(test_dog_start, test_dog_start + test_dog_count)]
        for fname in fnames:
            src = os.path.join(original_datatrain_dir, fname)
            dst = os.path.join(test_dogs_dir, fname)
            if os.path.exists(src):
                shutil.copyfile(src, dst)
    
    def _load_data(self) -> None:
        """加载数据"""
        base_dir = os.path.join(self.data_dir, 'cat_dog')
        
        # 根据是否为训练集选择相应的目录
        data_dir = os.path.join(base_dir, 'train' if self.is_training else 'test')
        
        # 创建图像和标签列表
        self.image_paths = []
        self.labels = []
        
        # 加载猫的图像
        cats_dir = os.path.join(data_dir, 'cats')
        if os.path.exists(cats_dir):
            cat_images = [os.path.join(cats_dir, fname) for fname in os.listdir(cats_dir)]
            self.image_paths.extend(cat_images)
            self.labels.extend([0] * len(cat_images))  # 猫的标签为0
        
        # 加载狗的图像
        dogs_dir = os.path.join(data_dir, 'dogs')
        if os.path.exists(dogs_dir):
            dog_images = [os.path.join(dogs_dir, fname) for fname in os.listdir(dogs_dir)]
            self.image_paths.extend(dog_images)
            self.labels.extend([1] * len(dog_images))  # 狗的标签为1
    
    def __len__(self) -> int:
        """获取数据集长度
        
        Returns:
            数据集长度
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            (图像, 标签) 的元组
        """
        # 读取图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label