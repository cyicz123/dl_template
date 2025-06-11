import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Any, Tuple, Optional, List

from .base_dataset import BaseDataset
from .registry import DatasetRegistry
from ..utils.logger import get_logger

@DatasetRegistry.register('plate')
class PlateDataset(BaseDataset):
    """盘子分类数据集（干净/脏盘子）"""
    
    def __init__(self, config: Dict[str, Any], is_training: bool = True):
        """初始化
        
        Args:
            config: 数据集配置
            is_training: 是否为训练集
        """
        super().__init__(config, is_training)
        self.logger = get_logger(__name__)
        self.transform = self._get_transforms()
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
        
        # 中心裁剪
        if 'center_crop' in transform_config:
            size = transform_config['center_crop']
            transform_list.append(transforms.CenterCrop(size))
        
        # 如果是训练集，添加数据增强
        if self.is_training:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
            ])
        
        # 转换为张量
        transform_list.append(transforms.ToTensor())
        
        # 标准化
        if 'normalize' in transform_config:
            mean = transform_config['normalize'].get('mean', [0.485, 0.456, 0.406])
            std = transform_config['normalize'].get('std', [0.229, 0.224, 0.225])
            transform_list.append(transforms.Normalize(mean, std))
        
        return transforms.Compose(transform_list)
    
    def _load_data(self) -> None:
        """加载数据"""
        # 获取盘子数据的根目录
        plate_dir = os.path.join(self.data_dir, self.config.get('data_name', 'plate'))
        
        if not os.path.exists(plate_dir):
            raise FileNotFoundError(f"盘子数据集目录不存在: {plate_dir}")
        
        # 创建图像路径和标签列表
        self.image_paths = []
        self.labels = []
        
        # 检查是否使用分割文件夹结构
        use_split_folders = self.config.get('use_split_folders', False)
        
        if use_split_folders:
            self._load_data_from_split_folders(plate_dir)
        else:
            self._load_data_from_class_folders(plate_dir)
        
        # 检查是否有足够的数据
        if len(self.image_paths) == 0:
            mode = "训练" if self.is_training else "测试"
            raise ValueError(f"没有找到{mode}数据。请确保数据目录结构正确。")
        
        # 打印数据集信息
        self.logger.info(f"{'训练' if self.is_training else '测试'}集加载完成: "
              f"总共 {len(self.image_paths)} 张图片, "
              f"干净盘子: {self.labels.count(0)}, "
              f"脏盘子: {self.labels.count(1)}")
    
    def _load_data_from_split_folders(self, plate_dir: str) -> None:
        """从分割文件夹结构加载数据 (train/val/test)
        
        Args:
            plate_dir: 盘子数据集根目录
        """
        # 确定要加载的文件夹
        if self.is_training:
            # 训练时加载train文件夹
            split_folder = 'train'
        else:
            # 测试时优先加载test文件夹，如果不存在则加载val文件夹
            if os.path.exists(os.path.join(plate_dir, 'test')):
                split_folder = 'test'
            elif os.path.exists(os.path.join(plate_dir, 'val')):
                split_folder = 'val'
            else:
                raise FileNotFoundError(f"在 {plate_dir} 中找不到 'test' 或 'val' 文件夹")
        
        split_dir = os.path.join(plate_dir, split_folder)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"分割文件夹不存在: {split_dir}")
        
        # 加载干净盘子图像
        clean_dir = os.path.join(split_dir, 'clean_dish')
        if os.path.exists(clean_dir):
            clean_images = [os.path.join(clean_dir, fname) for fname in os.listdir(clean_dir) 
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
            clean_images.sort()
            self.image_paths.extend(clean_images)
            self.labels.extend([0] * len(clean_images))  # 干净盘子的标签为0
        
        # 加载脏盘子图像
        dirty_dir = os.path.join(split_dir, 'dirty_dish')
        if os.path.exists(dirty_dir):
            dirty_images = [os.path.join(dirty_dir, fname) for fname in os.listdir(dirty_dir) 
                           if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
            dirty_images.sort()
            self.image_paths.extend(dirty_images)
            self.labels.extend([1] * len(dirty_images))  # 脏盘子的标签为1
        
        self.logger.info(f"从 {split_folder} 文件夹加载数据")
    
    def _load_data_from_class_folders(self, plate_dir: str) -> None:
        """从类别文件夹结构加载数据 (原有的clean_dish/dirty_dish结构)
        
        Args:
            plate_dir: 盘子数据集根目录
        """
        # 设置训练/测试分割比例
        train_ratio = self.config.get('train_ratio', 0.8)
        
        # 加载干净盘子图像
        clean_dir = os.path.join(plate_dir, 'clean_dish')
        if os.path.exists(clean_dir):
            clean_images = [os.path.join(clean_dir, fname) for fname in os.listdir(clean_dir) 
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 对文件名排序以确保可重复性
            clean_images.sort()
            
            # 根据训练/测试划分选择相应的图像
            split_idx = int(len(clean_images) * train_ratio)
            if self.is_training:
                selected_clean = clean_images[:split_idx]
            else:
                selected_clean = clean_images[split_idx:]
            
            self.image_paths.extend(selected_clean)
            self.labels.extend([0] * len(selected_clean))  # 干净盘子的标签为0
        
        # 加载脏盘子图像
        dirty_dir = os.path.join(plate_dir, 'dirty_dish')
        if os.path.exists(dirty_dir):
            dirty_images = [os.path.join(dirty_dir, fname) for fname in os.listdir(dirty_dir) 
                           if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 对文件名排序以确保可重复性
            dirty_images.sort()
            
            # 根据训练/测试划分选择相应的图像
            split_idx = int(len(dirty_images) * train_ratio)
            if self.is_training:
                selected_dirty = dirty_images[:split_idx]
            else:
                selected_dirty = dirty_images[split_idx:]
            
            self.image_paths.extend(selected_dirty)
            self.labels.extend([1] * len(selected_dirty))  # 脏盘子的标签为1
    
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
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"加载图像 {img_path} 时出错: {e}")
            # 如果出错，返回数据集中的另一个图像
            return self.__getitem__((idx + 1) % len(self))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
