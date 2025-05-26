from .base_dataset import BaseDataset
from .registry import DatasetRegistry
from .cifar10 import CIFAR10Dataset
from .cat_dog import CatDogDataset
from .plate import PlateDataset

__all__ = [
    'BaseDataset',
    'DatasetRegistry',
    'CIFAR10Dataset',
    'CatDogDataset',
    'PlateDataset',
]
