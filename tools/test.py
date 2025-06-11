#!/usr/bin/env python3
"""
测试脚本 - 用于模型推理和结果提交
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Any
import re

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl_framework.models.registry import ModelRegistry
from dl_framework.datasets.registry import DatasetRegistry
from dl_framework.utils.config import load_config
from dl_framework.utils.logger import get_logger, configure_logging
from dl_framework.utils.checkpoint import load_checkpoint


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PlateNet 测试脚本')
    parser.add_argument('--config', type=str, required=True,
                        help='数据集配置文件路径')
    parser.add_argument('--model-config', type=str, required=True,
                        help='模型配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备类型 (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='结果输出目录')
    
    return parser.parse_args()


def extract_image_id(image_path: str) -> str:
    """从图像路径中提取ID
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        图像ID（去除路径和扩展名）
    """
    # 获取文件名（不含路径）
    filename = os.path.basename(image_path)
    
    # 去除扩展名
    name_without_ext = os.path.splitext(filename)[0]
    
    # 尝试提取数字ID（去除前导零）
    # 支持格式如: 0028.jpg -> 28, test_0028.jpg -> 28, image28.jpg -> 28
    match = re.search(r'(\d+)', name_without_ext)
    if match:
        return str(int(match.group(1)))  # 转换为int再转回str以去除前导零
    else:
        # 如果没有找到数字，返回原始文件名（不含扩展名）
        return name_without_ext


def build_model(model_config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """构建模型
    
    Args:
        model_config: 模型配置
        device: 设备
        
    Returns:
        模型实例
    """
    model_type = model_config.get('type')
    if not model_type:
        raise ValueError("模型类型未指定")
        
    model_class = ModelRegistry.get(model_type)
    model = model_class(model_config)
    model = model.to(device)
    
    return model


def build_test_dataset(dataset_config: Dict[str, Any]):
    """构建测试数据集
    
    Args:
        dataset_config: 数据集配置
        
    Returns:
        测试数据集
    """
    dataset_type = dataset_config.get('type')
    if not dataset_type:
        raise ValueError("数据集类型未指定")
        
    dataset_class = DatasetRegistry.get(dataset_type)
    test_dataset = dataset_class(dataset_config, is_training=False)
    
    return test_dataset


def run_inference(model: torch.nn.Module, 
                  test_loader: DataLoader, 
                  device: torch.device,
                  logger) -> List[Dict[str, Any]]:
    """运行推理
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        logger: 日志器
        
    Returns:
        推理结果列表，每个元素包含 {'image_path': str, 'prediction': int, 'confidence': float}
    """
    model.eval()
    results = []
    
    logger.info(f"开始推理，共 {len(test_loader)} 个批次")
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            images = images.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 获取预测概率
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            
            # 获取当前批次的图像路径
            start_idx = batch_idx * test_loader.batch_size
            end_idx = min(start_idx + test_loader.batch_size, len(test_loader.dataset))
            
            for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                idx = start_idx + i
                if idx < len(test_loader.dataset):
                    image_path = test_loader.dataset.image_paths[idx]
                    results.append({
                        'image_path': image_path,
                        'prediction': pred.item(),
                        'confidence': conf.item()
                    })
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"已处理 {batch_idx + 1}/{len(test_loader)} 个批次")
    
    logger.info("推理完成")
    return results


def filter_clean_dishes(results: List[Dict[str, Any]]) -> List[str]:
    """筛选出被预测为干净盘子的图像ID
    
    Args:
        results: 推理结果列表
        
    Returns:
        干净盘子的ID列表
    """
    clean_ids = []
    
    for result in results:
        if result['prediction'] == 0:  # 0表示干净盘子
            image_id = extract_image_id(result['image_path'])
            clean_ids.append(image_id)
    
    return clean_ids


def generate_request_commands(clean_ids: List[str]) -> Dict[str, str]:
    """生成不同操作系统的请求命令
    
    Args:
        clean_ids: 干净盘子ID列表
        
    Returns:
        包含不同操作系统命令的字典
    """
    # 构造参数值
    ids_param = ','.join(clean_ids)
    base_url = "http://202.207.12.156:20000/calculate_accuracy"
    full_url = f"{base_url}?cleaned_ids={ids_param}"
    
    commands = {
        'windows': f'curl "{full_url}"',
        'linux': f"curl '{full_url}'",
        'mac': f"curl '{full_url}'"
    }
    
    return commands


def save_results(results: List[Dict[str, Any]], 
                 clean_ids: List[str], 
                 commands: Dict[str, str],
                 output_dir: str,
                 logger):
    """保存结果到文件
    
    Args:
        results: 完整推理结果
        clean_ids: 干净盘子ID列表
        commands: 请求命令
        output_dir: 输出目录
        logger: 日志器
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整结果
    results_file = os.path.join(output_dir, 'inference_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("图像路径\t预测类别\t置信度\t图像ID\n")
        for result in results:
            image_id = extract_image_id(result['image_path'])
            class_name = "干净盘子" if result['prediction'] == 0 else "脏盘子"
            f.write(f"{result['image_path']}\t{class_name}\t{result['confidence']:.4f}\t{image_id}\n")
    
    # 保存干净盘子ID列表
    clean_ids_file = os.path.join(output_dir, 'clean_dish_ids.txt')
    with open(clean_ids_file, 'w', encoding='utf-8') as f:
        f.write(','.join(clean_ids))
    
    # 保存请求命令
    commands_file = os.path.join(output_dir, 'request_commands.txt')
    with open(commands_file, 'w', encoding='utf-8') as f:
        f.write("=== 大作业结果测评系统 - 请求命令 ===\n\n")
        f.write(f"干净盘子ID列表: {','.join(clean_ids)}\n")
        f.write(f"总共识别出 {len(clean_ids)} 个干净盘子\n\n")
        
        f.write("Windows 系统请求命令:\n")
        f.write(f"{commands['windows']}\n\n")
        
        f.write("Linux 系统请求命令:\n")
        f.write(f"{commands['linux']}\n\n")
        
        f.write("Mac 系统请求命令:\n")
        f.write(f"{commands['mac']}\n\n")
    
    logger.info(f"结果已保存到 {output_dir}")
    logger.info(f"完整结果: {results_file}")
    logger.info(f"干净盘子ID: {clean_ids_file}")
    logger.info(f"请求命令: {commands_file}")


def main():
    """主函数"""
    args = parse_args()
    
    # 配置日志
    configure_logging(args.output_dir, "test")
    logger = get_logger("test")
    
    logger.info("开始测试脚本")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"模型配置: {args.model_config}")
    logger.info(f"权重文件: {args.checkpoint}")
    logger.info(f"设备: {args.device}")
    
    # 检查权重文件是否存在
    if not os.path.exists(args.checkpoint):
        logger.error(f"权重文件不存在: {args.checkpoint}")
        return
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"使用设备: {device}")
    
    try:
        # 加载配置
        dataset_config = load_config(args.config)['dataset']
        model_config = load_config(args.model_config)['model']
        
        # 构建模型
        logger.info("构建模型...")
        model = build_model(model_config, device)
        
        # 加载权重
        logger.info("加载模型权重...")
        load_checkpoint(model, args.checkpoint, device)
        logger.info("模型权重加载完成")
        
        # 构建测试数据集
        logger.info("构建测试数据集...")
        test_dataset = build_test_dataset(dataset_config)
        logger.info(f"测试集大小: {len(test_dataset)}")
        
        # 创建数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # 运行推理
        results = run_inference(model, test_loader, device, logger)
        
        # 筛选干净盘子
        clean_ids = filter_clean_dishes(results)
        logger.info(f"识别出 {len(clean_ids)} 个干净盘子")
        logger.info(f"干净盘子ID: {clean_ids}")
        
        # 生成请求命令
        commands = generate_request_commands(clean_ids)
        
        # 打印结果
        print("\n" + "="*60)
        print("测试结果")
        print("="*60)
        print(f"总测试图像数: {len(results)}")
        print(f"识别为干净盘子的数量: {len(clean_ids)}")
        print(f"识别为脏盘子的数量: {len(results) - len(clean_ids)}")
        print(f"\n干净盘子ID列表: {','.join(clean_ids)}")
        
        print("\n" + "="*60)
        print("请求命令")
        print("="*60)
        print("Windows 系统:")
        print(commands['windows'])
        print("\nLinux 系统:")
        print(commands['linux'])
        print("\nMac 系统:")
        print(commands['mac'])
        print("="*60)
        
        # 保存结果
        save_results(results, clean_ids, commands, args.output_dir, logger)
        
        logger.info("测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        raise


if __name__ == '__main__':
    main() 