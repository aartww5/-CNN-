# split_data_standalone.py
import os
import shutil
import random
from collections import Counter
import torch
import numpy as np

def seed_everything(seed=42):
    """设置随机种子以保证结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, verbose=True):


    seed_everything() # 确保划分过程可复现

    if verbose:
        print("=" * 50)
        print("开始数据集划分...")
        print(f"原始数据目录: {data_dir}")
        print(f"输出目录: {output_dir}")
        print(f"划分比例: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
        print("=" * 50)

    if not os.path.exists(data_dir):
        print(f"错误: 原始数据集目录 '{data_dir}' 不存在!")
        return False

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        print("错误: train, val, test 比例之和必须为 1.0")
        return False

    # 如果输出目录已存在且包含数据，可以选择清空或提示
    if os.path.exists(output_dir):
        if any(os.scandir(output_dir)): # 检查目录是否为空
            print("splited_dataset已存在，跳过划分")
            pass 
    
    # 创建输出目录结构 (train, val, test)
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_dir, split)
        os.makedirs(split_path, exist_ok=True)

    # 获取所有类别 (子文件夹名称)
    all_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not all_classes:
        print(f"错误: 在 '{data_dir}' 中没有找到子目录作为类别。")
        return False
    
    if verbose:
        print(f"发现 {len(all_classes)} 个类别: {sorted(all_classes)}")

    # 统计每个集合的数据量
    split_counts = {'train': Counter(), 'val': Counter(), 'test': Counter()}
    total_images_processed = 0

    for class_name in all_classes:
        class_dir = os.path.join(data_dir, class_name)
        
        # 确保只处理图片文件
        images = [
            f for f in os.listdir(class_dir) 
            if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        
        if not images:
            if verbose:
                print(f"警告: 类别 '{class_name}' 在 '{class_dir}' 中没有图片文件。")
            continue

        random.shuffle(images) # 打乱每个类别下的图片

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # 确保测试集至少分配剩余的图片，即使总数很少
        n_test = n_total - n_train - n_val 
        if n_train + n_val + n_test < n_total : # 处理因取整导致的总数不足
            n_test += (n_total - (n_train + n_val + n_test))


        # 分配图片到各个集合
        train_imgs = images[:n_train]
        val_imgs = images[n_train : n_train + n_val]
        test_imgs = images[n_train + n_val :] # 确保所有剩余图片都进入测试集

        datasets_map = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }

        for split_name, split_images in datasets_map.items():
            split_class_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            for img_name in split_images:
                src_path = os.path.join(class_dir, img_name)
                dst_path = os.path.join(split_class_dir, img_name)
                
                # 避免重复复制（如果输出目录已存在部分数据）
                if not os.path.exists(dst_path):
                    shutil.copyfile(src_path, dst_path)
                
                split_counts[split_name][class_name] += 1
                total_images_processed += 1
    
    if verbose:
        print("-" * 50)
        print("数据集划分完成。")
        print(f"总共处理了 {total_images_processed} 张图片。")
        print("\n各集合数据量统计:")
        for split_name, counts_per_class in split_counts.items():
            total_in_split = sum(counts_per_class.values())
            print(f"  {split_name.capitalize()}: {total_in_split}")
        print("=" * 50)
        
    return True
