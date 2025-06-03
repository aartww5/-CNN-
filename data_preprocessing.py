import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import shutil
import random
import numpy as np



class HandSignDataset(Dataset):
    """手语数据集类"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 获取类别信息
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if not self.classes:
            raise ValueError(f"在目录 {data_dir} 中没有找到类别文件夹。")
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        # 加载所有图片路径和标签
        for class_name in self.classes:
            class_path = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误: 无法加载图片 {img_path}. {e}")
            # 返回占位符图像
            placeholder_image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                placeholder_image = self.transform(placeholder_image)
            return placeholder_image, -1

        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class HandSignSpecificTransforms:
    """手语特定的数据增强类"""
    
    def __init__(self, tremor_prob=0.3, lighting_prob=0.4, blur_prob=0.0):
        self.tremor_prob = tremor_prob
        self.lighting_prob = lighting_prob
        self.blur_prob = blur_prob

    def simulate_hand_tremor(self, image, max_offset=2):
        """模拟手部轻微颤抖"""
        if random.random() < self.tremor_prob:
            offset_x = random.randint(-max_offset, max_offset)
            offset_y = random.randint(-max_offset, max_offset)
            return transforms.functional.affine(
                image, angle=0, translate=[offset_x, offset_y], 
                scale=1.0, shear=[0, 0], fill=0
            )
        return image
    
    def simulate_lighting_change(self, image):
        """模拟光照变化"""
        if random.random() < self.lighting_prob:
            gamma = random.uniform(0.7, 1.3)
            return transforms.functional.adjust_gamma(image, gamma)
        return image

    def add_realistic_background_blur(self, image):
        """添加背景模糊"""
        if random.random() < self.blur_prob:
            return transforms.functional.gaussian_blur(image, kernel_size=3, sigma=(0.1, 0.5))
        return image

    def __call__(self, img):
        img = self.simulate_hand_tremor(img)
        img = self.simulate_lighting_change(img)
        img = self.add_realistic_background_blur(img)
        return img

def get_sign_language_transforms(input_size=224, aggressive=False):
    """获取针对手语优化的数据增强变换"""
    
    custom_pil_transforms = HandSignSpecificTransforms(
        tremor_prob=0.3 if not aggressive else 0.5, 
        lighting_prob=0.4 if not aggressive else 0.6
    )

    if aggressive:
        print("使用激进的数据增强策略。")
        train_transform_list = [
            transforms.Resize((int(input_size * 1.2), int(input_size * 1.2))),
            custom_pil_transforms,
            transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(degrees=10, fill=0),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4, fill=0),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        print("使用保守的数据增强策略。")
        train_transform_list = [
            transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),
            custom_pil_transforms,
            transforms.CenterCrop(input_size),
            transforms.RandomRotation(degrees=5, fill=0),
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=None, scale=(0.95, 1.05), shear=None),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    train_transform = transforms.Compose(train_transform_list)
    
    val_test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(train_dir, val_dir, test_dir, batch_size=32, input_size=224, 
                      aggressive_aug=False, num_workers=2):
    """创建数据加载器"""
    
    train_transform, val_test_transform = get_sign_language_transforms(input_size, aggressive=aggressive_aug)
    
    train_dataset = HandSignDataset(train_dir, transform=train_transform)
    val_dataset = HandSignDataset(val_dir, transform=val_test_transform)
    test_dataset = HandSignDataset(test_dir, transform=val_test_transform)

    if len(train_dataset) == 0:
        raise ValueError(f"训练数据集为空，请检查路径: {train_dir}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx, train_dataset.idx_to_class

def analyze_dataset_statistics(train_dir, aggressive_threshold=150):
    """分析数据集统计信息，决定是否使用激进增强"""
    
    num_train_images = 0
    num_classes_in_train = 0
    
    if os.path.exists(train_dir):
        class_folders = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        num_classes_in_train = len(class_folders)
        for class_folder in class_folders:
            class_path = os.path.join(train_dir, class_folder)
            num_train_images += len([name for name in os.listdir(class_path) 
                                   if name.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    avg_images_per_class = num_train_images / num_classes_in_train if num_classes_in_train > 0 else 0
    
    print(f"训练集统计:")
    print(f"  总图片数: {num_train_images}")
    print(f"  类别数: {num_classes_in_train}")
    print(f"  平均每个类别图片数: {avg_images_per_class:.2f}")
    
    use_aggressive_augmentation = False
    if avg_images_per_class < aggressive_threshold and avg_images_per_class > 0:
        print(f"  -> 样本数较少，启用激进数据增强")
        use_aggressive_augmentation = True
    elif avg_images_per_class == 0 and num_classes_in_train > 0:
        print(f"  -> 警告: 有类别文件夹但没有图片")
        return None, num_train_images, num_classes_in_train, avg_images_per_class
    elif num_classes_in_train == 0:
        print(f"  -> 错误: 没有找到类别文件夹")
        return None, num_train_images, num_classes_in_train, avg_images_per_class
    else:
        print(f"  -> 样本数充足，使用保守数据增强")
    
    return use_aggressive_augmentation, num_train_images, num_classes_in_train, avg_images_per_class