import os
import matplotlib.pyplot as plt
from split import(
    seed_everything, split_dataset
)
from data_preprocessing import (
    create_dataloaders, analyze_dataset_statistics
)
from model import create_model
from training import train_model
import torch
from evaluation import (
    evaluate_model, plot_confusion_matrix, plot_training_history, 

)

def main():
    """主程序"""
    
    # ==================== 配置参数 ====================
    ORIGINAL_DATA_DIR = 'asl_dataset'  # 原始数据集路径
    SPLIT_DATA_DIR = 'splited_dataset'  # 划分后数据路径
    MODEL_SAVE_PATH = 'best_optimized_hand_sign_model.pth'
    
    # 数据划分参数
    DO_DATA_SPLIT = True  # 是否进行数据划分
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 模型训练参数
    INPUT_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4 # L2正则化强度
    DROPOUT_RATE = 0.4 # 模型中的dropout率
    PATIENCE_EARLY_STOPPING = 15 # 早停的耐心轮数
    NUM_WORKERS = 4 # DataLoader的工作进程数
    AGGRESSIVE_AUGMENTATION_THRESHOLD = 150 # 平均每类样本数低于此值时启用激进增强
    
    # ==================== 初始化 ====================
    # 设置随机种子
    seed_everything(42)
    
    # 设置matplotlib中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("="*60)
    print("手语识别模型训练程序")
    print("="*60)
    
    # ==================== 数据划分 ====================
    if DO_DATA_SPLIT or not os.path.exists(os.path.join(SPLIT_DATA_DIR, 'train')):
        print("\n[1/5] 数据划分阶段")
        print("-" * 30)
        if not split_dataset(ORIGINAL_DATA_DIR, SPLIT_DATA_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
            print("数据划分失败，程序退出。")
            return
    else:
        print("\n[1/5] 使用已存在的数据划分")
        print("-" * 30)
    
    # 设置数据路径
    train_dir = os.path.join(SPLIT_DATA_DIR, 'train')
    val_dir = os.path.join(SPLIT_DATA_DIR, 'val')
    test_dir = os.path.join(SPLIT_DATA_DIR, 'test')
    
    # ==================== 数据分析 ====================
    print("\n[2/5] 数据分析和预处理")
    print("-" * 30)
    
    # 分析数据集统计信息
    use_aggressive_augmentation, num_train_images, num_classes_in_train, avg_images_per_class = \
        analyze_dataset_statistics(train_dir, AGGRESSIVE_AUGMENTATION_THRESHOLD)
    
    if use_aggressive_augmentation is None:
        print("数据分析失败，程序退出。")
        return
    
    # 创建数据加载器
    try:
        train_loader, val_loader, test_loader, class_to_idx, idx_to_class = create_dataloaders(
            train_dir, val_dir, test_dir, BATCH_SIZE, INPUT_SIZE, 
            use_aggressive_augmentation, NUM_WORKERS
        )
    except ValueError as e:
        print(f"创建DataLoaders失败: {e}")
        return
    
    num_classes = len(class_to_idx)
    if num_classes == 0:
        print("错误: 未能从数据集中加载任何类别。")
        return
    
    print(f"成功加载 {num_classes} 个类别: {sorted(class_to_idx.keys())}")
    
    # ==================== 模型创建 ====================
    print("\n[3/5] 模型创建")
    print("-" * 30)
    
    model = create_model(num_classes=num_classes, input_size=INPUT_SIZE, dropout_rate=DROPOUT_RATE)
    
    # ==================== 模型训练 ====================
    print("\n[4/5] 模型训练")
    print("-" * 30)
    print(f"训练配置:")
    print(f"  最大轮数: {NUM_EPOCHS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  批大小: {BATCH_SIZE}")
    print(f"  早停耐心: {PATIENCE_EARLY_STOPPING}")
    print(f"  数据增强: {'激进' if use_aggressive_augmentation else '保守'}")
    
    trained_model, history = train_model(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE_EARLY_STOPPING, 
        model_save_path=MODEL_SAVE_PATH
    )
    
    # ==================== 模型评估 ====================
    print("\n[5/5] 模型评估和可视化")
    print("-" * 30)
    
    # 绘制训练曲线
    print("绘制训练曲线...")
    plot_training_history(history)
    
    # 在测试集上评估
    print("在测试集上评估模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确保加载最佳模型权重
    if os.path.exists(MODEL_SAVE_PATH):
        trained_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"已加载最佳模型权重: {MODEL_SAVE_PATH}")
    
    test_accuracy, all_preds, all_labels = evaluate_model(trained_model, test_loader, device, idx_to_class)
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    plot_confusion_matrix(all_labels, all_preds, idx_to_class)
    
    
    # ==================== 结果总结 ====================
    print("\n" + "="*60)
    print("训练完成！结果总结:")
    print("="*60)
    print(f"数据集信息:")
    print(f"  训练样本数: {num_train_images}")
    print(f"  类别数: {num_classes}")
    print(f"  平均每类样本数: {avg_images_per_class:.1f}")
    print(f"")
    print(f"训练结果:")
    if history['val_acc']:
        print(f"  最佳验证准确率: {max(history['val_acc']):.2f}%")
    print(f"  最终测试准确率: {test_accuracy:.2f}%")
    print(f"  训练轮数: {len(history['train_loss'])}")
    print(f"")
    print(f"模型文件: {MODEL_SAVE_PATH}")
    print("="*60)

if __name__ == '__main__':
    main()