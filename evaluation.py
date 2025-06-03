import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

def evaluate_model(model, test_loader, device, idx_to_class):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        idx_to_class: 索引到类别的映射
    
    Returns:
        tuple: (测试准确率, 预测结果, 真实标签)
    """
    model.eval()
    all_preds, all_labels = [], []
    correct_test, total_test = 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if labels.min() < 0:
                continue
                
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct_test / total_test if total_test > 0 else 0
    print(f'测试集准确率: {test_accuracy:.2f}%')
    
    # 生成分类报告
    if total_test > 0 and len(idx_to_class) > 0:
        class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
        valid_labels = [l for l in np.unique(all_labels) if l < len(class_names)]
        valid_preds = [p for p in np.unique(all_preds) if p < len(class_names)]
        target_labels = sorted(list(set(valid_labels) | set(valid_preds)))
        
        if target_labels:
            target_names_for_report = [class_names[i] for i in target_labels]
            print("\n分类报告:")
            print(classification_report(all_labels, all_preds, labels=target_labels, 
                                      target_names=target_names_for_report, zero_division=0))
    
    return test_accuracy, all_preds, all_labels

def plot_confusion_matrix(all_labels, all_preds, idx_to_class, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    if len(idx_to_class) == 0:
        print("无法生成混淆矩阵，因为没有类别信息。")
        return
    
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    valid_labels = [l for l in np.unique(all_labels) if l < len(class_names)]
    valid_preds = [p for p in np.unique(all_preds) if p < len(class_names)]
    target_labels = sorted(list(set(valid_labels) | set(valid_preds)))
    
    if not target_labels:
        print("无法生成混淆矩阵，因为没有有效的标签。")
        return
    
    target_names = [class_names[i] for i in target_labels]
    cm = confusion_matrix(all_labels, all_preds, labels=target_labels)
    
    plt.figure(figsize=(max(10, len(target_names)//2), max(8, len(target_names)//2.5)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    
    print(f"混淆矩阵已保存到: {save_path}")

def plot_training_history(history, save_path='training_curves.png'):
    """绘制训练历史曲线"""
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='训练损失', alpha=0.8)
    plt.plot(epochs_range, history['val_loss'], label='验证损失', alpha=0.8)
    plt.legend(loc='upper right')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.grid(True, alpha=0.3)

    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='训练准确率', alpha=0.8)
    plt.plot(epochs_range, history['val_acc'], label='验证准确率', alpha=0.8)
    plt.legend(loc='lower right')
    plt.title('训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.grid(True, alpha=0.3)

    # 学习率曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, history['lr'], label='学习率', color='green', alpha=0.8)
    plt.legend(loc='upper right')
    plt.title('学习率变化')
    plt.xlabel('Epoch')
    plt.ylabel('学习率')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 过拟合指标
    plt.subplot(2, 2, 4)
    loss_diff = np.array(history['train_loss']) - np.array(history['val_loss'])
    plt.plot(epochs_range, loss_diff, label='训练-验证损失差异', color='purple', alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title('过拟合指标 (训练损失 - 验证损失)')
    plt.xlabel('Epoch')
    plt.ylabel('损失差异')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    
    print(f"训练曲线已保存到: {save_path}")
