import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0.001, verbose=False, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_weights = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'验证损失下降 ({self.val_loss_min:.6f} --> {val_loss:.6f})。保存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.best_weights = model.state_dict()

    def load_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print("已加载早停保存的最佳模型权重。")
        elif os.path.exists(self.path):
            model.load_state_dict(torch.load(self.path))
            if self.verbose:
                print(f"已从文件加载最佳模型权重: {self.path}")

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, 
                weight_decay=1e-4, patience=10, model_save_path='best_hand_sign_model.pth'):
    """
    训练模型
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 最大训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        patience: 早停耐心值
        model_save_path: 模型保存路径
    
    Returns:
        tuple: (训练好的模型, 训练历史)
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_save_path)
    
    # 训练历史记录
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [], 
        'lr': []
    }
    
    print("开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # 跳过有问题的样本
            if labels.min() < 0:
                print(f"跳过有问题的batch (label < 0)")
                continue

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # 计算训练指标
        epoch_train_loss = running_loss / total_train if total_train > 0 else 0
        epoch_train_acc = 100 * correct_train / total_train if total_train > 0 else 0
        
        # 验证阶段
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if labels.min() < 0:
                    continue
                    
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # 计算验证指标
        epoch_val_loss = val_loss / total_val if total_val > 0 else 0
        epoch_val_acc = 100 * correct_val / total_val if total_val > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)
        
        # 打印进度
        print(f'Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.2e} | '
              f'Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | '
              f'Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%')
        
        # 更新学习率
        scheduler.step()
        
        # 早停检查
        if early_stopping(epoch_val_loss, model):
            print("早停触发!")
            break
            
    print("训练完成。")
    early_stopping.load_best_weights(model)  # 加载最佳权重
    return model, history