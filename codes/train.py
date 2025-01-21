import os
import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import f1_score

from load_feature import load_feature
from model0 import MultimodalClassifier0
from model1 import MultimodalClassifier1
from model2 import MultimodalClassifier2
from utils import save_checkpoint, save_loss_accuracy_data, plot_loss_accuracy, save_model

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train(batch_size=16, epochs=50, accumulation_steps=4, lr=1e-4, val_ratio=0.2, disable_save=False, model_type="model0", mode='multimodal'):
    """
    训练模型。
    :param batch_size: 批量大小
    :param epochs: 训练轮数
    :param accumulation_steps: 梯度累积步数
    :param lr: 学习率
    :param val_ratio: 验证集比例
    :param disable_save: 是否禁用保存
    :param model_type: 模型类型（model0, model1, model2）
    :param mode: 模式（multimodal, text_only, image_only）
    """
    # 初始化变量
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1s = []
    val_f1s = []
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience = 15
    patience_counter = 0

    # 加载数据
    print("Loading data...")
    current_dir = os.path.dirname(__file__)
    model_dir = os.path.join(current_dir, '..', 'models')

    # 创建带时间戳的检查点目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.normpath(os.path.join(current_dir, '..', 'checkpoints', f"checkpoints_{timestamp}"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    data_path = os.path.join(current_dir, '..', 'AAAdata', 'train_data.csv')
    # 接收 load_feature 返回的所有值
    train_text_features, train_image_features, train_labels, val_text_features, val_image_features, val_labels, class_weights = load_feature(data_path, val_ratio=val_ratio)
    print(f"Data loaded successfully. Train text features shape: {train_text_features.shape}, Train image features shape: {train_image_features.shape}, Val text features shape: {val_text_features.shape}, Val image features shape: {val_image_features.shape}")

    # 归一化类别权重
    class_weights = class_weights / class_weights.sum()
    print(f"Class weights (normalized): {class_weights}")

    # 创建训练集和验证集的数据集
    train_dataset = TensorDataset(train_text_features, train_image_features, train_labels)
    val_dataset = TensorDataset(val_text_features, val_image_features, val_labels)

    # 创建数据加载器
    print("\nCreating dataset and data loader...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Dataset created with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # 初始化模型
    print("\nInitializing model...")
    text_feature_dim = train_text_features.shape[1]
    image_feature_dim = train_image_features.shape[1]
    num_classes = 3

    # 根据 model_type 选择模型
    if model_type == "model0":
        model = MultimodalClassifier0(text_feature_dim, image_feature_dim, num_classes)
    elif model_type == "model1":
        model = MultimodalClassifier1(text_feature_dim, image_feature_dim, num_classes)
    elif model_type == "model2":
        model = MultimodalClassifier2(text_feature_dim, image_feature_dim, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 启用 GPU 加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model initialized. Text feature dimension: {text_feature_dim}, Image feature dimension: {image_feature_dim}, Number of classes: {num_classes}")
    print(f"Using device: {device}")

    # 定义损失函数和优化器
    print("\nDefining loss function and optimizer...")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler()

    # 定义学习率调度器（ReduceLROnPlateau）
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    print("Loss function (CrossEntropyLoss) and optimizer (Adam with L2 regularization) defined.")

    # 训练模型
    print("\nStarting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        all_preds_train = []
        all_labels_train = []
        optimizer.zero_grad()

        # 使用 tqdm 显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch_idx, (batch_text, batch_image, batch_labels) in enumerate(progress_bar):
            # 将数据移动到 GPU
            batch_text, batch_image, batch_labels = batch_text.to(device), batch_image.to(device), batch_labels.to(device)

            # 确保 batch_labels 是 LongTensor 类型
            batch_labels = batch_labels.long()

            # 使用混合精度训练
            with autocast():
                if mode == 'text_only':
                    outputs = model(batch_text, None, mode=mode)
                elif mode == 'image_only':
                    outputs = model(None, batch_image, mode=mode)
                elif mode == 'multimodal':
                    outputs = model(batch_text, batch_image, mode=mode)
                else:
                    raise ValueError("Invalid mode. Choose from 'text_only', 'image_only', or 'multimodal'.")

                loss = criterion(outputs, batch_labels)
                # 归一化损失
                loss = loss / accumulation_steps

            # 反向传播和优化
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 累积训练损失
            epoch_train_loss += loss.item() * accumulation_steps

            # 计算训练集准确率
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()

            # 存储训练集的预测结果和真实标签
            all_preds_train.extend(predicted.cpu().numpy())
            all_labels_train.extend(batch_labels.cpu().numpy())

            # 更新进度条描述
            loss_value = loss.item() * accumulation_steps
            progress_bar.set_postfix({"Train Loss": f"{loss_value:.4f}"})

        # 计算并保存当前 epoch 的训练集平均损失和准确率
        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_epoch_train_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # 计算训练集的 F1 分数
        train_f1 = f1_score(all_labels_train, all_preds_train, average='macro')
        train_f1s.append(train_f1)

        print(f"Epoch {epoch + 1}/{epochs} completed. Train Loss: {avg_epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Train F1: {train_f1:.4f}")

        # 验证模型
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds_val = []
        all_labels_val = []
        with torch.no_grad():
            for batch_text, batch_image, batch_labels in val_loader:
                # 将数据移动到 GPU
                batch_text, batch_image, batch_labels = batch_text.to(device), batch_image.to(device), batch_labels.to(device)

                # 确保 batch_labels 是 LongTensor 类型
                batch_labels = batch_labels.long()

                # 根据模式计算验证损失
                if mode == 'text_only':
                    outputs = model(batch_text, None, mode=mode)
                elif mode == 'image_only':
                    outputs = model(None, batch_image, mode=mode)
                elif mode == 'multimodal':
                    outputs = model(batch_text, batch_image, mode=mode)
                else:
                    raise ValueError("Invalid mode. Choose from 'text_only', 'image_only', or 'multimodal'.")

                loss = criterion(outputs, batch_labels)
                epoch_val_loss += loss.item()

                # 计算验证集准确率
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_labels.size(0)
                correct_val += (predicted == batch_labels).sum().item()

                # 存储验证集的预测结果和真实标签
                all_preds_val.extend(predicted.cpu().numpy())
                all_labels_val.extend(batch_labels.cpu().numpy())

        # 计算并保存当前 epoch 的验证集平均损失和准确率
        avg_epoch_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_epoch_val_loss)
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        # 计算验证集的 F1 分数
        val_f1 = f1_score(all_labels_val, all_preds_val, average='macro')
        val_f1s.append(val_f1)

        print(f"Epoch {epoch + 1}/{epochs} completed. Val Loss: {avg_epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Val F1: {val_f1:.4f}")

        # 更新学习率调度器（ReduceLROnPlateau）
        scheduler.step(avg_epoch_val_loss)

        # 早停机制
        if avg_epoch_val_loss < best_val_loss or val_accuracy > best_val_accuracy:
            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        # 如果验证损失连续 patience 个 epoch 没有下降，则停止训练
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}!")
            break

        # 每 30 个 epoch 保存一次
        if not disable_save and (epoch + 1) % 30 == 0:
            save_checkpoint(checkpoint_dir, epoch + 1, model, optimizer, avg_epoch_train_loss, avg_epoch_val_loss, train_accuracy, val_accuracy)

    # 保存最终模型
    if not disable_save:
        save_model(model_dir, model, model_type, timestamp)
        save_loss_accuracy_data(current_dir, train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s, model_type, timestamp)
        plot_loss_accuracy(current_dir, train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s, model_type, timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--disable_save", action="store_true", help="Disable saving checkpoints, data, and images")
    parser.add_argument("--model_type", type=str, default="model0", choices=["model0", "model1", "model2"], help="Type of model to use (model0, model1, model2)")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["multimodal", "text_only", "image_only"], help="Input mode: multimodal, text_only, or image_only")
    args = parser.parse_args()

    train(batch_size=args.batch_size, epochs=args.epochs, accumulation_steps=args.accumulation_steps, lr=args.lr, val_ratio=args.val_ratio, disable_save=args.disable_save, model_type=args.model_type, mode=args.mode)