import os
import torch
import csv
import matplotlib.pyplot as plt

def save_checkpoint(checkpoint_dir, epoch, model, optimizer, train_loss, val_loss, train_accuracy, val_accuracy):
    """
    保存检查点。
    :param checkpoint_dir: 检查点保存目录
    :param epoch: 当前训练的 epoch
    :param model: 模型对象
    :param optimizer: 优化器对象
    :param train_loss: 训练损失
    :param val_loss: 验证损失
    :param train_accuracy: 训练准确率
    :param val_accuracy: 验证准确率
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def save_model(model_dir, model, model_type, timestamp):
    """
    保存最终模型。
    :param model_dir: 模型保存目录
    :param model: 模型对象
    :param model_type: 模型类型
    :param timestamp: 时间戳
    """
    os.makedirs(model_dir, exist_ok=True)
    model_name = f"multimodal_model_{model_type}_{timestamp}.pth"
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def save_loss_accuracy_data(current_dir, train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s, model_type, timestamp):
    """
    保存损失、准确率和 F1 分数数据到 CSV 文件。
    :param current_dir: 当前目录
    :param train_losses: 训练损失列表
    :param val_losses: 验证损失列表
    :param train_accuracies: 训练准确率列表
    :param val_accuracies: 验证准确率列表
    :param train_f1s: 训练 F1 分数列表
    :param val_f1s: 验证 F1 分数列表
    :param model_type: 模型类型
    :param timestamp: 时间戳
    """
    loss_data_path = os.path.join(current_dir, '..', 'data', f"epoch_losses_accuracies_f1_{model_type}_{timestamp}.csv")
    with open(loss_data_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy', 'Train F1', 'Val F1'])
        for epoch, (train_loss, val_loss, train_acc, val_acc, train_f1, val_f1) in enumerate(zip(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s)):
            writer.writerow([epoch + 1, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1])
    print(f"Loss, accuracy, and F1 data saved to {loss_data_path}")

def plot_loss_accuracy(current_dir, train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s, model_type, timestamp):
    """
    绘制并保存损失、准确率和 F1 分数图像。
    :param current_dir: 当前目录
    :param train_losses: 训练损失列表
    :param val_losses: 验证损失列表
    :param train_accuracies: 训练准确率列表
    :param val_accuracies: 验证准确率列表
    :param train_f1s: 训练 F1 分数列表
    :param val_f1s: 验证 F1 分数列表
    :param model_type: 模型类型
    :param timestamp: 时间戳
    """
    plt.figure(figsize=(18, 6))
    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='#FF9A9A', label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', linestyle='-', color='#C99BFA', label='Val Loss')
    plt.title('Epoch vs Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', linestyle='-', color='#FF9A9F', label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o', linestyle='-', color='#C99BFF', label='Val Accuracy')
    plt.title('Epoch vs Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # 绘制 F1 分数曲线
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(train_f1s) + 1), train_f1s, marker='o', linestyle='-', color='#FF9A9F', label='Train F1')
    plt.plot(range(1, len(val_f1s) + 1), val_f1s, marker='o', linestyle='-', color='#C99BFF', label='Val F1')
    plt.title('Epoch vs F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    # 保存图像
    loss_image_path = os.path.join(current_dir, '..', 'imgs', f"epoch_losses_accuracies_f1_{model_type}_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(loss_image_path)
    print(f"Loss, accuracy, and F1 image saved to {loss_image_path}")