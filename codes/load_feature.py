import os
import pandas as pd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from extract_features import TextFeatureExtractor, ImageFeatureExtractor
from torch.utils.data import random_split
import numpy as np

class CrossModalProjector(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, hidden_dim=256):
        """
        初始化跨模态投影器。
        :param text_dim: 文本特征维度
        :param image_dim: 图像特征维度
        :param hidden_dim: 隐藏层维度
        """
        super().__init__()
        # 文本特征投影层
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        # 图像特征投影层
        self.image_proj = nn.Linear(image_dim, hidden_dim)

    def forward(self, text_feat, image_feat):
        """
        前向传播。
        :param text_feat: 文本特征
        :param image_feat: 图像特征
        :return: 投影后的文本和图像特征
        """
        # 返回投影后的文本和图像特征
        return self.text_proj(text_feat), self.image_proj(image_feat)

def load_feature(data_path, batch_size=32, force_reload=False, val_ratio=0.2):
    """
    特征标准化分离 + SMOTE统一应用。
    :param data_path: 数据路径
    :param batch_size: 批量大小
    :param force_reload: 是否强制重新加载
    :param val_ratio: 验证集比例
    :return: 训练和验证的文本特征、图像特征、标签以及类别权重
    """
    # 缓存路径设置
    cache_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'cache0'))
    os.makedirs(cache_dir, exist_ok=True)

    cache_train_text_path = os.path.join(cache_dir, "train_text_features.pt")
    cache_train_image_path = os.path.join(cache_dir, "train_image_features.pt")
    cache_train_labels_path = os.path.join(cache_dir, "train_labels.pt")
    cache_val_text_path = os.path.join(cache_dir, "val_text_features.pt")
    cache_val_image_path = os.path.join(cache_dir, "val_image_features.pt")
    cache_val_labels_path = os.path.join(cache_dir, "val_labels.pt")
    cache_class_weights_path = os.path.join(cache_dir, "class_weights.pt")

    # 如果缓存文件存在且不需要强制重新加载，则直接加载
    if not force_reload and all([
        os.path.exists(p) for p in [
            cache_train_text_path, cache_train_image_path, cache_train_labels_path,
            cache_val_text_path, cache_val_image_path, cache_val_labels_path,
            cache_class_weights_path
        ]
    ]):
        return (
            torch.load(cache_train_text_path),
            torch.load(cache_train_image_path),
            torch.load(cache_train_labels_path),
            torch.load(cache_val_text_path),
            torch.load(cache_val_image_path),
            torch.load(cache_val_labels_path),
            torch.load(cache_class_weights_path)
        )

    # ---------------------- 数据加载与预处理 ----------------------
    data = pd.read_csv(data_path)
    project_root = os.path.normpath(os.path.dirname(__file__))
    data['image_path'] = data['image_path'].apply(
        lambda x: os.path.normpath(os.path.abspath(os.path.join(project_root, x.replace('\\', os.sep))))
    )
    missing_images = data[~data['image_path'].apply(os.path.exists)]
    if not missing_images.empty:
        raise FileNotFoundError(f"以下图片路径不存在：\n{missing_images['image_path'].tolist()}")

    # 标签编码
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['label'])
    class_names = label_encoder.classes_
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = (1.0 / class_counts.float()).to(torch.float32)
    class_weights = class_weights / class_weights.sum()

    # 数据集划分
    dataset = list(zip(data['text'].tolist(), data['image_path'].tolist(), labels))
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # 训练集特征提取
    train_texts, train_image_paths, train_labels = zip(*train_dataset)
    text_extractor = TextFeatureExtractor()
    image_extractor = ImageFeatureExtractor()
    train_text_features = text_extractor.extract_features(train_texts, batch_size=batch_size)
    train_image_features = image_extractor.extract_features(train_image_paths)

    # ---------------------- 特征标准化（独立Scaler）----------------------
    text_scaler = StandardScaler()
    image_scaler = StandardScaler()

    train_text_features = torch.tensor(
        text_scaler.fit_transform(train_text_features.numpy()),
        dtype=torch.float32
    )
    train_image_features = torch.tensor(
        image_scaler.fit_transform(train_image_features.numpy()),
        dtype=torch.float32
    )

    # ---------------------- 合并特征并应用SMOTE ----------------------
    print("Applying SMOTE with combined features...")
    combined_features = np.hstack([train_text_features.numpy(), train_image_features.numpy()])
    smote = SMOTE(random_state=42)
    combined_resampled, labels_resampled = smote.fit_resample(combined_features, np.array(train_labels))

    # 拆分回文本和图像特征
    text_dim = train_text_features.shape[1]
    train_text_resampled = combined_resampled[:, :text_dim]
    train_image_resampled = combined_resampled[:, text_dim:]

    # 转换为Tensor
    train_text_features_resampled = torch.tensor(train_text_resampled, requires_grad=False)
    train_image_features_resampled = torch.tensor(train_image_resampled, requires_grad=False)
    train_labels_resampled = torch.tensor(labels_resampled, requires_grad=False)

    # ---------------------- 验证集处理 ----------------------
    val_texts, val_image_paths, val_labels = zip(*val_dataset)
    val_text_features = text_extractor.extract_features(val_texts, batch_size=batch_size)
    val_image_features = image_extractor.extract_features(val_image_paths)

    val_text_features = torch.tensor(
        text_scaler.transform(val_text_features.numpy()),  # 使用文本Scaler
        dtype=torch.float32
    )
    val_image_features = torch.tensor(
        image_scaler.transform(val_image_features.numpy()),  # 使用图像Scaler
        dtype=torch.float32
    )

    # ---------------------- 跨模态投影对齐 ----------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    projector = CrossModalProjector().to(device)
    with torch.no_grad():
        train_text_proj, train_image_proj = projector(
            train_text_features_resampled.to(device),
            train_image_features_resampled.to(device)
        )
        val_text_proj, val_image_proj = projector(
            val_text_features.to(device),
            val_image_features.to(device)
        )

    # ---------------------- 缓存数据 ----------------------
    torch.save(train_text_proj.cpu(), cache_train_text_path)
    torch.save(train_image_proj.cpu(), cache_train_image_path)
    torch.save(train_labels_resampled.cpu(), cache_train_labels_path)
    torch.save(val_text_proj.cpu(), cache_val_text_path)
    torch.save(val_image_proj.cpu(), cache_val_image_path)
    torch.save(torch.tensor(val_labels).cpu(), cache_val_labels_path)
    torch.save(class_weights.cpu(), cache_class_weights_path)

    return (
        train_text_proj.cpu(),
        train_image_proj.cpu(),
        train_labels_resampled.cpu(),
        val_text_proj.cpu(),
        val_image_proj.cpu(),
        torch.tensor(val_labels).cpu(),
        class_weights.cpu()
    )


if __name__ == "__main__":
    data_path = "../AAAdata/train_data.csv"
    features = load_feature(data_path)
    print(f"\n训练集文本特征形状: {features[0].shape}")
    print(f"训练集图像特征形状: {features[1].shape}")
    print(f"训练集标签形状: {features[2].shape}")
    print(f"验证集文本特征形状: {features[3].shape}")
    print(f"验证集图像特征形状: {features[4].shape}")
    print(f"验证集标签形状: {features[5].shape}")
    print(f"\n类别权重: {features[6]}")