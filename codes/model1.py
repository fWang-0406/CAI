import torch
import torch.nn as nn

class MultimodalClassifier1(nn.Module):
    def __init__(self, text_dim=768, image_dim=1280, hidden_dim=256, num_classes=3):
        """
        初始化 MultimodalClassifier1 模型。
        :param text_dim: 文本特征维度
        :param image_dim: 图像特征维度
        :param hidden_dim: 隐藏层维度
        :param num_classes: 类别数量
        """
        super(MultimodalClassifier1, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 文本特征处理层
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 图像特征处理层
        self.image_fc = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_features, image_features, mode='multimodal'):
        """
        前向传播。
        :param text_features: 文本特征
        :param image_features: 图像特征
        :param mode: 模式（text_only, image_only, multimodal）
        :return: 分类器输出
        """
        # 根据模式选择处理方式
        if mode == 'text_only':
            features = self.text_fc(text_features)
        elif mode == 'image_only':
            features = self.image_fc(image_features)
        elif mode == 'multimodal':
            # 处理文本和图像特征
            text_features = self.text_fc(text_features)
            image_features = self.image_fc(image_features)
            # 拼接文本和图像特征
            features = torch.cat((text_features, image_features), dim=1)
        else:
            raise ValueError("Invalid mode. Choose from 'text_only', 'image_only', or 'multimodal'.")

        # 返回分类器输出
        return self.classifier(features)