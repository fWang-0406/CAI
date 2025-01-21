import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=256):
        """
        初始化交叉注意力模块。
        :param hidden_dim: 隐藏层维度
        """
        super().__init__()
        # 定义多头注意力层，包含4个头和0.2的dropout
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.2)

    def forward(self, text_features, image_features):
        """
        前向传播。
        :param text_features: 文本特征
        :param image_features: 图像特征
        :return: 注意力输出
        """
        # 调整输入维度为 [seq_len, batch, hidden_dim]
        text_in = text_features.unsqueeze(0)
        image_in = image_features.unsqueeze(0)

        # 计算注意力输出
        attn_output, _ = self.attention(
            query=text_in,
            key=image_in,
            value=image_in
        )
        return attn_output.squeeze(0)

class MultimodalClassifier2(nn.Module):
    def __init__(self, text_dim, image_dim, num_classes, hidden_dim=256):
        """
        初始化 MultimodalClassifier2 模型。
        :param text_dim: 文本特征维度
        :param image_dim: 图像特征维度
        :param num_classes: 类别数量
        :param hidden_dim: 隐藏层维度
        """
        super(MultimodalClassifier2, self).__init__()
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

        # 交叉注意力机制
        self.cross_attention = CrossAttention(hidden_dim)

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
            # 计算交叉注意力特征
            attn_features = self.cross_attention(text_features, image_features)
            # 拼接文本和注意力特征
            features = torch.cat((text_features, attn_features), dim=1)
        else:
            raise ValueError("Invalid mode. Choose from 'text_only', 'image_only', or 'multimodal'.")

        # 返回分类器输出
        return self.classifier(features)