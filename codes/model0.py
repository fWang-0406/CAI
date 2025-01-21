import torch
import torch.nn as nn

class MultimodalClassifier0(nn.Module):
    def __init__(self, text_feature_dim, image_feature_dim, num_classes):
        super(MultimodalClassifier0, self).__init__()
        self.text_feature_dim = text_feature_dim
        self.image_feature_dim = image_feature_dim
        self.num_classes = num_classes

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(text_feature_dim + image_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, text_features, image_features, mode='multimodal'):
        # 根据模式选择处理方式
        if mode == 'text_only':
            if image_features is None:
                image_features = torch.zeros(text_features.size(0), self.image_feature_dim).to(text_features.device)
            else:
                raise ValueError("In 'text_only' mode, image_features should be None.")
        elif mode == 'image_only':
            if text_features is None:
                text_features = torch.zeros(image_features.size(0), self.text_feature_dim).to(image_features.device)
            else:
                raise ValueError("In 'image_only' mode, text_features should be None.")
        elif mode == 'multimodal':
            if text_features is None or image_features is None:
                raise ValueError("In 'multimodal' mode, both text_features and image_features are required.")
        else:
            raise ValueError("Invalid mode. Choose from 'text_only', 'image_only', or 'multimodal'.")

        # 拼接文本和图像特征
        fused_features = torch.cat((text_features, image_features), dim=1)
        # 返回全连接层输出
        return self.fc(fused_features)