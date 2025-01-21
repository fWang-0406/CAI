import os
import torch
from PIL import Image
from torch import nn
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from torch.nn import Module, Linear, Softmax, AdaptiveAvgPool2d, Sequential, ReLU, Sigmoid, MultiheadAttention, Conv2d

# ---------------------- 文本特征提取器（加入多头注意力池化） ----------------------
class TextFeatureExtractor:
    def __init__(self, device='cuda', max_length=128):
        """
        初始化文本特征提取器。
        :param device: 设备（cuda 或 cpu）
        :param max_length: 最大文本长度
        """
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'pretrain_models', 'bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertModel.from_pretrained(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

        # 多头注意力池化层
        self.attention_pool = MultiheadAttention(768, num_heads=4).to(self.device)

    def extract_features(self, texts, batch_size=32):
        """
        提取文本特征。
        :param texts: 文本列表
        :param batch_size: 批量大小
        :return: 文本特征张量
        """
        print("Starting extract text_features...")
        features = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state

                # 多头注意力池化
                attn_output, _ = self.attention_pool(hidden_states, hidden_states, hidden_states)
                batch_features = attn_output.mean(dim=1)

                features.append(batch_features.cpu())

            torch.cuda.empty_cache()

        features = torch.cat(features, dim=0)
        print("Extracted text_feature successfully...")
        return features

# ---------------------- 图像特征提取器（加入CBAM模块） ----------------------
class CBAM(Module):
    def __init__(self, channel):
        """
        初始化 CBAM 模块。
        :param channel: 输入通道数
        """
        super().__init__()
        # 通道注意力
        self.channel_att = Sequential(
            AdaptiveAvgPool2d(1),
            nn.Flatten(),
            Linear(channel, channel // 16),
            ReLU(),
            Linear(channel // 16, channel),
            Sigmoid()
        )
        # 空间注意力
        self.spatial_att = Sequential(
            Conv2d(2, 1, kernel_size=7, padding=3),
            Sigmoid()
        )

    def forward(self, x):
        """
        前向传播。
        :param x: 输入特征
        :return: 经过 CBAM 处理后的特征
        """
        # 通道注意力
        channel_att = self.channel_att(x).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_att
        # 空间注意力（保持不变）
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        return x * spatial_att

class ImageFeatureExtractor:
    def __init__(self, device='cuda'):
        """
        初始化图像特征提取器。
        :param device: 设备（cuda 或 cpu）
        """
        model_path = os.path.join(os.path.dirname(__file__), '..', 'pretrain_models', 'resnet50.pth')
        self.model = models.resnet50(pretrained=False)
        self.model.load_state_dict(torch.load(model_path))

        # 在 layer4 后添加 CBAM 模块
        self.model.layer4.add_module("cbam", CBAM(2048))
        self.model.fc = Sequential()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.failed_images = []

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_paths):
        """
        提取图像特征。
        :param image_paths: 图像路径列表
        :return: 图像特征张量
        """
        print("Starting extract image_features...")
        features = []
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                image = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feature = self.model(image)
                features.append(feature.squeeze().cpu())
            except Exception as e:
                print(f"Warning: Failed to process image {path}. Error: {e}")
                self.failed_images.append(path)
                features.append(torch.zeros(2048))
            torch.cuda.empty_cache()
        return torch.stack(features)