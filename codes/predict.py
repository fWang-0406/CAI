import torch
import pandas as pd
import os
import numpy as np
import argparse  # 导入 argparse 模块

from extract_features import TextFeatureExtractor, ImageFeatureExtractor
from model2 import MultimodalClassifier2  # 确保导入正确的模型

def load_test_data(test_csv):
    """
    加载测试数据 CSV 文件。
    :param test_csv: 测试数据 CSV 文件路径
    :return: DataFrame，包含 guid, text, image_path
    """
    return pd.read_csv(test_csv)

def extract_features_with_cache(test_data, cache_dir, force_reload=False):
    """
    提取特征并支持缓存功能。
    :param test_data: 测试数据 DataFrame
    :param cache_dir: 缓存文件夹路径
    :param force_reload: 是否强制重新生成缓存文件
    :return: 文本特征、图像特征
    """
    # 确保缓存文件夹存在
    os.makedirs(cache_dir, exist_ok=True)

    # 生成缓存文件路径
    cache_text_path = os.path.join(cache_dir, "test_text_features.pt")
    cache_image_path = os.path.join(cache_dir, "test_image_features.pt")

    # 如果缓存文件存在且不需要强制重新加载，则直接加载
    if not force_reload and os.path.exists(cache_text_path) and os.path.exists(cache_image_path):
        print("Loading features from cache...")
        text_features = torch.load(cache_text_path)
        image_features = torch.load(cache_image_path)
    else:
        # 提取文本特征
        print("Extracting text features...")
        text_extractor = TextFeatureExtractor()
        text_features = text_extractor.extract_features(test_data['text'].tolist())

        # 提取图像特征
        print("Extracting image features...")
        image_extractor = ImageFeatureExtractor()
        image_features = image_extractor.extract_features(test_data['image_path'].tolist())

        # 保存特征到缓存文件
        print("Saving features to cache...")
        torch.save(text_features, cache_text_path)
        torch.save(image_features, cache_image_path)

    return text_features, image_features

def predict(test_csv, model_type="model2", model_name="multimodal_model_model0_20250120_213422.pth", cache_dir="../cache", force_reload=False):
    """
    对测试数据进行预测并保存结果。
    :param test_csv: 测试数据 CSV 文件路径
    :param model_type: 模型类型（model0, model1, model2）
    :param model_name: 训练好的模型文件名
    :param cache_dir: 缓存文件夹路径
    :param force_reload: 是否强制重新生成缓存文件
    """
    # 加载测试数据
    test_data = load_test_data(test_csv)

    # 提取特征（支持缓存）
    text_features, image_features = extract_features_with_cache(test_data, cache_dir, force_reload)

    # 将特征转换为 PyTorch 张量
    text_features = text_features.clone().detach().float()  # 修复警告
    image_features = image_features.clone().detach().float()  # 修复警告

    # 加载模型
    print("Starting load models...")
    text_feature_dim = text_features.shape[1]
    image_feature_dim = image_features.shape[1]
    num_classes = 3

    # 根据 model_type 选择模型
    if model_type == "model2":
        model = MultimodalClassifier2(text_feature_dim, image_feature_dim, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 加载最新保存的模型
    print("Starting load saved models...")
    current_dir = os.path.dirname(__file__)
    model_dir = os.path.join(current_dir, '..', 'models')
    model_path = os.path.join(model_dir, model_name)  # 使用传入的模型文件名

    # 加载模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    # 预测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    text_features = text_features.to(device)
    image_features = image_features.to(device)

    with torch.no_grad():
        outputs = model(text_features, image_features)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # 将预测结果映射为情感标签
    label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
    test_data['tag'] = preds
    test_data['tag'] = test_data['tag'].map(label_map)

    # 保存预测结果
    result_path = os.path.join(current_dir, '..', 'AAAdata', 'result.txt')
    test_data[['guid', 'tag']].to_csv(result_path, index=False, sep=',')
    print(f"测试集预测结果已保存为 {result_path}")

if __name__ == "__main__":
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Predict using a trained multimodal model.")
    parser.add_argument("--model_type", type=str, default="model2", choices=["model2"], help="Type of model to use (model2)")
    parser.add_argument("--model_name", type=str, default="multimodal_model_model2_20250120_132611.pth", help="Name of the trained model file")
    parser.add_argument("--test_csv", type=str, default=os.path.join(os.path.dirname(__file__), '..', 'AAAdata', 'test_data.csv'), help="Path to the test data CSV file")
    args = parser.parse_args()

    # 生成预测结果
    predict(test_csv=args.test_csv, model_type=args.model_type, model_name=args.model_name)