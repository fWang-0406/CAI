import os
import pandas as pd
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def read_guid_label(train_file):
    """
    读取 train.txt 文件，获取 guid 和情感标签。
    :param train_file: train.txt 文件路径
    :return: 字典，键为 guid，值为标签
    """
    guid_to_label = {}
    with open(train_file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过标题行
        for line in f:
            guid, label = line.strip().split(',')
            guid_to_label[guid] = label
    return guid_to_label

def read_text_file(txt_file):
    """
    读取文本文件内容，尝试多种编码方式。
    :param txt_file: 文本文件路径
    :return: 文本内容
    """
    try:
        # 1：尝试 UTF-8 编码
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    except UnicodeDecodeError:
        try:
            # 2：尝试 GBK 编码
            with open(txt_file, 'r', encoding='gbk') as f:
                text = f.read().strip()
        except UnicodeDecodeError:
            try:
                # 3：尝试 ISO-8859-1 编码
                with open(txt_file, 'r', encoding='iso-8859-1') as f:
                    text = f.read().strip()
            except UnicodeDecodeError:
                # 4：二进制模式读取并忽略错误
                with open(txt_file, 'rb') as f:
                    raw_data = f.read()
                    text = raw_data.decode('utf-8', errors='ignore').strip()
    return text

def analyze_text_lengths(texts):
    """
    统计文本长度范围和 85 分位数长度。
    :param texts: 文本列表
    :return: 文本长度范围（最小值，最大值），85 分位数长度
    """
    text_lengths = [len(text) for text in texts]
    min_length = min(text_lengths)
    max_length = max(text_lengths)
    quantile_85 = np.percentile(text_lengths, 85)
    return text_lengths, (min_length, max_length), quantile_85

def analyze_image_metadata(image_paths):
    """
    统计图片的元数据（尺寸、颜色通道等）。
    :param image_paths: 图片路径列表
    :return: 图片尺寸列表，图片尺寸范围（最小尺寸，最大尺寸），颜色通道统计（字典，键为通道数，值为图片数量）
    """
    sizes = []
    channels = defaultdict(int)
    failed_images = []  # 用于存储无法读取的图片路径

    for path in image_paths:
        try:
            with Image.open(path) as img:
                sizes.append(img.size)  # 图片尺寸 (width, height)
                channels[len(img.getbands())] += 1  # 颜色通道数
        except Exception as e:
            print(f"Warning: Failed to process image {path}. Error: {e}")
            failed_images.append(path)

    # 统计图片尺寸范围
    min_size = (min([size[0] for size in sizes]), min([size[1] for size in sizes]))
    max_size = (max([size[0] for size in sizes]), max([size[1] for size in sizes]))

    # 输出无法读取的图片路径
    if failed_images:
        print(f"无法读取的图片数量: {len(failed_images)}")
        print("无法读取的图片路径:")
        for path in failed_images:
            print(path)

    return sizes, (min_size, max_size), channels

def plot_label_and_text_length_distribution(labels, text_lengths, output_dir):
    """
    绘制标签分布的饼图和文本长度分布的柱状图，并将它们合并到一张图片中。
    :param labels: 标签列表
    :param text_lengths: 文本长度列表
    :param output_dir: 输出目录
    """
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1行2列的子图布局

    # 绘制饼图（标签分布）
    label_counts = pd.Series(labels).value_counts()
    colors = ['#FF9A9F', '#C99BFF', '#3ABF99']  # 使用指定颜色
    axes[0].pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    axes[0].set_title("Label Distribution")

    # 绘制柱状图（文本长度分布）
    axes[1].hist(text_lengths, bins=50, color='#3ABF99', edgecolor='black')  # 使用指定颜色
    axes[1].set_title("Text Length Distribution")
    axes[1].set_xlabel("Text Length")
    axes[1].set_ylabel("Frequency")

    # 调整布局并保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "label_and_text_length_distribution.png"))
    plt.close()

def plot_image_size_distribution(image_sizes, output_dir):
    """
    绘制图片尺寸分布的 Hexbin 图和二维核密度图，并将两张图合并到一张图片中。
    :param image_sizes: 图片尺寸列表
    :param output_dir: 输出目录
    """
    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]

    # 自定义颜色映射
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#E0F3FF", "#3ABF99", "#C99BFF", "#FF9A9F"]  # 从浅到深
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))  # 1行2列的子图布局

    # Hexbin 图
    hexbin = axes[0].hexbin(
        widths,
        heights,
        gridsize=100,
        cmap=custom_cmap,
        mincnt=0,
        bins='log',
        vmin=1,
        vmax=50
    )
    fig.colorbar(hexbin, ax=axes[0], label='Count in bin', extend='max')  # 添加颜色条并扩展范围
    axes[0].set_title("Image Size Distribution (Hexbin Plot)")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")
    axes[0].grid(True)

    # 二维核密度图
    sns.kdeplot(
        x=widths,
        y=heights,
        cmap=custom_cmap,  # 使用自定义颜色映射
        fill=True,
        thresh=0.05,
        levels=10,
        ax=axes[1]  # 在第二个子图中绘制
    )
    axes[1].set_title("Image Size Distribution (2D Density Plot)")
    axes[1].set_xlabel("Width")
    axes[1].set_ylabel("Height")
    axes[1].grid(True)

    # 保存合并后的图片
    plt.tight_layout()  # 调整子图间距
    plt.savefig(os.path.join(output_dir, "image_size_distribution_combined.png"))
    plt.close()

def prepare_data(guid_to_label, data_folder, output_csv, imgs_dir):
    """
    准备数据并保存为 CSV 文件。
    :param guid_to_label: 字典，键为 guid，值为标签
    :param data_folder: data 文件夹路径
    :param output_csv: 输出 CSV 文件路径
    :param imgs_dir: 图片保存目录
    :return: 包含文本长度范围和 85 分位数长度的统计结果，以及图片元数据统计结果
    """
    data = []
    texts = []  # 用于存储所有文本内容
    image_paths = []  # 用于存储所有图片路径
    labels = []  # 用于存储所有标签

    # 遍历 guid_to_label 中的所有 guid
    for guid, label in guid_to_label.items():
        # 构建文本文件路径
        txt_file = os.path.join(data_folder, f"{guid}.txt")
        # 构建图片文件路径
        img_file = os.path.join(data_folder, f"{guid}.jpg")

        # 检查文本和图片文件是否存在
        if os.path.exists(txt_file) and os.path.exists(img_file):
            # 读取文本内容
            text = read_text_file(txt_file)
            texts.append(text)

            # 将图片路径保存为相对路径
            relative_img_path = os.path.relpath(img_file, start=os.path.dirname(output_csv))
            image_paths.append(img_file)

            # 将数据添加到列表中
            data.append({
                'guid': guid,
                'text': text,
                'image_path': relative_img_path,
                'label': label
            })
            labels.append(label)
        else:
            print(f"Warning: Missing files for guid {guid}")

    # 统计文本长度范围和 85 分位数长度
    text_lengths, length_range, quantile_85 = analyze_text_lengths(texts)
    print(f"文本长度范围: {length_range[0]} - {length_range[1]}")
    print(f"85 分位数长度: {quantile_85}")

    # 统计图片元数据
    image_sizes, image_size_range, image_channels = analyze_image_metadata(image_paths)
    print(f"图片尺寸范围: 最小 {image_size_range[0]}, 最大 {image_size_range[1]}")
    print("图片颜色通道统计:")
    for channel, count in image_channels.items():
        print(f"{channel} 通道: {count} 张图片")

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)

    # 保存为 CSV 文件
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"CSV 文件已生成: {output_csv}")

    # 可视化标签分布和文本长度分布
    plot_label_and_text_length_distribution(labels, text_lengths, imgs_dir)

    # 可视化图片尺寸分布
    plot_image_size_distribution(image_sizes, imgs_dir)

    return length_range, quantile_85, image_size_range, image_channels

def prepare_test_data(test_file, data_folder, output_csv):
    """
    准备测试数据并保存为 CSV 文件。
    :param test_file: test_without_label.txt 文件路径
    :param data_folder: 数据文件夹路径
    :param output_csv: 输出 CSV 文件路径
    """
    data = []
    texts = []  # 用于存储所有文本内容
    image_paths = []  # 用于存储所有图片路径

    # 读取测试文件
    with open(test_file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过标题行
        for line in f:
            guid, _ = line.strip().split(',')  # 使用逗号分隔
            # 构建文本文件路径
            txt_file = os.path.join(data_folder, f"{guid}.txt")
            # 构建图片文件路径
            img_file = os.path.join(data_folder, f"{guid}.jpg")

            # 检查文本和图片文件是否存在
            if os.path.exists(txt_file) and os.path.exists(img_file):
                # 读取文本内容
                text = read_text_file(txt_file)
                texts.append(text)

                # 将图片路径保存为相对路径
                relative_img_path = os.path.relpath(img_file, start=os.path.dirname(output_csv))

                # 将数据添加到列表中
                data.append({
                    'guid': guid,
                    'text': text,
                    'image_path': relative_img_path
                })
            else:
                print(f"Warning: Missing files for guid {guid}")

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 保存为 CSV 文件
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"测试数据 CSV 文件已生成: {output_csv}")

def main():
    # 定义路径
    current_dir = os.path.dirname(__file__)
    data_folder = os.path.join(current_dir, '..', 'original_data', 'data')
    train_file = os.path.join(current_dir, '..', 'original_data', 'train.txt')
    test_file = os.path.join(current_dir, '..', 'original_data', 'test_without_label.txt')
    output_csv = os.path.join(current_dir, '..', 'AAAdata', 'train_data.csv')
    test_output_csv = os.path.join(current_dir, '..', 'AAAdata', 'test_data.csv')
    imgs_dir = os.path.join(current_dir, '..', 'imgs')  # 图片保存目录

    # 读取 guid 和标签
    guid_to_label = read_guid_label(train_file)

    # 准备训练数据并保存为 CSV 文件
    prepare_data(guid_to_label, data_folder, output_csv, imgs_dir)

    # 准备测试数据并保存为 CSV 文件
    prepare_test_data(test_file, data_folder, test_output_csv)

if __name__ == "__main__":
    main()