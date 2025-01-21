## 实验5：多模态情感分析

这是2024秋当代人工智能课程实验5的仓库。  
目标是给定配对的文本和图像，预测对应的情感标签。  
三分类任务：positive, neutral, negative。

---

## 实验环境

本项目依赖 Python 3.x，需要以下依赖：

- `torch~=2.0.1+cu118`
- `matplotlib~=3.9.2`
- `tqdm~=4.66.5`
- `pandas~=2.2.3`
- `optuna~=4.1.0`
- `numpy~=1.24.1`
- `pillow~=11.0.0`
- `torchvision~=0.15.2+cu118`
- `transformers~=4.45.2`

可以通过以下命令安装依赖：

```bash
pip install -r requirements.txt
```

---

## 文件目录结构

详见 [file_structure.txt](file_structure.txt)

```
└── task5/
│   └── .idea\ (未展开)  # PyCharm IDE 的配置文件目录，通常包含项目配置和设置
│   └── AAAdata\  # 存放处理后的数据集和预测结果
│   │   └── result.txt  # 预测结果文件，包含测试集的预测标签
│   │   └── test_data.csv  # 测试集数据文件，包含测试集的文本和图像路径
│   │   └── train_data.csv  # 训练集数据文件，包含训练集的文本、图像路径和标签
│   └── best_params.txt  # 保存模型调优后的最佳超参数
│   └── cache\  # 缓存目录，用于存储预处理后的数据或特征
│   │   └── class_weights.pt  # 类别权重文件，用于处理类别不平衡问题
│   │   └── image_features.pt  # 图像特征文件，存储提取的图像特征
│   │   └── labels.pt  # 标签数据文件，存储训练集和验证集的标签
│   │   └── test_image_features.pt  # 测试集图像特征文件
│   │   └── test_text_features.pt  # 测试集文本特征文件
│   │   └── text_features.pt  # 文本特征文件，存储提取的文本特征
│   │   └── train_image_features.pt  # 训练集图像特征文件
│   │   └── train_labels.pt  # 训练集标签文件
│   │   └── train_text_features.pt  # 训练集文本特征文件
│   │   └── val_image_features.pt  # 验证集图像特征文件
│   │   └── val_labels.pt  # 验证集标签文件
│   │   └── val_text_features.pt  # 验证集文本特征文件
│   └── checkpoints\ (未展开)  # 训练过程中保存的模型检查点
│   └── codes\  # 项目代码目录
│   │   └── autoTunner.py  # 自动调参代码，用于优化模型超参数
│   │   └── data_preprocess.py  # 数据预处理代码，用于生成训练集和测试集的 CSV 文件
│   │   └── extract_features.py  # 特征提取代码，用于从文本和图像中提取特征
│   │   └── load_feature.py  # 加载特征的代码，支持缓存功能
│   │   └── model0.py  # 模型0的定义和实现，简单的全连接网络
│   │   └── model1.py  # 模型1的定义和实现，增加了特征提取的灵活性
│   │   └── model2.py  # 模型2的定义和实现，增加了注意力机制
│   │   └── performance_analysis.py  # 性能分析代码，用于分析模型性能
│   │   └── predict.py  # 预测代码，用于对测试数据进行预测
│   │   └── tmp.py  # 临时代码，用于测试或调试
│   │   └── train.py  # 模型训练代码，用于训练模型
│   │   └── utils.py  # 工具函数代码，包含保存模型、绘制图表等功能
│   └── data\ (未展开)  # 存放训练过程中的损失、准确率等数据
│   └── file_structure.txt  # 项目文件结构说明文件
│   └── img.png  # 项目相关的图片文件
│   └── imgs\ (未展开)  # 图片资源目录，存放可视化结果
│   └── models\ (未展开)  # 训练过程中保存的模型文件
│   └── original_data\ (未展开)  # 原始数据集目录
│   └── pretrain_models\ (未展开)  # 预训练模型目录
│   └── README.md  # 项目说明文档
│   └── requirements.txt  # 项目依赖库列表
│   └── tree.py  # 生成文件目录结构的脚本
```

---

## 代码执行流程

进入项目根目录 `task5`。

### 1. 对原始数据集进行预处理（可选）

运行以下命令，将生成文本-图像配对的 CSV 文件，用于后续的深度学习。  
当然，经过处理的数据文件已经保存在 `data/` 中，可以跳过这一步。

```bash
python codes/data_preprocess.py
```

### 2. 训练模型

运行以下命令，将构建模型并训练：

```bash
python codes/train.py --model_type="model0" --epochs=100 --batch_size=32 --lr=0.001 --accumulation_steps=1
```

#### 参数说明：
- `--model_type`: 选择模型类型，可选 `model0`、`model1`、`model2`，默认为 `model0`。
- `--epochs`: 训练的总轮数，默认为 50。
- `--batch_size`: 每个批次的样本数，默认为 16。
- `--lr`: 学习率，默认为 0.0001。
- `--accumulation_steps`: 梯度累积步数，默认为 16。
- `--val_ratio`: 验证集比例，默认为 0.2。
- `--disable_save`: 禁用保存检查点、数据和图像，默认为 False。
- `--mode`: 输入模式，可选 `multimodal`（多模态）、`text_only`（仅文本）、`image_only`（仅图像），默认为 `multimodal`。

### 3. 自动调参

为了提高训练速度，本实验调参过程中使用了多线程策略。运行以下命令进行自动调参：

```bash
python codes/autoTunner.py --n_trials=100 --n_jobs=8 --model_type="model2" --optimize_target="f1"
```

#### 参数说明：
- `--n_trials`: 调参的试验次数，默认为 100。
- `--n_jobs`: 并行任务数，默认为 8。请根据计算机内核数合理设置，一般不超过内核数。
- `--model_type`: 选择模型类型，可选 `model0`、`model1`、`model2`，默认为 `model0`。
- `--optimize_target`: 优化目标，可选 `accuracy`（准确率）或 `f1`（F1 分数），默认为 `f1`。

### 4. 预测

训练完成后，可以使用以下命令对测试数据进行预测：

```bash
python codes/predict.py --model_type="model2" --model_name="multimodal_model_model2_20250120_132611.pth"
```

#### 参数说明：
- `--model_type`: 选择模型类型，可选 `model0`、`model1`、`model2`，默认为 `model2`。
- `--model_name`: 训练好的模型文件名，默认为 `multimodal_model_model2_20250120_132611.pth`。
- `--test_csv`: 测试数据 CSV 文件路径，默认为 `AAAdata/test_data.csv`。

---

## 模型说明

### 1. `MultimodalClassifier0`
- **结构**: 简单的全连接网络，直接拼接文本和图像特征。
- **特点**: 适合基线模型，结构简单，易于训练。

### 2. `MultimodalClassifier1`
- **结构**: 分别对文本和图像特征进行全连接处理，再拼接后进行最终分类。
- **特点**: 增加了特征提取的灵活性，适合中等复杂度的任务。

### 3. `MultimodalClassifier2`
- **结构**: 在 `MultimodalClassifier1` 的基础上增加了注意力机制。
- **特点**: 通过注意力机制增强多模态特征的融合效果，适合复杂任务。

---

## 注意事项

1. **多线程调参**: 为了提高训练速度，调参过程中使用了多线程策略。请根据计算机内核数合理设置 `--n_jobs` 参数，一般不超过内核数。如果想要确认最优线程数，可以参考 [这篇文章](https://blog.csdn.net/hxxjxw/article/details/119531239)。
2. **显存管理**: 训练过程中，如果显存不足，可以尝试减小 `batch_size` 或增加 `accumulation_steps`。
3. **早停机制**: 训练过程中使用了早停机制，如果验证集损失连续 15 个 epoch 没有下降，训练将提前终止。

---

## 结果保存

- **模型检查点**: 训练过程中每 30 个 epoch 保存一次检查点，保存在 `checkpoints/` 目录下。
- **最终模型**: 训练完成后，模型将保存在 `models/` 目录下。
- **损失和准确率数据**: 训练过程中的损失、准确率和 F1 分数数据将保存在 `data/` 目录下。
- **可视化结果**: 训练过程中的损失、准确率和 F1 分数曲线将保存在 `imgs/` 目录下。

---

## 参考

- [Optuna 官方文档](https://optuna.readthedocs.io/en/stable/)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [Transformers 官方文档](https://huggingface.co/docs/transformers/index)
- [PyTorch 梯度累积](https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/9)
- [PyTorch 混合精度训练](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [PyTorch 早停机制](https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch)
- [PyTorch 学习率调度器](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [PyTorch 数据加载器](https://pytorch.org/docs/stable/data.html)
- [PyTorch 损失函数](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [PyTorch 优化器](https://pytorch.org/docs/stable/optim.html)
- [PyTorch 模型保存与加载](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch 多线程调参](https://pytorch.org/docs/stable/notes/multiprocessing.html)
- [PyTorch 显存管理](https://pytorch.org/docs/stable/notes/cuda.html)