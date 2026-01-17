# 多模态虚假新闻检测项目 (Multimodal Fake News Detection)

## 1. 项目概述
本项目为机器学习课程大作业，旨在解决社交媒体环境下的虚假新闻检测问题。由于虚假新闻往往在文本、图像和关联知识上具有协同欺骗性，我们构建了一个**多模态深度学习模型**，整合了新闻文本、视觉信息以及知识图谱嵌入，通过端到端的训练实现二分类任务（真/假新闻）。

**核心指标：** F1-Score (目前本地验证集最高分：0.75+)

## 2. 项目结构说明
项目采用模块化设计，便于团队协作与代码维护：

```text
FakeNewsDetection/
├── data/                  # 数据存储（需自行放置 train.pkl, image/ 等）
├── preprocess/            # 数据处理模块
│   └── dataset.py         # 核心 Dataset 类，负责多模态数据对齐与加载
├── models/                # 模型定义
│   └── baseline.py        # BERT + ResNet50 + MLP 晚期融合模型
├── utils/                 # 通用工具
│   └── config.py          # 超参数管理（学习率、BatchSize、阈值等）
├── checkpoints/           # 训练过程中保存的最佳模型权重 (.pth)
├── submission/            # 生成的预测结果 CSV 文件
├── train.py               # 训练与验证主脚本
├── inference.py           # 预测与生成提交文件脚本
└── README.md              # 项目文档
```



## 3. 技术架构设计
模型采用 晚期融合 (Late Fusion) 架构，通过不同分支提取各模态的高维特征：

文本分支 (Text Branch): 采用预训练的 bert-base-uncased。输入包括新闻正文、实体描述，并可扩展拼接评论信息。

图像分支 (Image Branch): 采用预训练的 ResNet50。利用 ImageNet 预训练权重提取视觉语义，经过 Pooling 层得到 2048 维向量。

知识分支 (Knowledge Branch): 处理数据集自带的 knowledge_embedding，通过全连接层（MLP）映射至共享特征空间。

融合层 (Fusion Layer): 将上述特征拼接（Concatenate）后，送入 512 维的隐藏层，最后通过 Softmax 输出分类概率。



## 4. 实验流程与复现
环境准备
```Bash
pip install torch torchvision transformers pandas scikit-learn pillow
```

第一步：初始化与数据准备
确保将 train.pkl、test_no_lable.pkl 和 image/ 文件夹放入 data/ 目录。

第二步：模型训练
运行以下命令开始训练。脚本会自动监控验证集的 F1 指标，并在 checkpoints/ 下保存最佳权重。
```Bash
python train.py
```

第三步：生成提交文件
运行推理脚本。脚本默认采用经过优化的阈值（如 0.45）来生成最终预测。
```Bash
python inference.py
```

## 5. 关键优化策略 
在项目迭代过程中，我们实施了以下优化手段：

阈值优化 (Threshold Tuning): 将分类阈值从默认的 0.5 调整为 0.45，显著提升了模型对假新闻的召回率，F1 分值提升约 0.6%。

过拟合控制: 观察到模型在 Epoch 2 之后出现过拟合，通过引入 Dropout (0.3) 和 早停策略 保护了模型的泛化能力。

多模态特征融合: 相比单文本 Baseline，加入图像特征和知识嵌入后，模型在处理具有迷惑性图片的新闻时表现更加稳健。
