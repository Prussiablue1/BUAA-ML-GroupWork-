import torch

# 路径相关
DATA_DIR = "data"
TRAIN_PKL = "data/train.pkl"
TEST_PKL = "data/test_no_label.pkl"
IMAGE_DIR = "data/image"

# 训练超参数
BATCH_SIZE = 8       # 如果显存够大可以改回 16
LEARNING_RATE = 2e-5 # 学习率，BERT 建议设得很小
EPOCHS = 10          # 训练轮数
IMG_SIZE = 224       # ResNet 标准输入尺寸
MAX_LEN = 128        # 文本最大长度

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"