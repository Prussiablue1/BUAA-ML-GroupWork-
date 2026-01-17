# test_pipeline.py 的内容示例
import pandas as pd
from preprocess.dataset import FakeNewsDataset
from transformers import BertTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import config

# 1. 加载数据
df = pd.read_pickle(config.TRAIN_PKL)

# 2. 定义图片预处理
img_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 定义分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 4. 实例化 Dataset 和 DataLoader
dataset = FakeNewsDataset(df, config.IMAGE_DIR, transform=img_transform, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# 5. 取一个 batch 看看
batch = next(iter(dataloader))
print(f"文本张量形状: {batch['input_ids'].shape}")
print(f"图片张量形状: {batch['image'].shape}")
print(f"知识嵌入形状: {batch['k_embed'].shape}")
print(f"标签: {batch['label']}")









import torch

# 在 test_pipeline.py 末尾添加
from models.baseline import MultimodalFakeNewsModel

# 1. 初始化模型
model = MultimodalFakeNewsModel(k_embed_dim=100)
model.to(config.DEVICE)
model.eval() # 测试模式

# 2. 模拟一个 batch 的推理
with torch.no_grad():
    input_ids = batch['input_ids'].to(config.DEVICE)
    attention_mask = batch['attention_mask'].to(config.DEVICE)
    image = batch['image'].to(config.DEVICE)
    k_embed = batch['k_embed'].to(config.DEVICE)
    
    outputs = model(input_ids, attention_mask, image, k_embed)

print(f"模型输出形状: {outputs.shape}") # 应该是 [16, 2]
print("模型前向传播测试成功！")