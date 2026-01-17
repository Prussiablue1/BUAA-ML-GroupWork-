import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW  # 使用 PyTorch 自带的 AdamW
from torchvision import transforms
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from preprocess.dataset import FakeNewsDataset
from models.baseline import MultimodalFakeNewsModel
from utils import config

# 1. 初始化设置
device = torch.device(config.DEVICE)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 图片预处理
img_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train():
    # 2. 加载数据并切分验证集
    full_df = pd.read_pickle(config.TRAIN_PKL)
    # 简单切分 20% 作为验证集
    train_df = full_df.sample(frac=0.8, random_state=42)
    val_df = full_df.drop(train_df.index)

    train_dataset = FakeNewsDataset(train_df, config.IMAGE_DIR, transform=img_transform, tokenizer=tokenizer)
    val_dataset = FakeNewsDataset(val_df, config.IMAGE_DIR, transform=img_transform, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # 3. 初始化模型、损失函数、优化器
    model = MultimodalFakeNewsModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    best_f1 = 0.0

    # 4. 训练循环
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 将数据送入设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            k_embed = batch['k_embed'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, image, k_embed)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 5. 验证环节
        val_f1 = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{config.EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        # 保存 F1 最好的模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("更好的模型，已保存！")

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            k_embed = batch['k_embed'].to(device)
            labels = batch['label'].cpu().numpy()

            outputs = model(input_ids, attention_mask, image, k_embed)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    return f1_score(all_labels, all_preds)

if __name__ == "__main__":
    train()