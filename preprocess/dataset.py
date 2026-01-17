import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import numpy as np

class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, tokenizer=None, max_len=128):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. 处理文本 (合并正文和描述)
        text = str(row['text']) + " " + str(row['description'])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 2. 处理图片
        # 转换路径格式: 处理 Windows/Linux 路径差异
        img_rel_path = row['image_path'].lstrip('\\').replace('\\', '/')
        img_path = os.path.join("data", img_rel_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 如果图片损坏或找不到，生成一张全黑图
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        # 3. 处理知识嵌入 (Knowledge Embedding)
        # 假设它是 list 格式，转为 tensor
        k_embed = torch.tensor(row['knowledge_embedding'], dtype=torch.float)
        
        # 4. 标签
        label = torch.tensor(row['label'], dtype=torch.long) if 'label' in row else -1
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'image': image,
            'k_embed': k_embed,
            'label': label
        }