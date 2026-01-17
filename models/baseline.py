import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

class MultimodalFakeNewsModel(nn.Module):
    def __init__(self, k_embed_dim=100, num_classes=2):
        super(MultimodalFakeNewsModel, self).__init__()
        
        # 1. 文本分支：BERT (输出维度 768)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 2. 图像分支：ResNet50 (预训练)
        # resnet = models.resnet50(pretrained=True)
        resnet = models.resnet50(weights='DEFAULT')
        # 去掉最后的全连接层，只用它提取特征 (输出维度 2048)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 3. 知识嵌入分支：简单的映射层 (100 -> 128)
        self.k_fc = nn.Linear(k_embed_dim, 128)
        
        # 4. 融合后的分类器
        # 总维度 = BERT(768) + ResNet(2048) + Knowledge(128) = 2944
        combined_dim = 768 + 2048 + 128
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, image, k_embed):
        # 文本特征提取
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.pooler_output  # [batch_size, 768]
        
        # 图像特征提取
        img_features = self.vision_backbone(image)
        img_features = img_features.view(img_features.size(0), -1)  # [batch_size, 2048]
        
        # 知识特征提取
        k_features = torch.relu(self.k_fc(k_embed))  # [batch_size, 128]
        
        # 特征拼接 (Fusion)
        combined = torch.cat((text_features, img_features, k_features), dim=1)
        
        # 分类
        logits = self.classifier(combined)
        return logits