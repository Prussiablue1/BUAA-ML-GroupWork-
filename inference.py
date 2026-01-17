import torch
import torch.nn.functional as F  # 导入 F 以使用 softmax
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from preprocess.dataset import FakeNewsDataset
from models.baseline import MultimodalFakeNewsModel
from utils import config
import os

def inference(threshold=0.45): # <--- 固定设定为 0.45
    device = torch.device(config.DEVICE)
    
    # 1. 加载测试数据
    test_df = pd.read_pickle(config.TEST_PKL)
    print(f"检测到测试集样本数: {len(test_df)}")

    # 2. 初始化工具
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    img_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. 准备 Dataset
    test_dataset = FakeNewsDataset(
        test_df, 
        config.IMAGE_DIR, 
        transform=img_transform, 
        tokenizer=tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 4. 加载模型权重
    model = MultimodalFakeNewsModel().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()

    # 5. 开始预测
    all_fake_probs = []
    print(f"正在进行推理 (判定阈值: {threshold})...")
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            k_embed = batch['k_embed'].to(device)

            outputs = model(input_ids, attention_mask, image, k_embed)
            
            # 将模型输出转为概率
            probs = F.softmax(outputs, dim=1)
            # 获取“假新闻”(Label 1) 的预测概率
            fake_probs = probs[:, 1].cpu().numpy()
            all_fake_probs.extend(fake_probs)

    # 6. 根据 0.45 阈值生成预测标签
    # 概率 > 0.45 判定为 1 (Fake)，否则判定为 0 (Real)
    final_labels = [1 if p >= threshold else 0 for p in all_fake_probs]

    # 7. 生成提交文件
    # 确保列名 ID 和 label 大小写符合 CSV 样例要求
    submission = pd.DataFrame({
        'ID': test_df.index,
        'label': final_labels
    })
    
    # 保存结果
    os.makedirs("submission", exist_ok=True)
    save_path = "submission/submission.csv"
    submission.to_csv(save_path, index=False)
    print(f"预测完成！已将阈值设定为 {threshold}")
    print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    inference()