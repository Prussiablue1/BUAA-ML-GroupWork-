import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from preprocess.dataset import FakeNewsDataset
from models.baseline import MultimodalFakeNewsModel
from utils import config
import os

def inference(threshold=0.5): # <--- 增加阈值参数
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
    all_probs = [] # 存储概率而不是直接存 0/1
    print(f"正在进行推理 (当前阈值: {threshold})...")
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            k_embed = batch['k_embed'].to(device)

            outputs = model(input_ids, attention_mask, image, k_embed)
            
            # 使用 softmax 转化为概率
            probs = F.softmax(outputs, dim=1)
            # 取出类别为 1 (假新闻) 的概率
            fake_probs = probs[:, 1].cpu().numpy()
            all_probs.extend(fake_probs)

    # 6. 根据阈值生成标签
    # 如果概率 > threshold，则判定为 1，否则为 0
    final_preds = [1 if p > threshold else 0 for p in all_probs]

    # 7. 生成提交文件
    submission = pd.DataFrame({
        'ID': test_df.index,
        'label': final_preds
    })
    
    os.makedirs("submission", exist_ok=True)
    save_path = f"submission/submission_th{threshold}.csv"
    submission.to_csv(save_path, index=False)
    print(f"预测完成！结果已保存至: {save_path}")

if __name__ == "__main__":
    # 你可以手动尝试几个不同的阈值，生成多个文件分别提交测试
    # 建议尝试范围：0.4, 0.45, 0.5, 0.55, 0.6
    for t in [0.45, 0.5, 0.55]:
        inference(threshold=t)