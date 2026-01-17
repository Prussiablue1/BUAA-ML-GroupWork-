import os

def create_project_structure():
    # 定义需要创建的目录
    dirs = [
        "data/image",         # 原始数据和图片
        "preprocess",         # 数据处理逻辑
        "models",             # 模型架构
        "utils",              # 工具函数
        "checkpoints",        # 存放训练好的模型权重
        "submission",         # 存放生成的csv结果
        "notebooks"           # 存放EDA分析过程
    ]
    
    # 定义初始化的空文件
    files = [
        "preprocess/__init__.py",
        "preprocess/dataset.py",
        "models/__init__.py",
        "models/baseline.py",
        "utils/__init__.py",
        "utils/config.py",
        "train.py",
        "inference.py"
    ]

    # 创建目录
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

    # 创建文件
    for f in files:
        with open(f, 'w', encoding='utf-8') as file:
            if f.endswith('config.py'):
                file.write("# 配置文件\nBATCH_SIZE = 16\nLEARNING_RATE = 2e-5\nEPOCHS = 10\n")
        print(f"Created file: {f}")

    print("\n✅ 项目结构初始化完成！请将 train.pkl 和 image 文件夹放入 data/ 目录下。")

if __name__ == "__main__":
    create_project_structure()