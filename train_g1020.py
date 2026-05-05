import os
import sys
import torch
import torch.nn as nn
import subprocess
from torch.utils.data import DataLoader, random_split

# 导入改进后的模型和 G1020 数据集
from models.medvit_arch import MedViT_small
from dataset_g1020 import G1020Dataset, get_g1020_transforms

def get_best_gpu():
    """通过 nvidia-smi 自动搜寻空闲显存最大的 GPU"""
    try:
        # 查询所有 GPU 的剩余显存
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        # 将结果转换为整数列表 [free_mem_0, free_mem_1, ...]
        free_memory = [int(x.strip()) for x in result.strip().split('\n')]
        # 找到显存最大的 GPU 索引
        best_gpu_index = free_memory.index(max(free_memory))
        print(f"检测到各显卡剩余显存: {free_memory} MiB")
        print(f"自动选择空闲显存最大的显卡: cuda:{best_gpu_index}")
        return torch.device(f"cuda:{best_gpu_index}")
    except Exception as e:
        print(f"无法自动检测 GPU (可能 nvidia-smi 不可用): {e}")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 1. 设备配置 (自动搜寻空闲卡)
    device = get_best_gpu()
    print(f"--- 启动 G1020 训练任务 ---")
    print(f"当前设备: {device}")

    # 2. 初始化改进后的模型
    # 设置为 384 以匹配 G1020 的训练需求
    img_size = 384
    model = MedViT_small(num_classes=2, img_size=img_size)
    
    # 3. 加载预训练权重
    weight_path = "weights/MedViT_small_im1k.pth"
    if os.path.exists(weight_path):
        print(f"加载预训练权重: {weight_path}")
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # 过滤掉形状不匹配的权重（主要是分类头和新增加的 FFT 掩码）
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
        
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"权重加载状态: {msg}")
    else:
        print("警告: 未找到预训练权重，将从随机初始化开始训练。")

    model.to(device)

    # 4. 准备数据集
    # 方案：创建两个独立的 Dataset 实例，分别绑定训练和验证的 transform
    train_dataset = G1020Dataset(
        csv_path="/home/wyh/data2/G1020/G1020.csv", 
        img_dir="/home/wyh/data2/G1020/Images",
        transform=get_g1020_transforms(img_size=img_size, is_train=True)
    )
    
    val_dataset = G1020Dataset(
        csv_path="/home/wyh/data2/G1020/G1020.csv", 
        img_dir="/home/wyh/data2/G1020/Images",
        transform=get_g1020_transforms(img_size=img_size, is_train=False)
    )
    
    dataset_size = len(train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # 使用索引进行划分，确保两个 Subset 指向不同的 Dataset 实例
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    # 5. 优化器与损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # 6. 训练循环
    num_epochs = 20
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f} - Val Accuracy: {accuracy:.2f}%")
        
        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "weights/medvit_g1020_best.pth")

if __name__ == "__main__":
    train()
