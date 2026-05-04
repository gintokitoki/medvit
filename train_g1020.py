import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# 导入改进后的模型和 G1020 数据集
from models.medvit_arch import MedViT_small
from dataset_g1020 import G1020Dataset, get_g1020_transforms

def train():
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 启动 G1020 训练任务 ---")
    print(f"当前设备: {device}")

    # 2. 初始化改进后的模型
    # 注意：根据 models/medvit_arch.py，分类头是 proj_head
    model = MedViT_small(num_classes=2)
    
    # 3. 加载预训练权重
    weight_path = "weights/MedViT_small_im1k.pth"
    if os.path.exists(weight_path):
        print(f"加载预训练权重: {weight_path}")
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # 过滤掉形状不匹配的权重（主要是分类头）
        model_dict = model.state_dict()
        # 改进：由于新模型增加了 fft_block.mask，需要允许这部分不加载
        state_dict = {k: v for k, v in state_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
        
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"权重加载状态: {msg}")
    else:
        print("警告: 未找到预训练权重，将从随机初始化开始训练。")

    model.to(device)

    # 4. 准备数据集 (注意：目前模型 FFT 模块固定为 224x224 输入对应的 28x28 特征图)
    # 如果要使用 384x384，请参考后续说明修改 models/medvit_arch.py
    img_size = 224 
    full_dataset = G1020Dataset(
        csv_path="/home/wyh/data2/G1020/G1020.csv", 
        img_dir="/home/wyh/data2/G1020/Images",
        transform=get_g1020_transforms(img_size=img_size)
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

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
