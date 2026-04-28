# main_baseline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Refuge1Dataset
from models.medvit_arch import MedViT_small
import os


def train_baseline():
    # 1. 基础配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = "/home/wyh/data2/REFUGE1"
    weight_path = "weights/MedViT_small_im1k.pth"
    batch_size = 16
    epochs = 20
    lr = 1e-4

    # 2. 数据准备
    train_ds = Refuge1Dataset(data_root, mode='train')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    val_ds = Refuge1Dataset(data_root, mode='val')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3. 模型初始化与权重加载
    print("正在初始化 MedViT 基准模型...")
    model = MedViT_small(num_classes=1000)
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)

    # 精准替换：MedViT 的分类头是 proj_head[0]
    in_features = model.proj_head[0].in_features
    model.proj_head[0] = nn.Linear(in_features, 2)

    model = model.to(device)

    # 4. 优化器与损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # 5. 训练主循环
    print(f"开始实验 1 (Baseline) 训练，总样本: {len(train_ds)}")
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 验证环节
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(
            f"Epoch [{epoch + 1}/{epochs}] Loss: {train_loss / len(train_loader):.4f} | Train Acc: {100. * correct / total:.2f}% | Val Acc: {val_acc:.2f}%")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/baseline_best.pth")

    print(f"实验 1 完成！最优验证集准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    train_baseline()