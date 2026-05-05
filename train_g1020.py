import os
import sys
import torch
import torch.nn as nn
import subprocess
from sklearn.metrics import recall_score, f1_score
from torch.utils.data import DataLoader, random_split

# 导入改进后的模型和 G1020 数据集
from models.medvit_arch import MedViT_small
from dataset_g1020 import G1020Dataset, get_g1020_transforms

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

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

    # --- 第一步：冻结主干网络 ---
    # 先让随机初始化的分类头和 FFT 模块适应数据
    print("正在冻结主干网络，仅训练 FFT 模块和分类头...")
    for name, param in model.named_parameters():
        if "fft_block" not in name and "proj_head" not in name:
            param.requires_grad = False
    
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
    # 策略 C：使用 Focal Loss 替代 CrossEntropy
    criterion = FocalLoss(alpha=0.4).to(device) # alpha 偏向少数类
    
    # 第三步：调整学习率策略
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 6. 训练循环
    num_epochs = 40 # 增加 Epoch 以确保有足够时间进行全量精细微调
    best_acc = 0.0
    
    # 明确定义 optimizer 和 scheduler，确保在循环内更新时作用域正确
    for epoch in range(num_epochs):
        # 策略 A：延长冻结期至 10 轮
        if epoch == 10:
            print("--- 启动精细微调：已解冻主干网络 ---")
            for param in model.parameters():
                param.requires_grad = True
            
            # 策略 C：分层学习率 (主干极小 1e-6，新模块 1e-4)
            optimizer = torch.optim.AdamW([
                {'params': model.fft_block.parameters(), 'lr': 1e-4},
                {'params': model.proj_head.parameters(), 'lr': 1e-4},
                {'params': [p for n, p in model.named_parameters() if "fft" not in n and "proj" not in n], 'lr': 1e-6}
            ])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

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
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f} - Val Accuracy: {accuracy:.2f}% - Sensitivity (Recall): {recall:.4f} - F1-Score: {f1:.4f}")
        
        # 更新学习率调度器
        scheduler.step(accuracy)

        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "weights/medvit_g1020_best.pth")

if __name__ == "__main__":
    train()
