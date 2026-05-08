import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MedicalCLAHE:
    """针对医学影像的自适应直方图均衡化"""
    def __init__(self, clip_limit=1.5, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)
        # 转换为 LAB 颜色空间处理亮度通道
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_enhanced)

class G1020Dataset(Dataset):
    def __init__(self, csv_path="/home/wyh/data2/G1020/G1020.csv", img_dir="/home/wyh/data2/G1020/Images", transform=None):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform

        # --- 根据截图精确指定列名 ---
        self.image_col = 'imageID'      # 对应第一列 (image_0.jpg)
        self.label_col = 'binaryLabels' # 对应第二列 (0 或 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 直接获取带后缀的文件名
        img_name = str(self.df.iloc[idx][self.image_col])
        img_path = os.path.join(self.img_dir, img_name)
        
        # 2. 标签读取
        label = int(self.df.iloc[idx][self.label_col])

        # 3. 增强读取的健壮性：确保只读图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"读取图片失败: {img_path}, 错误: {e}")
            # 如果读取失败，返回一个全黑图占位，避免训练崩溃（也可以选择抛出异常）
            image = Image.new('RGB', (384, 384), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

def get_g1020_transforms(img_size=384, is_train=True):
    if is_train:
        return transforms.Compose([
            MedicalCLAHE(clip_limit=1.5),  # 1. 首先进行对比度增强
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),   # 调回 15 度以降低噪点
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # 增加光影扰动
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            MedicalCLAHE(clip_limit=1.5),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
