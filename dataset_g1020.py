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

        # 适配 G1020 数据集常见的列名
        self.image_col = 'image_id' if 'image_id' in self.df.columns else self.df.columns[0]
        # binary_label 为青光眼分类常用列名
        self.label_col = 'binary_label' if 'binary_label' in self.df.columns else self.df.columns[-1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx][self.image_col])
        if not img_name.lower().endswith('.jpg'):
            img_name += '.jpg'

        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.open(os.path.join(self.img_dir, str(self.df.iloc[idx][self.image_col]))).convert('RGB')
            
        label = int(self.df.iloc[idx][self.label_col])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_g1020_transforms(img_size=384, is_train=True):
    transform_list = [
        MedicalCLAHE(),  # 1. 首先进行对比度增强，让 FFT 能看到更清晰的边缘
        transforms.Resize((img_size, img_size)),
    ]
    
    if is_train:
        # 2. 训练集加入几何变换
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15), # 小角度旋转，模拟拍摄偏差
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)
