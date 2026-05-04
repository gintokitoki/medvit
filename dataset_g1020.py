import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
            # 备选方案：尝试直接使用 img_name 如果 csv 里已经带了后缀
            image = Image.open(os.path.join(self.img_dir, str(self.df.iloc[idx][self.image_col]))).convert('RGB')
            
        label = int(self.df.iloc[idx][self.label_col])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_g1020_transforms(img_size=384):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
