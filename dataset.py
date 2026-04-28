# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class Refuge1Dataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=224):
        """
        root_dir: /home/wyh/data2/REFUGE1
        mode: 'train', 'val', or 'test'
        """
        self.img_dir = os.path.join(root_dir, mode, "images")
        # 确保读取顺序一致
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"路径不存在: {self.img_dir}")

        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # 严格执行前缀标签规则
        label = 1 if img_name.startswith('g') else 0

        return image, torch.tensor(label, dtype=torch.long)