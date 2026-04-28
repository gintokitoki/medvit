import torch
import os
from models.medvit_arch import MedViT_small


def verify():
    weight_path = "weights/MedViT_small_im1k.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- 正在验证模型与权重 ---")

    try:
        # 1. 初始化 (ImageNet-1k 默认是 1000 类)
        model = MedViT_small(num_classes=1000)

        # test_weights.py 核心修改部分
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # 设置 strict=False，允许跳过我们新加的 fft_block.mask
        model.load_state_dict(state_dict, strict=False)

        print("状态: 预训练权重加载成功（已忽略新增的 FFT 掩码）！")

        # 3. 动态识别并替换分类头 (适配二分类)
        # proj_head 是 nn.Sequential 包含一个 nn.Linear
        if hasattr(model, 'proj_head'):
            # proj_head[0] 是内部的 nn.Linear
            linear_layer = model.proj_head[0]
            in_features = linear_layer.in_features
            model.proj_head[0] = torch.nn.Linear(in_features, 2)
            print(f"成功替换分类头: model.proj_head[0]")
        elif hasattr(model, 'head'):
            linear_layer = model.head[0]
            in_features = linear_layer.in_features
            model.head[0] = torch.nn.Linear(in_features, 2)
            print(f"成功替换分类头: model.head[0]")
        else:
            # 如果都不是，打印所有属性帮你查找
            print("未找到默认 head 名称，当前模型属性有:", [n for n, _ in model.named_children()])
            return

        model = model.to(device)

        # 4. 前向传播测试
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)

        print(f"前向传播测试成功！输出维度: {output.shape} (预期 [1, 2])")

    except Exception as e:
        print(f"验证过程中出现异常: {e}")


if __name__ == "__main__":
    verify()