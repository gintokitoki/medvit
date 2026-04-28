import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize():
    checkpoint_path = "checkpoints/improved_fft_best.pth"
    save_path = "fft_mask_heatmap.png"

    if not os.path.exists(checkpoint_path):
        print("未找到权重文件，请确认路径！")
        return

    # 1. 加载权重
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # 2. 提取 fft_block.mask (Shape: [256, 28, 15])
    # 注意：rfft2 的输出尺寸 w 是 w//2 + 1
    if 'fft_block.mask' in state_dict:
        mask = state_dict['fft_block.mask'].detach().numpy()
    else:
        print("权重中未找到 fft_block.mask，请检查模型定义")
        return

    # 3. 对 256 个通道取平均，得到全局频域响应
    avg_mask = np.mean(mask, axis=0)

    # 4. 绘图
    plt.figure(figsize=(8, 6))
    # 使用 hot 或 viridis 映射，显示频率分布
    plt.imshow(avg_mask, cmap='hot', aspect='auto')
    plt.colorbar(label='Mask Weight')
    plt.title("Learnable Spectral Mask Heatmap (Averaged over 256 channels)")
    plt.xlabel("Frequency (Width - rfft)")
    plt.ylabel("Frequency (Height)")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"热力图已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    visualize()