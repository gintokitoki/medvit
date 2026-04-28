import torch
import torch.nn as nn
import torch.fft


class LearnableSpectralMask(nn.Module):
    def __init__(self, channels, h, w):
        super().__init__()
        # 创建一个与频谱尺寸一致的可学习权重 (实数域)
        # FFT后尺寸为 (h, w//2 + 1)
        self.mask = nn.Parameter(torch.ones(channels, h, w // 2 + 1))

    def forward(self, x):
        # x: [B, C, H, W]
        # 1. 转换到频域 (使用实数FFT)
        x_freq = torch.fft.rfft2(x, norm='ortho')

        # 2. 应用可学习掩码
        # 将掩码限制在 0-1 之间或使用缩放，这里直接相乘
        x_freq = x_freq * self.mask.to(x_freq.dtype)

        # 3. 逆变换回到空间域
        x_spatial = torch.fft.irfft2(x_freq, s=(x.shape[-2], x.shape[-1]), norm='ortho')

        return x_spatial