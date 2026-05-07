import torch
import torch.nn as nn
import torch.fft


class LearnableSpectralMask(nn.Module):
    def __init__(self, channels, h, w):
        super().__init__()
        # 1. 创建与频域尺寸一致的可学习权重
        # FFT后尺寸为 (h, w // 2 + 1)
        self.mask = nn.Parameter(torch.ones(channels, h, w // 2 + 1))

        # 2. 引入可学习的缩放因子，初始化为 0.1
        # 初始值为 0.1，给频域分支一个初始的“存在感”
        self.gamma = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, x):
        # x 形状: [B, C, H, W]
        identity = x

        # 1. 转换到频域 (使用实数FFT)
        x_freq = torch.fft.rfft2(x, norm='ortho')

        # 2. 应用可学习掩码
        # 限制 mask 在合理范围内（如使用 sigmoid 确保权重在 0-1 之间，可选）
        mask = torch.sigmoid(self.mask)
        x_freq = x_freq * mask.to(x_freq.dtype)

        # 3. 逆变换回到空间域
        x_spatial = torch.fft.irfft2(x_freq, s=(x.shape[-2], x.shape[-1]), norm='ortho')

        # 4. 残差输出：原始特征 + gamma * 频域增强特征
        return identity + self.gamma * x_spatial