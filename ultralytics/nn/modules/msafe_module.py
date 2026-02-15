"""MSAFE多尺度注意力特征增强模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    """SE通道注意力"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)


class MultiScaleFeatureExtractor(nn.Module):
    """双分支膏胀卷积(d=1,2) + 1x1融合"""

    def __init__(self, channels, scales=[1, 2]):
        super().__init__()
        self.channels = channels
        self.scales = scales
        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=2, dilation=2)
        self.fusion = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return self.fusion(torch.cat([self.conv1(x), self.conv2(x)], dim=1))


class FeatureEnhancer(nn.Module):
    """深度可分离卷积 + 残差连接"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.reconstructor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.reconstructor(x)


class FeatureFusion(nn.Module):
    """深度卷积 + 残差，做局部精细化"""

    def __init__(self, channels, temperature=0.1):
        super().__init__()
        self.channels = channels
        self.contrast_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x):
        return x + self.contrast_conv(x)


class MSAFE(nn.Module):
    """MSAFE主块：通道注意力 -> 多尺度提取 -> 特征增强 -> 融合"""

    def __init__(self, channels, scales=[1, 2], reduction=16, temperature=0.1):
        super().__init__()
        self.channels = channels
        self.channel_attention = ChannelAttention(channels, reduction)
        self.multi_scale_extractor = MultiScaleFeatureExtractor(channels, scales)
        self.feature_enhancer = FeatureEnhancer(channels)
        self.feature_fusion = FeatureFusion(channels, temperature)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.multi_scale_extractor(x)
        x = self.feature_enhancer(x)
        return self.feature_fusion(x)


class MSAFEBlock(nn.Module):
    """MSAFE包装块，兼容YOLO框架，支持通道适配"""

    def __init__(self, c1, c2=None, scales=None, reduction=16, temperature=0.1):
        super().__init__()
        if c2 is None:
            c2 = c1
        if scales is None:
            scales = [1, 2, 4]
        elif isinstance(scales, int):
            scales = [1, 2, scales]

        self.channel_adapter = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)
        self.msafe = MSAFE(c2, scales, reduction, temperature)

    def forward(self, x):
        return self.msafe(self.channel_adapter(x))


__all__ = ['MSAFE', 'MSAFEBlock']