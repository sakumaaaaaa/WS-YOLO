"""
MSAFE (Multi-Scale Attention Feature Enhancement) Module

多尺度注意力特征增强模块，专为半导体晶圆缺陷检测设计。

轻量化设计：
1. 通道注意力：SE注意力机制实现通道自适应加权
2. 多尺度特征提取：深度可分离卷积替代标准卷积，大幅降低计算复杂度
   - 深度卷积(DWConv)实现空间多尺度特征提取（dilation=1,2）
   - 点卷积(PWConv)实现跨通道特征融合
3. 特征增强：深度可分离卷积 + 残差连接
4. 全局残差连接：确保梯度稳定传播
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """空间注意力模块 - 定位小目标关键区域"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        gate = torch.sigmoid(self.bn(self.conv(torch.cat([avg_out, max_out], dim=1))))
        return x * gate


class MultiScaleFeatureExtractor(nn.Module):
    """轻量级多尺度特征提取器 - 深度可分离卷积实现
    
    采用深度卷积进行空间多尺度特征提取，点卷积进行跨通道特征融合，
    相比标准卷积大幅降低计算复杂度，同时保持多尺度感受野覆盖。
    """
    
    def __init__(self, channels, scales=[1, 2]):
        super().__init__()
        self.channels = channels
        self.scales = scales
        
        # 深度卷积：空间多尺度特征提取（FLOPs仅为标准卷积的1/C）
        self.dw_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                   dilation=1, groups=channels, bias=False)
        self.dw_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2,
                                   dilation=2, groups=channels, bias=False)
        # 点卷积：跨通道特征融合
        self.pw_conv = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        x1 = self.dw_conv1(x)
        x2 = self.dw_conv2(x)
        return self.act(self.bn(self.pw_conv(torch.cat([x1, x2], dim=1))))


class FeatureEnhancer(nn.Module):
    """轻量级特征增强器 - 深度可分离卷积 + 残差连接"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.reconstructor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.reconstructor(x)


class MSAFE(nn.Module):
    """MSAFE核心：多尺度提取 -> 特征增强 -> 空间注意力门控"""
    
    def __init__(self, channels, scales=[1, 2], reduction=16, temperature=0.1):
        super().__init__()
        self.channels = channels
        self.multi_scale_extractor = MultiScaleFeatureExtractor(channels, scales)
        self.feature_enhancer = FeatureEnhancer(channels)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
    def forward(self, x):
        x = self.multi_scale_extractor(x)
        x = self.feature_enhancer(x)
        x = self.spatial_attention(x)
        return x


class MSAFEBlock(nn.Module):
    """MSAFE倒残差块：通道扩展 -> MSAFE处理 -> 通道压缩 + 全局残差
    
    借鉴MobileNetV2倒残差设计，在紧凑的输入/输出通道间
    扩展到更高维度进行多尺度注意力特征增强，兼顾特征容量与计算效率。
    """
    
    def __init__(self, c1, c2=None, scales=None, reduction=16, temperature=0.1, expand_ratio=4):
        super().__init__()
        if c2 is None:
            c2 = c1
        if scales is None:
            scales = [1, 2]
        elif isinstance(scales, int):
            scales = [1, scales]
            
        c_mid = c2  # 不做通道膨胀，保持轻量
        
        # 通道适配（输入通道 -> 输出通道）
        self.channel_adapter = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)
        # 倒残差：扩展 -> 处理 -> 压缩
        self.expand = nn.Sequential(
            nn.Conv2d(c2, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True)
        ) if c_mid != c2 else nn.Identity()
        self.msafe = MSAFE(c_mid, scales, reduction, temperature)
        self.squeeze = nn.Sequential(
            nn.Conv2d(c_mid, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        ) if c_mid != c2 else nn.Identity()
        
    def forward(self, x):
        x = self.channel_adapter(x)
        identity = x
        x = self.expand(x)
        x = self.msafe(x)
        x = self.squeeze(x)
        return x + identity  # 全局残差连接


# 导出模块
__all__ = ['MSAFE', 'MSAFEBlock']
