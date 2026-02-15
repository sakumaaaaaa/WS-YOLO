"""
MSAFE (Multi-Scale Attention Feature Enhancement) Module

基于多尺度注意力机制和特征增强理论的创新模块，专为半导体晶圆缺陷检测设计。

理论基础：
1. 多尺度特征提取：利用不同膨胀率卷积捕获不同尺度的缺陷特征
2. 注意力机制：通过SE注意力增强重要特征通道
3. 特征增强：通过深度可分离卷积和残差连接增强特征表达
4. 轻量化设计：平衡检测精度与计算效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    """轻量级通道注意力模块（SE注意力机制）"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        
        # 简化的注意力权重生成
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 直接生成注意力权重，不进行额外的频域变换
        attention_weights = self.attention(x)
        
        # 应用注意力
        enhanced_features = x * attention_weights
        
        return enhanced_features


class MultiScaleFeatureExtractor(nn.Module):
    """轻量级多尺度特征提取器"""
    
    def __init__(self, channels, scales=[1, 2]):
        super().__init__()
        self.channels = channels
        self.scales = scales
        
        # 只使用两个尺度，减少参数量
        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=2, dilation=2)
        
        # 简化的融合
        self.fusion = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        # 两个尺度的特征提取
        scale1 = self.conv1(x)
        scale2 = self.conv2(x)
        
        # 拼接特征
        multi_scale_features = torch.cat([scale1, scale2], dim=1)
        
        # 融合特征
        fused_features = self.fusion(multi_scale_features)
        
        return fused_features


class FeatureEnhancer(nn.Module):
    """轻量级特征增强器"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 简化的重构网络
        self.reconstructor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # 深度可分离卷积
            nn.Conv2d(channels, channels, 1),  # 点卷积
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        # 简化的特征重构
        reconstructed = self.reconstructor(x)
        
        # 残差连接
        output = x + reconstructed
        
        return output


class FeatureFusion(nn.Module):
    """轻量级特征融合器"""
    
    def __init__(self, channels, temperature=0.1):
        super().__init__()
        self.channels = channels
        
        # 简化的对比增强
        self.contrast_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        
    def forward(self, x):
        # 简化的对比增强
        enhanced = self.contrast_conv(x)
        
        # 残差连接
        output = x + enhanced
        
        return output


class MSAFE(nn.Module):
    """轻量级MSAFE主模块"""
    
    def __init__(self, channels, scales=[1, 2], reduction=16, temperature=0.1):
        super().__init__()
        self.channels = channels
        
        # 核心组件
        self.channel_attention = ChannelAttention(channels, reduction)
        self.multi_scale_extractor = MultiScaleFeatureExtractor(channels, scales)
        self.feature_enhancer = FeatureEnhancer(channels)
        self.feature_fusion = FeatureFusion(channels, temperature)
        
    def forward(self, x):
        # 1. 通道注意力增强
        attention_enhanced = self.channel_attention(x)
        
        # 2. 多尺度特征提取
        multi_scale_features = self.multi_scale_extractor(attention_enhanced)
        
        # 3. 特征增强
        enhanced = self.feature_enhancer(multi_scale_features)
        
        # 4. 特征融合
        fused = self.feature_fusion(enhanced)
        
        # 直接返回增强后的特征，不进行额外融合
        return fused


class MSAFEBlock(nn.Module):
    """MSAFE块，包含残差连接"""
    
    def __init__(self, c1, c2=None, scales=None, reduction=16, temperature=0.1):
        super().__init__()
        # 兼容YOLO框架的参数处理
        if c2 is None:
            c2 = c1
        if scales is None:
            scales = [1, 2, 4]
        elif isinstance(scales, int):
            scales = [1, 2, scales]
            
        # 通道适配
        self.channel_adapter = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)
        
        self.msafe = MSAFE(c2, scales, reduction, temperature)
        
    def forward(self, x):
        x = self.channel_adapter(x)
        return self.msafe(x)


# 导出模块
__all__ = ['MSAFE', 'MSAFEBlock']