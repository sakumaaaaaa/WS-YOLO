"""
ASDA (Adaptive Spatial Dynamic Aggregation) Module
自适应空间动态聚合模块

基于以下理论：
1. 空间自适应注意力理论 - 基于空间位置的自适应权重分配
2. 动态特征聚合理论 - 多层级特征动态融合
3. 位置敏感检测理论 - 基于目标位置的特征增强

专为YOLO检测头设计的轻量级创新模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAdaptiveAttention(nn.Module):
    """空间自适应注意力模块"""
    
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        
        # 简化的通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 8), channels, 1),
            nn.Sigmoid()
        )
        
        # 简化的空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x


class DynamicReceptiveField(nn.Module):
    """动态感受野调整模块"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 确保通道数能被3整除
        self.branch_channels = channels // 3
        if channels % 3 != 0:
            self.branch_channels = channels // 4  # 使用4分支以避免除法问题
        
        # 多尺度卷积分支 - 使用深度可分离卷积
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, self.branch_channels, 1)
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels),
            nn.Conv2d(channels, self.branch_channels, 1)
        )
        self.conv_7x7 = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels),
            nn.Conv2d(channels, self.branch_channels, 1)
        )
        
        # 动态权重选择
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 3, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征融合
        fusion_channels = self.branch_channels * 3
        self.fusion = nn.Conv2d(fusion_channels, channels, 1)
        
    def forward(self, x):
        # 多尺度特征提取
        feat_3x3 = self.conv_3x3(x)
        feat_5x5 = self.conv_5x5(x)
        feat_7x7 = self.conv_7x7(x)
        
        # 拼接多尺度特征
        multi_scale_feat = torch.cat([feat_3x3, feat_5x5, feat_7x7], dim=1)
        
        # 动态权重选择
        weights = self.selector(x)  # [B, 3, 1, 1]
        
        # 最终融合
        output = self.fusion(multi_scale_feat) + x
        
        return output


class FeatureAggregator(nn.Module):
    """轻量级特征聚合器"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 深度可分离卷积
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        
        # 批归一化
        self.bn = nn.BatchNorm2d(channels)
        
        # 激活函数
        self.activation = nn.SiLU(inplace=True)
        
    def forward(self, x):
        # 深度可分离卷积
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.activation(out)
        
        # 残差连接
        return out + x


class BoundaryEnhancer(nn.Module):
    """边界敏感增强器"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 边界检测卷积
        self.boundary_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        
        # 中心区域增强
        self.center_conv = nn.Conv2d(channels, channels, 1)
        
        # 融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # 边界特征提取
        boundary_feat = self.boundary_conv(x)
        
        # 中心特征增强
        center_feat = self.center_conv(x)
        
        # 自适应融合
        alpha = torch.sigmoid(self.fusion_weight)
        enhanced = alpha * boundary_feat + (1 - alpha) * center_feat
        
        return enhanced + x


class ASDA(nn.Module):
    """ASDA主模块 - 自适应空间动态聚合"""
    
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        
        # 核心组件
        self.spatial_attention = SpatialAdaptiveAttention(channels, reduction)
        self.dynamic_rf = DynamicReceptiveField(channels)
        self.feature_aggregator = FeatureAggregator(channels)
        self.boundary_enhancer = BoundaryEnhancer(channels)
        
    def forward(self, x):
        # 1. 空间自适应注意力
        x = self.spatial_attention(x)
        
        # 2. 动态感受野调整
        x = self.dynamic_rf(x)
        
        # 3. 特征聚合
        x = self.feature_aggregator(x)
        
        # 4. 边界增强
        x = self.boundary_enhancer(x)
        
        return x


class ASDABlock(nn.Module):
    """ASDA块 - 用于YOLO架构集成"""
    
    def __init__(self, c1, c2=None, reduction=8):
        super().__init__()
        c2 = c2 or c1
        
        # 通道适配
        self.channel_adapter = None
        if c1 != c2:
            self.channel_adapter = nn.Conv2d(c1, c2, 1)
        
        # ASDA核心模块
        self.asda = ASDA(c2, reduction)
        
    def forward(self, x):
        # 通道适配
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        
        # ASDA处理
        return self.asda(x)


# 导出模块
__all__ = ['ASDA', 'ASDABlock', 'SpatialAdaptiveAttention', 'DynamicReceptiveField', 
           'FeatureAggregator', 'BoundaryEnhancer']