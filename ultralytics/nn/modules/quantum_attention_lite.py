"""CMConv跨通道混合卷积模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .conv import Conv


class MixingGate(nn.Module):
    """跨通道混合门控，用可学习的相位参数做通道间信息交换"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.theta = nn.Parameter(torch.randn(1) * 0.1)
        self.phi = nn.Parameter(torch.randn(1) * 0.1)
        self.channel_modulator = nn.Parameter(torch.randn(channels, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)
        mixed_state_0 = cos_theta * x
        mixed_state_1 = sin_theta * torch.roll(x, shifts=1, dims=1)
        alpha = torch.cos(self.phi)
        beta = torch.sin(self.phi)
        mixed_features = alpha * mixed_state_0 + beta * mixed_state_1
        modulation_weights = torch.sigmoid(self.channel_modulator).view(1, -1, 1, 1)
        mixed_features = mixed_features * modulation_weights
        return mixed_features


class EfficientChannelSelection(nn.Module):
    """SE通道注意力，带可学习的筛选强度"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.selection_strength = nn.Parameter(torch.tensor(0.5))
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // 16), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.channel_attention(x)
        selection_probability = torch.sigmoid(self.selection_strength)
        selected_features = x * attention_weights * selection_probability
        return selected_features


class EfficientMultiStateFusion(nn.Module):
    """多状态特征融合，权重可学习"""

    def __init__(self, channels: int, num_states: int = 2):
        super().__init__()
        self.channels = int(channels)
        self.num_states = int(num_states)
        self.modulation_params = nn.Parameter(torch.randn(num_states) * 0.1)
        self.state_weights = nn.Parameter(torch.ones(num_states) / num_states)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feature_states = []
        for i in range(self.num_states):
            modulation_factor = torch.cos(self.modulation_params[i])
            state_weight = torch.softmax(self.state_weights, dim=0)[i]
            if i == 0:
                feature_state = x * modulation_factor * state_weight
            else:
                feature_state = torch.roll(x, shifts=i, dims=2) * modulation_factor * state_weight
            feature_states.append(feature_state)
        fused_state = torch.stack(feature_states, dim=0).sum(dim=0)
        return fused_state


class CCAM(nn.Module):
    """跨通道注意力模块：混合门控 + 通道注意力 + 空间注意力"""

    def __init__(self, channels: int, reduction: int = 8, num_mixing_states: int = 2):
        super().__init__()
        self.channels = int(channels)
        self.reduction = max(int(reduction), 8)
        self.mixing_gate = MixingGate(channels)
        reduced_channels = max(4, channels // self.reduction)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.fusion_weights = nn.Parameter(torch.tensor([0.6, 0.4]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed_features = self.mixing_gate(x)
        channel_att = self.channel_attention(x)
        channel_enhanced = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        spatial_enhanced = x * spatial_att
        normalized_weights = torch.softmax(self.fusion_weights, dim=0)
        fused_features = (
            normalized_weights[0] * mixed_features +
            normalized_weights[1] * (channel_enhanced + spatial_enhanced) * 0.5
        )
        return x + 0.1 * fused_features


class CMConv(nn.Module):
    """C3k2风格CSP块，通道够大时在最后一层瓶颈加CCAM"""

    def __init__(self, c1, c2, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__()
        c1, c2, n = int(c1), int(c2), int(n)
        g = max(1, int(g))
        e = float(e)
        k = int(k)
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            CMBottleneck(self.c, self.c, shortcut, g, k=(k, k), e=1.0)
            for _ in range(n)
        )
        if self.c >= 64 and n >= 2:
            self.ccam = CCAM(self.c, reduction=max(8, self.c // 8))
            self.use_mixing = True
        else:
            self.use_mixing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for i, m in enumerate(self.m):
            y.append(m(y[-1]))
            if self.use_mixing and i == len(self.m) - 1:
                y[-1] = self.ccam(y[-1])
        return self.cv2(torch.cat(y, 1))


class CMBottleneck(nn.Module):
    """瓶颈层，通道>=128时加CCAM增强"""

    def __init__(self, c1, c2, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__()
        c1, c2 = int(c1), int(c2)
        g = max(1, int(g))
        e = float(e)
        shortcut = bool(shortcut)
        c_ = int(c2 * e)
        if isinstance(k, (list, tuple)):
            k1 = int(k[0])
            k2 = int(k[1]) if len(k) > 1 else k1
        else:
            k1 = k2 = int(k)
        self.cv1 = Conv(c1, c_, k1, 1)
        self.cv2 = Conv(c_, c2, k2, 1, g=g)
        self.add = shortcut and c1 == c2
        if c2 >= 128:
            self.mixing_enhance = CCAM(c2, reduction=max(8, c2 // 8))
            self.use_mixing = True
        else:
            self.use_mixing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        if self.use_mixing:
            y = self.mixing_enhance(y)
        return x + y if self.add else y


# 别名兼容
CMConv = CMConv
OptimizedQIAM = CCAM
OptimizedMixingBottleneck = CMBottleneck
QIAMLite = OptimizedQIAM
MixingBottleneckLite = OptimizedMixingBottleneck


class EnhancedQIAM(CCAM):
    """CCAM变体，压缩比更小"""

    def __init__(self, channels, reduction: int = 4, num_mixing_states: int = 2):
        super().__init__(channels, reduction, num_mixing_states)


class EnhancedCMConv(CMConv):
    """CMConv变体，降低CCAM触发门槛"""

    def __init__(self, c1, c2, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        if self.c >= 32:
            self.ccam = CCAM(self.c, reduction=max(4, self.c // 16))
            self.use_mixing = True


class EnhancedMixingBottleneck(CMBottleneck):
    """CMBottleneck变体，降低触发门槛"""

    def __init__(self, c1, c2, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        if int(c2) >= 64:
            self.mixing_enhance = CCAM(int(c2), reduction=max(4, int(c2) // 16))
            self.use_mixing = True


class MultiScaleContrastAttention(nn.Module):
    """多尺度对比注意力：缺陷感知分支 + 多尺度卷积 + 对比/空间注意力"""

    def __init__(self, c1, c2=None, ratio: int = 16, kernel_sizes=[3, 5, 7]):
        super().__init__()
        if c2 is None:
            c2 = c1
        self.c1 = c1
        self.c2 = c2
        self.ratio = ratio
        self.kernel_sizes = kernel_sizes
        self.defect_extractor = nn.Sequential(
            nn.Conv2d(c1, max(1, c1 // ratio), 1, bias=False),
            nn.BatchNorm2d(max(1, c1 // ratio)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, c1 // ratio), c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.multi_scale_branches = nn.ModuleList([
            nn.Conv2d(c1, c2, k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.multi_scale_fusion = nn.Conv2d(c2 * len(kernel_sizes), c2, 1, bias=False)
        self.contrast_enhancer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, max(1, c1 // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, c1 // 4), c2, 1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.threshold_adapter = nn.Parameter(torch.ones(1, c2, 1, 1) * 0.5)
        if c1 != c2:
            self.channel_adapter = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.channel_adapter = nn.Identity()
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        identity = self.channel_adapter(x)
        defect_features = self.defect_extractor(x)
        multi_scale_features = [branch(x) for branch in self.multi_scale_branches]
        multi_scale_concat = torch.cat(multi_scale_features, dim=1)
        multi_scale_out = self.multi_scale_fusion(multi_scale_concat)
        contrast_weight = self.contrast_enhancer(x)
        enhanced_features = identity * contrast_weight
        avg_out = torch.mean(enhanced_features, dim=1, keepdim=True)
        max_out, _ = torch.max(enhanced_features, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.spatial_attention(spatial_input)
        threshold_mask = torch.sigmoid((enhanced_features - self.threshold_adapter) * 10)
        adaptive_features = enhanced_features * threshold_mask
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_features = (
            weights[0] * defect_features +
            weights[1] * multi_scale_out +
            weights[2] * adaptive_features
        )
        output = fused_features * spatial_weight + identity * 0.1
        return output


MSCA = MultiScaleContrastAttention

__all__ = [
    'MixingGate', 'EfficientChannelSelection', 'EfficientMultiStateFusion',
    'OptimizedQIAM', 'CMConv', 'OptimizedMixingBottleneck',
    'MultiScaleContrastAttention', 'MSCA',
    'CMConv', 'QIAMLite', 'MixingBottleneckLite',
    'EnhancedQIAM', 'EnhancedCMConv', 'EnhancedMixingBottleneck',
    'C3k2QC', 'MixingC3k2Lite',
    'CCAM', 'CMBottleneck'
]
