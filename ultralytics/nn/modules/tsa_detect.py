"""TSADetect三尺度自适应检测头，针对晶圆缺陷检测设计"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head import Detect
from .conv import Conv, DWConv


class SmallTargetEnhancer(nn.Module):
    """小目标增强器：高频分支 + 通道注意力 + 边缘精细化"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 高频分支
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.GroupNorm(16, channels // 2),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels // 2, 3, padding=1, dilation=2),
            nn.GroupNorm(16, channels // 2),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.GroupNorm(32, channels),
        )

        # 通道注意力
        self.small_target_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

        # 边缘增强（深度可分离卷积）
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
        )

    def forward(self, x):
        high_freq = self.high_freq_conv(x)
        attention = self.small_target_attention(x)
        edge_enhanced = self.edge_enhance(x)

        # 尺寸对齐
        target_size = x.shape[2:]
        if high_freq.shape[2:] != target_size:
            high_freq = F.interpolate(high_freq, size=target_size, mode="bilinear", align_corners=False)
        if edge_enhanced.shape[2:] != target_size:
            edge_enhanced = F.interpolate(edge_enhanced, size=target_size, mode="bilinear", align_corners=False)

        enhanced = high_freq * attention + edge_enhanced
        return enhanced + x


class AdaptiveFeatureAlignment(nn.Module):
    """可学习3x3偏移采样，做特征对齐"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 偏移预测（18 = 9个采样点 x 2个坐标）
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.GroupNorm(8, channels // 4),
            nn.SiLU(),
            nn.Conv2d(channels // 4, 18, 3, padding=1),
        )

        # 9个采样点的重要性权重
        self.importance_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 8, 9, 1),
            nn.Sigmoid(),
        )

        # 零初始化，让初始行为近似恒等映射
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        offset = self.offset_conv(x).view(B, 9, 2, H, W)
        importance = self.importance_conv(x)

        # 基础网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing="ij",
        )

        # 3x3邻域偏移
        sample_offsets = torch.tensor(
            [
                [-1, -1], [-1, 0], [-1, 1],
                [0, -1], [0, 0], [0, 1],
                [1, -1], [1, 0], [1, 1],
            ],
            device=x.device,
            dtype=x.dtype,
        ).view(9, 2, 1, 1)

        base_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).unsqueeze(0)  # (1, 1, 2, H, W)
        sample_grid = base_grid + sample_offsets + offset  # (B, 9, 2, H, W)

        # 归一化到[-1, 1]
        sample_grid[:, :, 0] = 2.0 * sample_grid[:, :, 0] / (W - 1) - 1.0
        sample_grid[:, :, 1] = 2.0 * sample_grid[:, :, 1] / (H - 1) - 1.0
        sample_grid = sample_grid.to(dtype=x.dtype)

        # 采样并堆叠
        aligned_features = []
        for i in range(9):
            grid = sample_grid[:, i].permute(0, 2, 3, 1)  # (B, H, W, 2)
            sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
            aligned_features.append(sampled)
        aligned_features = torch.stack(aligned_features, dim=1)  # (B, 9, C, H, W)

        # 加权求和
        importance = importance.unsqueeze(2)  # (B, 9, 1, H, W)
        aligned_output = (aligned_features * importance).sum(dim=1)  # (B, C, H, W)
        return aligned_output


class MultiScaleFusion(nn.Module):
    """双向FPN融合，每层都带特征对齐"""

    def __init__(self, channels_list):
        super().__init__()
        self.num_levels = len(channels_list)

        # 每层的对齐器
        self.feature_aligners = nn.ModuleList([AdaptiveFeatureAlignment(ch) for ch in channels_list])

        # 自顶向下路径
        self.upsample_convs = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.upsample_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels_list[i + 1], channels_list[i], 1),
                    nn.GroupNorm(16, channels_list[i]),
                    nn.SiLU(),
                )
            )

        # 自底向上路径
        self.downsample_convs = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.downsample_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels_list[i], channels_list[i + 1], 3, stride=2, padding=1),
                    nn.GroupNorm(16, channels_list[i + 1]),
                    nn.SiLU(),
                )
            )

        # 可学习的融合权重
        self.fusion_weights = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(self.num_levels)])

    def forward(self, features):
        enhanced_features = list(features)

        # 对齐
        aligned_features = []
        for i, feature in enumerate(enhanced_features):
            aligned = self.feature_aligners[i](feature)
            aligned_features.append(aligned)
        enhanced_features = aligned_features

        # 自顶向下（上采样 + 加权融合）
        for i in range(self.num_levels - 2, -1, -1):
            upsampled = F.interpolate(
                enhanced_features[i + 1], size=enhanced_features[i].shape[2:], mode="bilinear", align_corners=False
            )
            upsampled = self.upsample_convs[i](upsampled)
            weights = F.softmax(self.fusion_weights[i], dim=0)
            enhanced_features[i] = weights[0] * enhanced_features[i] + weights[1] * upsampled

        # 自底向上残差注入
        for i in range(self.num_levels - 1):
            downsampled = self.downsample_convs[i](enhanced_features[i])
            if downsampled.shape[2:] != enhanced_features[i + 1].shape[2:]:
                downsampled = F.interpolate(
                    downsampled, size=enhanced_features[i + 1].shape[2:], mode="bilinear", align_corners=False
                )
            enhanced_features[i + 1] = enhanced_features[i + 1] + downsampled * 0.3

        return enhanced_features


class TSADetect(Detect):
    """TSA检测头：多尺度融合 -> 小目标增强 -> 检测卷积"""

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        self.small_target_enhancers = nn.ModuleList([SmallTargetEnhancer(x) for x in ch])
        self.multi_scale_fusion = MultiScaleFusion(ch)

    def forward(self, x):
        if self.end2end:
            return self.forward_end2end(x)

        fused_features = self.multi_scale_fusion(x)

        enhanced_features = []
        for i in range(self.nl):
            enhanced = self.small_target_enhancers[i](fused_features[i])
            enhanced_features.append(enhanced)

        for i in range(self.nl):
            enhanced_features[i] = torch.cat(
                (self.cv2[i](enhanced_features[i]), self.cv3[i](enhanced_features[i])), 1
            )

        if self.training:
            return enhanced_features

        y = self._inference(enhanced_features)
        return y if self.export else (y, enhanced_features)

    def forward_end2end(self, x):
        """end2end前向，one2one分支"""
        if not hasattr(self, "one2one_cv2") or not hasattr(self, "one2one_cv3"):
            import copy

            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

        fused_features = self.multi_scale_fusion(x)

        enhanced_features = []
        for i, feature in enumerate(fused_features):
            enhanced = self.small_target_enhancers[i](feature)
            enhanced_features.append(enhanced)

        x_detach = [xi.detach() for xi in enhanced_features]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1)
            for i in range(self.nl)
        ]

        for i in range(self.nl):
            enhanced_features[i] = torch.cat(
                (self.cv2[i](enhanced_features[i]), self.cv3[i](enhanced_features[i])), 1
            )

        if self.training:
            return {"one2many": enhanced_features, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": enhanced_features, "one2one": one2one})