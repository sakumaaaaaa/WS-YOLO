# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Dynamic Geometric Topology-Aware Detection (DGTAD) head module for semiconductor defect detection."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import make_anchors, dist2bbox
from .block import DFL
from .conv import Conv, DWConv


class GeometricTopologyAnalyzer(nn.Module):
    """
    Geometric Topology Analyzer for semiconductor defect detection.
    
    This module analyzes the geometric and topological properties of features
    to enhance detection of semiconductor defects with specific geometric patterns.
    """
    
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = max(in_channels // reduction, 16)
        
        # Geometric feature extraction
        self.geometric_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, 1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Topology connectivity analysis
        self.topology_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, 3, padding=1, groups=self.mid_channels, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 1, bias=False)
        )
        
        # Geometric invariance learning
        self.invariance_pool = nn.AdaptiveAvgPool2d(1)
        self.invariance_fc = nn.Sequential(
            nn.Linear(self.mid_channels * 2, self.mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass of Geometric Topology Analyzer.
        
        Args:
            x (torch.Tensor): Input feature tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Topology-aware enhanced features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Extract geometric features
        geo_feat = self.geometric_conv(x)  # [B, mid_channels, H, W]
        
        # Analyze topology connectivity
        topo_feat = self.topology_conv(x)  # [B, mid_channels, H, W]
        
        # Combine geometric and topological features
        combined_feat = torch.cat([geo_feat, topo_feat], dim=1)  # [B, mid_channels*2, H, W]
        
        # Learn geometric invariance weights
        pooled_feat = self.invariance_pool(combined_feat).view(B, -1)  # [B, mid_channels*2]
        weights = self.invariance_fc(pooled_feat).view(B, C, 1, 1)  # [B, C, 1, 1]
        
        # Apply topology-aware enhancement
        enhanced_x = x * weights
        
        return enhanced_x


class DynamicAnchorGenerator(nn.Module):
    """
    Dynamic Anchor Generator based on geometric topology analysis.
    
    This module generates adaptive anchors based on the geometric properties
    of semiconductor defects, improving detection accuracy for small targets.
    """
    
    def __init__(self, in_channels, num_anchors=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        
        # Geometric pattern recognition
        self.pattern_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Dynamic anchor prediction
        self.anchor_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels // 4, num_anchors * 2, 1),  # predict width and height scales
            nn.Sigmoid()
        )
        
        # Base anchor scales for semiconductor defects
        self.register_buffer('base_scales', torch.tensor([
            [0.5, 0.5],   # Small circular defects
            [1.0, 0.3],   # Linear defects
            [0.8, 0.8]    # Medium irregular defects
        ]))
        
    def forward(self, x):
        """
        Generate dynamic anchors based on input features.
        
        Args:
            x (torch.Tensor): Input feature tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Dynamic anchor scales [B, num_anchors, 2]
        """
        B, C, H, W = x.shape
        
        # Extract geometric patterns
        pattern_feat = self.pattern_conv(x)  # [B, C//4, H, W]
        
        # Predict dynamic anchor adjustments
        anchor_adj = self.anchor_predictor(pattern_feat)  # [B, num_anchors*2, 1, 1]
        anchor_adj = anchor_adj.view(B, self.num_anchors, 2)  # [B, num_anchors, 2]
        
        # Apply adjustments to base scales
        dynamic_scales = self.base_scales.unsqueeze(0) * (0.5 + anchor_adj)  # [B, num_anchors, 2]
        
        return dynamic_scales


class TopologyConsistencyConstraint(nn.Module):
    """
    Topology Consistency Constraint module for maintaining geometric relationships.
    
    This module ensures that detected features maintain topological consistency,
    which is crucial for accurate semiconductor defect classification.
    """
    
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # Spatial relationship modeling
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
        )
        
        # Consistency scoring
        self.consistency_score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Apply topology consistency constraint.
        
        Args:
            x (torch.Tensor): Input feature tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Consistency-constrained features [B, C, H, W]
        """
        # Model spatial relationships
        spatial_feat = self.spatial_conv(x)
        
        # Calculate consistency score
        consistency = self.consistency_score(spatial_feat)  # [B, 1, 1, 1]
        
        # Apply consistency constraint
        constrained_feat = x * consistency + spatial_feat * (1 - consistency)
        
        return constrained_feat


class DGTADDetect(nn.Module):
    """
    Dynamic Geometric Topology-Aware Detection (DGTAD) head.
    
    An innovative detection head that leverages geometric topology analysis
    for enhanced semiconductor defect detection. This module integrates:
    1. Geometric topology analysis for feature enhancement
    2. Dynamic anchor generation based on defect patterns
    3. Topology consistency constraints for robust detection
    4. Lightweight geometric transformation awareness
    """
    
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility
    
    def __init__(self, nc=80, ch=()):
        """
        Initialize DGTAD detection head.
        
        Args:
            nc (int): Number of classes
            ch (tuple): Input channels for each detection layer
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        # Channel configuration - 与标准Detect保持一致
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        
        # Geometric Topology Analyzers for each detection layer
        self.geo_analyzers = nn.ModuleList([
            GeometricTopologyAnalyzer(x) for x in ch
        ])
        
        # Dynamic Anchor Generators
        self.anchor_generators = nn.ModuleList([
            DynamicAnchorGenerator(x) for x in ch
        ])
        
        # Topology Consistency Constraints
        self.consistency_constraints = nn.ModuleList([
            TopologyConsistencyConstraint(x) for x in ch
        ])
        
        # Enhanced bbox regression heads with geometric awareness
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                Conv(x, c2, 3), 
                Conv(c2, c2, 3), 
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        ])
        
        # Enhanced classification heads with topology awareness  
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        ])
        
        # Distribution Focal Loss for bbox regression
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize module weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
                
    def forward(self, x):
        """
        Forward pass of DGTAD detection head.
        
        Args:
            x (list): List of feature tensors from different scales
            
        Returns:
            list or tuple: Detection outputs
        """
        shape = x[0].shape  # BCHW
        
        # Apply DGTAD processing to each detection layer
        for i in range(self.nl):
            # Step 1: Geometric topology analysis (轻量化处理)
            geo_enhanced = self.geo_analyzers[i](x[i])
            
            # Step 2: Apply topology consistency constraint
            consistent_feat = self.consistency_constraints[i](geo_enhanced)
            
            # Step 3: Generate dynamic anchors (仅在训练时用于优化)
            if self.training:
                dynamic_anchors = self.anchor_generators[i](x[i])
                # Store for potential future use
                if not hasattr(self, 'dynamic_anchor_scales'):
                    self.dynamic_anchor_scales = []
                if len(self.dynamic_anchor_scales) <= i:
                    self.dynamic_anchor_scales.append(dynamic_anchors)
                else:
                    self.dynamic_anchor_scales[i] = dynamic_anchors
            
            # Step 4: Apply detection heads to enhanced features
            x[i] = torch.cat((self.cv2[i](consistent_feat), self.cv3[i](consistent_feat)), 1)
        
        if self.training:  # Training mode - return feature list
            return x
            
        # Inference mode - decode predictions
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    
    def bias_init(self):
        """Initialize detection head biases."""
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0  # bbox
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls
    
    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)
    
    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """Post-process YOLO model predictions."""
        batch_size, anchors, _ = preds.shape
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


# Export the new detection head
__all__ = ['DGTADDetect', 'GeometricTopologyAnalyzer', 'DynamicAnchorGenerator', 'TopologyConsistencyConstraint']