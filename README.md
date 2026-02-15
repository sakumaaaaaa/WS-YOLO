# YOLO11-CMConv-MSAFE-TSA

基于 YOLO11 的半导体晶圆缺陷检测改进模型。本项目在 Ultralytics YOLO11 框架上引入三个自研模块，针对晶圆表面微小缺陷的检测场景进行优化。

## 改进模块

CMConv（Cross-Channel Mixing Convolution）：跨通道混合卷积模块，采用 C3k2 风格的 CSP 结构，内置 CCAM（Cross-Channel Attention Module）。CCAM 通过可学习相位参数实现通道间信息混合，结合通道注意力与空间注意力进行特征筛选。该模块替换原始 backbone 和 neck 中的 C3k2 模块，在不显著增加计算量的前提下增强通道间的特征交互。

MSAFE（Multi-Scale Attention Feature Enhancement）：多尺度注意力特征增强模块，由 SE 通道注意力、双分支膨胀卷积（dilation=1,2）特征提取、深度可分离卷积残差增强三部分级联组成。该模块仅部署在 P3/8 小目标检测层，以集中增强小尺度缺陷的特征表达，同时控制额外开销。

TSADetect（Triple-Scale Adaptive Detection Head）：三尺度自适应检测头，替换原始 Detect 头。包含三个子组件：SmallTargetEnhancer 通过高频卷积分支和通道注意力增强小目标特征；AdaptiveFeatureAlignment 基于可变形卷积思想通过可学习偏移量实现特征对齐；多尺度双向融合采用 Top-Down softmax 加权与 Bottom-Up 残差注入的方式聚合三个尺度的特征。

## 环境配置

```bash
# 克隆项目
git clone <repo_url>
cd ultralytics-wafer

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装（推荐）
pip install -e .
```

Python >= 3.8，PyTorch >= 1.8.0。

## 数据集

晶圆缺陷检测数据集，包含 6 类缺陷：scratch、edge_bite、stains_embedded、gray_line、open、short_circuit。

链接：https://pan.baidu.com/s/1hO--mnHVe2jRG_Kg_U5GbQ 提取码: 1111 

数据集目录结构：

```
data/
├── wafer.yaml          # 数据集配置
└── Wafer/
    ├── images/         # 原始图像
    ├── labels/         # YOLO 格式标注
    ├── train.txt       # 训练集列表
    └── val.txt         # 验证集列表
```

使用前需修改 `data/wafer.yaml` 中的 `path` 字段为实际数据集路径。

## 训练

```bash
python train.py
```

默认配置：输入分辨率 640、batch size 16、SGD 优化器、训练 300 epoch。如需调整，直接编辑 `train.py` 或使用 Python API：

```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/11/yolo11-cmconv-msafe-tsa.yaml')
model.load('yolo11n.pt')  # 可选：加载预训练权重
model.train(data='data/wafer.yaml', epochs=300, imgsz=640, batch=16, device='0')
```

训练结果保存在 `runs/train/exp/` 目录下。

## 模型配置

本项目提供多种模块组合的消融实验配置，位于 `ultralytics/cfg/models/11/`：

| 配置文件 | 模块组合 |
|---------|---------|
| yolo11-cmconv-msafe-tsa.yaml | CMConv + MSAFE + TSA（完整模型） |
| yolo11-cmconv-msafe.yaml | CMConv + MSAFE |
| yolo11-cmconv-tsa.yaml | CMConv + TSA |
| yolo11-msafe-tsa.yaml | MSAFE + TSA |
| yolo11-cmconv.yaml | 仅 CMConv |
| yolo11-msafe.yaml | 仅 MSAFE |
| yolo11-tsa.yaml | 仅 TSA |
| yolo11-baseline.yaml | YOLO11 原始基线 |

切换配置只需修改 `train.py` 中的 yaml 路径即可。

## 项目结构

```
ultralytics-wafer/
├── ultralytics/
│   ├── nn/modules/
│   │   ├── quantum_attention_lite.py   # CMConv、MixingGate、CCAM
│   │   ├── msafe_module.py             # MSAFE、MSAFEBlock
│   │   └── tsa_detect.py              # TSADetect、SmallTargetEnhancer、AdaptiveFeatureAlignment
│   ├── cfg/models/11/                  # 模型配置（含消融实验）
│   └── ...                             # Ultralytics 框架原始代码
├── data/
│   ├── wafer.yaml                      # 数据集配置
│   └── Wafer/                          # 数据集
├── train.py                            # 训练入口
├── requirements.txt                    # 依赖列表
└── pyproject.toml                      # 包配置
```

## License

AGPL-3.0
