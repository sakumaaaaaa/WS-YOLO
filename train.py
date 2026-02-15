import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11-CMConv-msafe-tsa.yaml')
    # model.load('yolo11n.pt')  # 可选：加载预训练权重
    model.train(data='data/wafer.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                workers=2,
                device='0',
                optimizer='SGD',
                project='runs/train',
                name='exp',
                verbose=True,
                plots=True,
                )
