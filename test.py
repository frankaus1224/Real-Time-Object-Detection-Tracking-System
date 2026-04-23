from ultralytics import YOLO
import torch

# 1. 加載模型 (第一次執行會自動下載 yolov8n.pt)
model = YOLO('yolov8n.pt')

# 2. 檢查是否有 M1 GPU (MPS) 並移動模型
if torch.backends.mps.is_available():
    model.to('mps')
    print("Using M1 GPU (MPS)!")

# 3. 執行推論 (使用範例圖片)
# 這會從網路抓取一張巴士的照片來跑偵測
results = model('https://ultralytics.com/images/bus.jpg')

# 4. 顯示結果
results[0].show()
