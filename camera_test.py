import cv2
from ultralytics import YOLO
import torch

# 加載你剛下載好的模型
model = YOLO('yolov8n.pt')

# 使用 M1 GPU 加速
if torch.backends.mps.is_available():
    model.to('mps')

# 開啟視訊鏡頭 (source="0" 通常是內建鏡頭)
# show=True 會直接彈出視窗，stream=True 則能節省記憶體
# 嘗試將 "0" 改為 0 (不加引號)，或者改為 1
results = model.predict(source=0, show=True, stream=True)

# 讓程式持續跑，直到你按下鍵盤上的 'q'
for r in results:
    print(f"FPS: {1000 / r.speed['inference']:.1f}") # 計算每秒跑幾張
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

