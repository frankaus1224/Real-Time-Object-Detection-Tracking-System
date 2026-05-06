import cv2
import torch
from ultralytics import YOLO
import os

class YOLODetector:
    def __init__(self, model_name='yolov8n.pt'):
        """
        Step 1: Initialize the model and check if the file exists.
        """
        print("Initializing YOLO Detector...")
        
        # 錯誤處理：檢查模型檔案是否存在
        if not os.path.exists(model_name):
            raise FileNotFoundError(f"❌ Error: Model file '{model_name}' not found. Please ensure the file exists.")

        try:
            self.model = YOLO(model_name)
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            self.model.to(self.device)
            print(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def run_detection(self, source=0):
        """
        Step 2: Start detection with safety checks for Webcam.
        """
        # 錯誤處理：檢查相機是否能開啟
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {source}. Is the webcam connected?")
            return
        
        # 釋放測試用的 cap，交給 YOLO 處理
        cap.release()

        print("Detection started. Press 's' to save, 'q' to quit.")
        img_counter = 0

        try:
            # 使用 stream=True
            results = self.model.predict(source=source, show=False, stream=True)
            
            for r in results:
                if r is None:
                    print("⚠️ Warning: Received empty frame.")
                    continue
                
                img_with_boxes = r.plot()
                
                # 邏輯計算
                class_ids = r.boxes.cls.int().tolist()
                person_count = class_ids.count(0)
                fps = 1000 / r.speed['inference']
                
                # UI 繪製
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_with_boxes, f"People Count: {person_count}", (20, 50), 
                            font, 1.0, (0, 255, 0), 2)
                cv2.putText(img_with_boxes, f"FPS: {fps:.1f}", (20, 90), 
                            font, 1.0, (255, 128, 0), 2)

                cv2.imshow("YOLO Real-Time Detection System", img_with_boxes)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    img_name = f"screenshot_{img_counter}.png"
                    cv2.imwrite(img_name, img_with_boxes)
                    print(f"✅ Screenshot saved: {img_name}")
                    img_counter += 1
                elif key == ord('q'):
                    break
                    
        except Exception as e:
            print(f"⚠️ An unexpected error occurred during detection: {e}")
        finally:
            # 確保無論如何都會關閉視窗
            cv2.destroyAllWindows()
            print("Program finished safely.")
            
if __name__ == "__main__":
    detector = YOLODetector()
    detector.run_detection(source=0)