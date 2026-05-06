# core/detector.py
from ultralytics import YOLO
import os

class YOLODetector:
    def __init__(self, model_name, device):
        if not os.path.exists(model_name):
            raise FileNotFoundError(f"Model file '{model_name}' not found.")
        
        self.model = YOLO(model_name)
        self.device = device
        self.model.to(self.device)

    def get_results(self, source):
        """Generator that yields inference results"""
        return self.model.predict(source=source, show=False, stream=True)

    def process_frame(self, result, target_id):
        """Parse detection data from a single frame"""
        class_ids = result.boxes.cls.int().tolist()
        count = class_ids.count(target_id)
        fps = 1000 / result.speed['inference']
        return count, fps, result.plot()