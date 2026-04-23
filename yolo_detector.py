import cv2
import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name='yolov8n.pt'):
        """
        Step 1: Initialize the model and hardware acceleration.
        """
        print("Initializing YOLO Detector...")
        self.model = YOLO(model_name)
        
        # Check for M1 GPU (MPS)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Model moved to device: {self.device}")

    def run_detection(self, source=0):
        """
        Step 2: Start real-time detection from webcam.
        """
        # We use stream=True to process frames one by one and save memory
        results = self.model.predict(source=source, show=True, stream=True)
        
        print("Detection started. Press 'q' on the pop-up window to stop.")
        
        for r in results:
            # Calculate FPS from inference speed
            inference_time = r.speed['inference']
            fps = 1000 / inference_time
            print(f"FPS: {fps:.1f}")
            
            # Press 'q' to exit the display window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Proper cleanup
        cv2.destroyAllWindows()

# This part ensures the code only runs if the script is executed directly
if __name__ == "__main__":
    detector = YOLODetector()
    detector.run_detection(source=0)