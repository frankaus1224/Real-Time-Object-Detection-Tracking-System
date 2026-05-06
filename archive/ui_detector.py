import cv2
import torch
import time
from collections import defaultdict
from ultralytics import YOLO

class UIDetector:
    COLORS = [
        (0, 255, 0),    (255, 0, 0),    (0, 0, 255),    (255, 255, 0),
        (0, 255, 255),  (255, 0, 255),  (128, 255, 0),  (255, 128, 0),
        (0, 128, 255),  (128, 0, 255),  (255, 0, 128),  (0, 255, 128),
        (64, 224, 208), (255, 165, 0),  (138, 43, 226), (220, 20, 60),
    ]

    def __init__(self, model_name='yolov8n.pt'):
        print("Initializing UI Detector...")
        self.model = YOLO(model_name)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Model running on: {self.device}")

        # FPS tracking
        self.prev_time = time.time()
        self.fps = 0.0

    def count_objects(self, result):
        
        counts = defaultdict(int)

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                counts[class_name] += 1

        return dict(counts)

    def _get_color(self, class_id):
        return self.COLORS[class_id % len(self.COLORS)]

    def _calculate_fps(self):
        current_time = time.time()
        elapsed = current_time - self.prev_time
        self.fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self.prev_time = current_time
        return self.fps

    def draw_bounding_boxes(self, frame, result):
        
        if result.boxes is None:
            return frame

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            color = self._get_color(class_id)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

            label = f"{class_name} {confidence * 100:.0f}%"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_y = max(y1, text_h + 6)
            cv2.rectangle(
                frame,
                (x1, label_y - text_h - 6),
                (x1 + text_w + 4, label_y + baseline - 4),
                color,
                thickness=-1  
            )
            cv2.putText(
                frame, label,
                (x1 + 2, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0),  
                thickness=1, lineType=cv2.LINE_AA
            )

        return frame

    def draw_fps(self, frame):

        fps_text = f"FPS: {self.fps:.1f}"

        cv2.rectangle(frame, (8, 8), (130, 38), (0, 0, 0), -1)
        cv2.putText(
            frame, fps_text,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 80),
            thickness=2, lineType=cv2.LINE_AA
        )
        return frame

    def draw_object_counts(self, frame, counts):
       
        if not counts:
            return frame

        h, w = frame.shape[:2]
        panel_x = w - 200
        panel_y = 10
        line_height = 26
        padding = 8

        total = sum(counts.values())
        lines = [f"Total: {total}"] + [f"  {name}: {cnt}" for name, cnt in sorted(counts.items())]
        panel_height = len(lines) * line_height + padding * 2

        cv2.rectangle(
            frame,
            (panel_x - padding, panel_y),
            (w - 6, panel_y + panel_height),
            (20, 20, 20),
            -1
        )
        cv2.rectangle(
            frame,
            (panel_x - padding, panel_y),
            (w - 6, panel_y + panel_height),
            (80, 80, 80),
            1
        )

        for i, line in enumerate(lines):
            color = (255, 255, 255) if i == 0 else (180, 220, 255)
            font_scale = 0.6 if i == 0 else 0.55
            thickness = 2 if i == 0 else 1
            cv2.putText(
                frame, line,
                (panel_x, panel_y + padding + (i + 1) * line_height - 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, thickness=thickness, lineType=cv2.LINE_AA
            )

        return frame

    def run(self, source=0):

        print("Starting UI Detector... Press 'Q' to quit.")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("ERROR: Could not open webcam. Check camera permissions.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to grab frame.")
                break
            results = self.model.predict(
                source=frame,
                stream=False,       
                verbose=False
            )
            result = results[0]

            counts = self.count_objects(result)

            self._calculate_fps()
            frame = self.draw_bounding_boxes(frame, result)
            frame = self.draw_fps(frame)
            frame = self.draw_object_counts(frame, counts)

            cv2.imshow("UI Detector - Real-Time Object Detection", frame)

            print(f"FPS: {self.fps:.1f} | Detections: {counts}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit signal received.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped. Resources released.")

if __name__ == "__main__":
    detector = UIDetector(model_name='yolov8n.pt')
    detector.run(source=0)
