# configs/settings.py
import torch

# Model & Hardware Configuration
MODEL_NAME = 'yolov8n.pt'
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Detection Settings
TARGET_CLASS_ID = 0  # ID 0 represents 'person' in COCO dataset
SOURCE = 0           # 0 is the default system webcam

# UI & Visualization Settings
FONT_SCALE = 1.0
THICKNESS = 2
COLOR_COUNT = (0, 255, 0)  # Green 
COLOR_FPS = (255, 128, 0)  # Orange-Blue