from ultralytics import YOLO
import torch

# Load the model (yolov8n.pt will be downloaded automatically on first execution)
model = YOLO('yolov8n.pt')

# Check for M1 GPU (MPS) and move the model to the device
if torch.backends.mps.is_available():
    model.to('mps')
    print("Using M1 GPU (MPS)!")

# Perform inference (using an example image)
# This will fetch a bus image from the web for detection
results = model('https://ultralytics.com/images/bus.jpg')

# Display the results
results[0].show()
