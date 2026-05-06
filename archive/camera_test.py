import cv2
from ultralytics import YOLO
import torch

# Load the pre-trained model
model = YOLO('yolov8n.pt')

# Enable M1 GPU acceleration (MPS)
if torch.backends.mps.is_available():
    model.to('mps')

# Initialize the webcam (source=0 is typically the built-in camera)
# show=True opens a display window; stream=True optimizes memory usage
# Note: Use integer 0 instead of string "0" for macOS compatibility
results = model.predict(source=0, show=True, stream=True)

# Keep the program running until the 'q' key is pressed
for r in results:
    print(f"FPS: {1000 / r.speed['inference']:.1f}") # Calculate and display the Frames Per Second (FPS)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

