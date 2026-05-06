# utils/visualizer.py
import cv2

def draw_info(image, count, fps, settings):
    """Render detection count and FPS metadata onto the video frame"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Render object count overlay
    cv2.putText(image, f"People Count: {count}", (20, 50), 
                font, settings.FONT_SCALE, settings.COLOR_COUNT, settings.THICKNESS)
    
    # Render FPS (Frames Per Second) counter
    cv2.putText(image, f"FPS: {fps:.1f}", (20, 90), 
                font, settings.FONT_SCALE, settings.COLOR_FPS, settings.THICKNESS)
    
    return image