# main.py
import cv2
from core.detector import YOLODetector
from utils.visualizer import draw_info
from configs import settings

def main():
    # Initialization
    detector = YOLODetector(settings.MODEL_NAME, settings.DEVICE)
    img_counter = 0
    
    print(f"✅ System started using {settings.DEVICE}. Press 's' to save, 'q' to quit.")

    # Execute Detection
    try:
        results = detector.get_results(settings.SOURCE)
        
        for r in results:
            if r is None: continue
            
            # Process inference data
            count, fps, img_with_boxes = detector.process_frame(r, settings.TARGET_CLASS_ID)
            
            # Render UI overlays
            final_img = draw_info(img_with_boxes, count, fps, settings)
            
            # Display output window
            cv2.imshow("Real-Time Object Detection & Tracking System", final_img)
            
            # Listen for keyboard events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.imwrite(f"screenshot_{img_counter}.png", final_img)
                img_counter += 1
            elif key == ord('q'):
                break
                
    except Exception as e:
        print(f"⚠️ Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()