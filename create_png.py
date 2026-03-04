import cv2
from picamera2 import Picamera2
import time
import os

os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Where to save the new real-world templates
save_dir = '/home/jaydenbryan/Project/Symbols_png/'

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("=== TEMPLATE CAPTURE TOOL ===")
print("1. Click the Video Window.")
print("2. Hold a symbol inside the green box.")
print("3. Press 's' to snap a photo.")

try:
    while True:
        frame = picam2.capture_array()
        display_frame = frame.copy()
        
        # Define the center targeting box (250x250 pixels)
        center_x, center_y = 320, 240
        box_size = 125
        x1, y1 = center_x - box_size, center_y - box_size
        x2, y2 = center_x + box_size, center_y + box_size
        
        # Draw the HUD
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, "Hold symbol in box. Press 's' to save.", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Capture Tool", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Safe Quit
        if key == ord('q') or cv2.getWindowProperty("Capture Tool", cv2.WND_PROP_VISIBLE) < 1:
            break
            
        # Snap Photo
        elif key == ord('s'):
            print("\n>>> PHOTO SNAPPED! Look at your terminal.")
            # Crop exactly what is inside the green box
            cropped_symbol = frame[y1:y2, x1:x2]
            
            # Ask you what to name it
            filename = input("Type the filename (e.g., symbol_qrcode.png) and press Enter: ")
            
            # Make sure it ends with .png
            if not filename.endswith(".png"):
                filename += ".png"
                
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, cropped_symbol)
            print(f"SUCCESS: Saved {filepath}!")
            print("Ready for the next one. Click the video window again!")

finally:
    picam2.stop()
    cv2.destroyAllWindows()