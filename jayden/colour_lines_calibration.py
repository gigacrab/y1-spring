import cv2
import numpy as np
from picamera2 import Picamera2
import time

def click_to_get_hsv(event, x, y, flags, hsv_frame):
    if event == cv2.EVENT_LBUTTONDOWN:
        h, s, v = hsv_frame[y, x]
        
        print("\n" + "="*40)
        print(f"Clicked Pixel at (X:{x}, Y:{y})")
        print(f"Exact HSV Value: [{h}, {s}, {v}]")
        
        h_lower = max(0, h - 10)
        h_upper = min(179, h + 10)
        
        s_lower = max(0, s - 50)
        s_upper = min(255, s + 50)
        
        v_lower = max(0, v - 50)
        v_upper = min(255, v + 50)
        
        print("-" * 40)
        print("COPY AND PASTE THESE LINES INTO YOUR MAIN CODE:")
        print(f"lower_color = np.array([{h_lower}, {s_lower}, {v_lower}])")
        print(f"upper_color = np.array([{h_upper}, {s_upper}, {v_upper}])")
        print("="*40 + "\n")

picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

cv2.namedWindow('HSV Color Picker')

print("Starting camera... Point it at the colored line and click on it!")
print("Press 'ESC' on your keyboard to quit.")

try:
    while True:
        frame = picam2.capture_array()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.setMouseCallback('HSV Color Picker', click_to_get_hsv, hsv_frame)
        cv2.drawMarker(frame, (320, 240), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow('HSV Color Picker', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except (KeyboardInterrupt, Exception) as e:
    print(f"Exiting: {e}")

finally:
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()