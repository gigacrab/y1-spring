import cv2
import numpy as np
from picamera2 import Picamera2
import time

# 1. Load the Master DNA you just created
try:
    master_dna = np.load('arrow_template.npy')
except:
    print("Error: Run create_template.py first to generate arrow_template.npy!")
    exit()

# 2. Setup Camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)

def get_color_name(hsv_frame, contour):
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val = cv2.mean(hsv_frame, mask=mask)
    hue = mean_val[0]
    
    # Simple Hue mapping for your specific project colors
    if (hue < 10) or (hue > 160): return "Red"
    if 15 < hue < 35: return "Orange/Yellow"
    if 40 < hue < 85: return "Green"
    if 90 < hue < 130: return "Blue"
    return "Unknown"

while True:
    frame = picam2.capture_array()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu to isolate shapes
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < 1500: continue # Ignore small noise

        # Calculate Hu Moments for the live shape
        live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
        
        # Compare live DNA to Master DNA (Logarithmic comparison is more stable)
        # Using a distance threshold (lower is a better match)
        distance = cv2.matchShapes(c, c, cv2.CONTOURS_MATCH_I1, 0) # Placeholder for direct logic
        # For Hu Moments, we manually check the distance between the 7 values:
        diff = np.sum(np.abs(live_moments - master_dna))

        if diff < 0.1: # Adjust this threshold based on testing
            color = get_color_name(hsv, c)
            
            # Draw the result
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{color} Arrow", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Recognition System 2026", frame)
    if cv2.waitKey(1) == ord('q'): break

picam2.stop()
cv2.destroyAllWindows()