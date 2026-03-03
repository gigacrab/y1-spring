import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_PLATFORM"] = "offscreen" # Change to "xcb" if you want to see the video window!

try:
    arrow_dna = np.load('arrow.npy')
    circle_dna = np.load('circle34.npy')
except FileNotFoundError:
    print("Error: Could not find the .npy files.")
    exit()

# --- 2. START THE CAMERA ---
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("Looking for symbols...")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Isolate the shapes (Black and White)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) > 1500: # Ignore tiny background noise
                
                # Calculate Hu Moments for the shape the camera is looking at right now
                live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                
                # Compare live DNA to your saved templates
                arrow_diff = np.sum(np.abs(live_moments - arrow_dna))
                circle_diff = np.sum(np.abs(live_moments - circle_dna))
                
                # Check which one is the closest match (lowest difference is better)
                if arrow_diff < 0.1 and arrow_diff < circle_diff:
                    label = "Arrow Detected"
                elif circle_diff < 0.1 and circle_diff < arrow_diff:
                    label = "3/4 Circle Detected"
                else:
                    label = "Unknown Shape"
                    
                # Print the result to the terminal!
                if label != "Unknown Shape":
                    print(label)
                    
                    # Optional: Draw a box if you are viewing the window
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # cv2.imshow("Robot View", frame) # Uncomment if you fixed the window/font stuff!
        # if cv2.waitKey(1) == ord('q'): break

finally:
    picam2.stop()
    cv2.destroyAllWindows()